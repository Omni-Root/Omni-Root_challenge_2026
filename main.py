"""
main.py — Inspeção de toras na máquina de campo (Windows Embedded / maquete)
Projeto: Qualidade da Madeira — Challenge FIAP x John Deere/Suzano

O que este script faz, em ordem:
  1. Captura um frame da câmera acoplada à garra do harvester
  2. Recorta a região de interesse (ROI) onde a tora sempre aparece —
     a câmera é fixa, então o resto do frame é fundo e pode ser ignorado
  3. Roda o modelo YOLO (8 classes de defeito) no recorte pré-processado
     (grayscale + CLAHE, igual ao treino)
  4. Filtra falsos positivos (área mínima, dentro do contorno, persistência)
  5. Converte pixel -> cm real usando uma distância câmera-tora FIXA e
     calibrada (não sensor físico)
  6. Calcula os 8 indicadores de qualidade — densidade vem de um lookup por
     clone/material genético (data/clones_densidade.json, sincronizado do
     PostgreSQL central), NÃO de sensor físico
  7. Classifica a tora (aprovado / quarentena / reprovado) ponderando a
     gravidade de cada defeito para a indústria de celulose
  8. Grava tudo no SQLite local (schema_sqlite.sql), pronto para o
     sync_daemon.py sincronizar com o PostgreSQL quando houver rede

Modo de uso:
  python main.py --gui                       # câmera do config.json
  python main.py --gui --fonte video.mp4     # vídeo gravado (plano B na demo)
  python main.py --gui --fonte ./fotos/      # pasta de imagens, em loop

Dependências (requirements.txt):
  ultralytics
  opencv-python (ou opencv-python-headless sem --gui)
  numpy
"""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from omniroot.banco_local import migrar_banco_local  # noqa: F401  (compatibilidade: testes importam de main)
from omniroot.luz import RealceLuz, descrever_luz, nivel_do_evento  # noqa: F401
from omniroot.posicao import LeitorPosicao

# ------------------------------------------------------------
# O código da máquina vive no pacote omniroot/ (um módulo por
# responsabilidade). Este arquivo é o PONTO DE ENTRADA: argumentos,
# threads e o loop de captura. Os nomes abaixo continuam importáveis
# de `main` por compatibilidade (calibrar.py, exportar_modelo.py e os
# testes fazem `from main import ...`).
# ------------------------------------------------------------
from omniroot.config import (  # noqa: F401
    Config,
    carregar_configuracao_json,
)
from omniroot.inventario import (  # noqa: F401
    _ARQUIVOS_INVENTARIO,
    _INVENTARIO,
    _INVENTARIO_LOCK,
    _assinatura_inventario,
    calcular_densidade_estimada,
    carregar_inventario_florestal,
    inventario_atual,
)
from omniroot.deteccao import (  # noqa: F401
    CONF_POR_CLASSE_PADRAO,
    FiltroPersistencia,
    _CLAHE,
    carregar_modelo,
    deslocar_defeitos,
    detectar_yolo,
    extrair_defeitos_yolo,
    filtrar_dentro_do_contorno,
    filtrar_por_area_minima,
    limiar_da_classe,
    preprocessar_para_modelo,
)
from omniroot.medidas import (  # noqa: F401
    CASCA_FRACAO_BRILHO,
    _ARUCO_DETECTOR,
    _ULTIMO_AVISO_CASCA,
    _detector_aruco,
    calcular_cm_por_px,
    calcular_dimensao_real_cm,
    calcular_massa_seca_kg,
    calcular_porcentagem_casca,
    calcular_tortuosidade,
    calcular_volume_m3,
    detectar_rachaduras_secao,
    escala_por_marcador,
)
from omniroot.segmentacao import (  # noqa: F401
    _componente_central,
    _mascara_do_contorno,
    balancear_branco,
    classificar_vista,
    extrair_contorno_tora,
    fundo_e_escuro,
    mascara_interior,
    recortar_roi,
    roi_em_pixels,
    segmentar_fundo_escuro,
    segmentar_por_calor,
    segmentar_por_fundo,
    segmentar_tora,
    segmentar_tora_origem,
    validar_tora,
)
from omniroot.classificacao import (  # noqa: F401
    DEFEITOS_GRAVES,
    DEFEITOS_LEVES,
    DEFEITOS_MODERADOS,
    PESO_PADRAO,
    PESO_SEVERIDADE,
    calcular_confianca_saude,
    classificar_qualidade,
    gerar_hash_sha256,
    severidade_do_defeito,
)
from omniroot.registro import (  # noqa: F401
    CAMPOS_POSICAO,
    _hash_inspecao,
    atualizar_inspecao,
    conectar_banco,
    salvar_inspecao,
)
from omniroot.hud import (  # noqa: F401
    CORES_STATUS,
    COR_IGNORADO,
    desenhar_analise,
)
from omniroot.analise import (  # noqa: F401
    analisar_frame,
    descrever_posicao,
    resumir_analise,
)
from omniroot.evento import (  # noqa: F401
    RastreadorTora,
    _bbox_contorno,
    _iou_bbox,
    consolidar_defeitos,
    consolidar_evento,
)
from omniroot.stream import (  # noqa: F401
    _token_stream,
    codificar_quadro_stream,
    enviar_quadro_stream,
    loop_stream,
)
from omniroot.fontes import (  # noqa: F401
    FonteImagens,
    abrir_fonte,
)


def __getattr__(nome: str):
    """`main.INVENTARIO_FLORESTAL` devolve o valor ATUAL (é recarregado em omniroot.inventario)."""
    if nome == "INVENTARIO_FLORESTAL":
        from omniroot import inventario
        return inventario.INVENTARIO_FLORESTAL
    raise AttributeError(nome)


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="IA de Qualidade da Madeira — máquina de campo")
    parser.add_argument(
        "--simulado", action="store_true",
        help="Abre a janela com bounding boxes (mesmo efeito de --gui, mantido por compatibilidade)"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Abre janela gráfica exibindo as bounding boxes da câmera ao vivo"
    )
    parser.add_argument(
        "--fonte", type=str, default=None,
        help="Índice de câmera, arquivo de vídeo ou pasta de imagens. Padrão: camera_index do config.json"
    )
    parser.add_argument(
        "--config-file", type=str, default="./config.json",
        help="Caminho do arquivo JSON de configuração da máquina/operação"
    )
    parser.add_argument(
        "--clone", type=str, default=None,
        help="Sobrescreve o Código do Clone/Material Genético (ex: SP3108, SP2974, CO41H_TEST)"
    )
    parser.add_argument(
        "--idade", type=float, default=None,
        help="Sobrescreve a Idade do talhão em anos na colheita (ex: 5.3)"
    )
    parser.add_argument(
        "--gnss", type=str, nargs="?", const="windows", default=None,
        help="Fonte de posição: porta serial NMEA (ex: COM5), arquivo .nmea gravado, ou 'windows' "
             "(Localização do Windows, no notebook da maquete; é o que `--gnss` sozinho usa). "
             "Sobrescreve o config.json"
    )
    args = parser.parse_args()

    cfg = carregar_configuracao_json(args.config_file)
    if args.clone is not None:
        cfg.clone_id = args.clone
    if args.idade is not None:
        cfg.idade_talhao_anos = args.idade
    if args.gnss is not None:
        if args.gnss.strip().lower() != "windows" and Path(args.gnss).is_file():
            cfg.gnss_arquivo, cfg.gnss_porta = args.gnss, ""
        else:
            cfg.gnss_arquivo, cfg.gnss_porta = "", args.gnss

    # Posição (GNSS): thread própria; a inspeção só lê a última posição válida.
    origem_gnss = (cfg.gnss_arquivo or cfg.gnss_porta or "").strip()
    if cfg.gnss_arquivo and not Path(cfg.gnss_arquivo).is_file():
        print(f"⚠️  gnss_arquivo '{cfg.gnss_arquivo}' não encontrado — toras serão gravadas sem posição.")
        origem_gnss = ""
    leitor_gnss = LeitorPosicao(origem_gnss, cfg.gnss_baud).iniciar() if origem_gnss else None
    if leitor_gnss is None:
        print("ℹ️  Sem fonte de posição (gnss_porta/gnss_arquivo vazios): toras gravadas sem posição.")

    def posicao_atual() -> dict | None:
        return leitor_gnss.atual(cfg.gnss_validade_s) if leitor_gnss is not None else None

    print("📦 Carregando modelo IA...")
    modelo = carregar_modelo(cfg)

    captura = abrir_fonte(args.fonte, cfg)
    if not captura.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo ({args.fonte or cfg.camera_index}).")
    fonte_e_video = bool(args.fonte) and Path(args.fonte).is_file()

    # Sem cm_por_px calibrado, avisa uma vez: as medidas em cm são só ordem de grandeza.
    if not (cfg.cm_por_px and cfg.cm_por_px > 0):
        print("ℹ️  'cm_por_px' não calibrado no config.json — diâmetro/comprimento usam a fórmula de GSD com óptica genérica de webcam (ordem de grandeza).")
    if cfg.roi:
        print(f"🎯 ROI ativa: {cfg.roi} (fração do frame). Detecções fora dela são ignoradas.")

    # ------------------------------------------------------------------
    # Arquitetura em tempo real (a imagem nunca congela):
    #   - Thread PRINCIPAL: só captura e exibe o vídeo (roda a ~FPS da
    #     câmera, sempre fluido).
    #   - Thread de TRABALHO: roda o pipeline no frame mais recente e devolve
    #     a análise para a principal desenhar. A inferência pesada nunca
    #     bloqueia o vídeo — no máximo as caixas atualizam um pouco atrás.
    # ------------------------------------------------------------------
    # Três threads:
    #   - PRINCIPAL: captura e exibe (fps da câmera).
    #   - YOLO: roda o modelo no frame mais recente, no ritmo que a CPU der
    #     (~0,5 s com OpenVINO, ~2,5 s com .pt a 1024). Publica as caixas.
    #   - ANÁLISE: visão clássica (~70 ms: contorno, casca, diâmetro,
    #     rachadura, status) a ~10 fps, reaproveitando as últimas caixas do
    #     YOLO. É o que faz a tela responder na hora; só os nós do modelo
    #     atualizam com atraso.
    estado = {"frame": None, "analise": None, "yolo": None, "yolo_id": 0, "fundo": None, "luz": None}

    # Pouca luz: medido e realçado na captura, ANTES de tudo — o modelo, a
    # visão clássica, o fundo de referência e a câmera ao vivo recebem o
    # mesmo quadro. Ver omniroot/luz.py.
    realce = RealceLuz(
        ligado=cfg.luz_realce,
        limiar_boa=cfg.luz_limiar_boa,
        limiar_critica=cfg.luz_limiar_critica,
        ruido_baixa=cfg.luz_ruido_baixa,
        ruido_critico=cfg.luz_ruido_critico,
        ganho_max=cfg.luz_ganho_max,
        empilhar_max=cfg.luz_empilhar_max,
    )
    ultimo_nivel_luz = {"v": None}
    lock = threading.Lock()
    parar = threading.Event()

    def worker_yolo():
        while not parar.is_set():
            with lock:
                frame = None if estado["frame"] is None else estado["frame"].copy()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                recorte, _ = recortar_roi(frame, cfg)
                defeitos = detectar_yolo(recorte, modelo, cfg)
                with lock:
                    estado["yolo"] = defeitos
                    estado["yolo_id"] += 1
            except Exception as e:
                print(f"⚠️  Erro no modelo (seguindo em frente): {e}")
                time.sleep(0.05)

    def worker_analise():
        # A conexão SQLite é criada AQUI: objetos sqlite3 só podem ser usados
        # na mesma thread em que foram criados.
        conexao = conectar_banco(cfg)
        ultima_gravacao = 0.0
        ultimo_yolo_id = -1
        # O filtro de persistência guarda estado entre análises, então
        # precisa viver fora do loop.
        persistencia = FiltroPersistencia(min_ocorrencias=max(1, cfg.filtro_persistencia))
        # Uma tora = um registro (ver RastreadorTora). No modo "intervalo"
        # fica None e o loop grava a cada intervalo_captura_seg como antes.
        rastreador = RastreadorTora(cfg) if cfg.modo_gravacao == "evento" else None
        try:
            while not parar.is_set():
                with lock:
                    frame = None if estado["frame"] is None else estado["frame"].copy()
                    defeitos_yolo = estado["yolo"]
                    yolo_id = estado["yolo_id"]
                    fundo = estado["fundo"]
                    luz = estado.get("luz")
                if frame is None or defeitos_yolo is None:
                    time.sleep(0.01)
                    continue
                try:
                    a = analisar_frame(
                        frame, None, cfg, persistencia,
                        defeitos_yolo=defeitos_yolo, yolo_novo=(yolo_id != ultimo_yolo_id),
                        fundo=fundo,
                    )
                    ultimo_yolo_id = yolo_id

                    # Luz do quadro analisado (medida na captura, ver omniroot/luz.py).
                    # Nunca recusa por luz: o nível só vai junto para a
                    # proveniência (consolidar_evento) e para o HUD.
                    a["luz"] = luz["nivel"] if luz else "boa"
                    if luz is not None:
                        a["hud"].append(descrever_luz(luz))

                    if leitor_gnss is not None:
                        a["hud"].append(leitor_gnss.descricao(cfg.gnss_validade_s))

                    if rastreador is not None:
                        fechado = rastreador.atualizar(a)
                        a["hud"].append(rastreador.descricao())
                        with lock:
                            estado["analise"] = a
                        agora = time.monotonic()
                        if fechado is not None:
                            # Fechou: última consolidação no MESMO registro.
                            if fechado.get("uuid_local"):
                                atualizar_inspecao(conexao, fechado["uuid_local"], fechado["indicadores"], fechado["defeitos"], fechado["status"], fechado["confianca_saude"])
                                print("🏁 " + resumir_analise(fechado["uuid_local"], cfg, fechado) + " [fechada]")
                            else:
                                u = salvar_inspecao(conexao, cfg, fechado["indicadores"], fechado["defeitos"], fechado["status"], fechado["confianca_saude"], posicao=posicao_atual())
                                print("🏁 " + resumir_analise(u, cfg, fechado))
                        elif rastreador.aberto:
                            parcial = rastreador.parcial()
                            if parcial is not None and rastreador.uuid_atual is None:
                                # Abriu: grava JÁ (o sync manda em segundos; o dashboard mostra a tora).
                                # A posição é a DESTE momento (o corte); as atualizações não a mudam.
                                posicao = posicao_atual()
                                rastreador.uuid_atual = salvar_inspecao(
                                    conexao, cfg, parcial["indicadores"], parcial["defeitos"],
                                    parcial["status"], parcial["confianca_saude"], posicao=posicao,
                                )
                                ultima_gravacao = agora
                                print("🟢 " + resumir_analise(rastreador.uuid_atual, cfg, parcial) + " [aberta]" + descrever_posicao(posicao))
                            elif parcial is not None and agora - ultima_gravacao >= cfg.intervalo_captura_seg:
                                # Segue na frente da câmera: atualiza o mesmo registro
                                ultima_gravacao = agora
                                atualizar_inspecao(conexao, rastreador.uuid_atual, parcial["indicadores"], parcial["defeitos"], parcial["status"], parcial["confianca_saude"])
                                print("↻ " + resumir_analise(rastreador.uuid_atual, cfg, parcial) + " [atualizada]")
                    else:
                        with lock:
                            estado["analise"] = a
                        # Modo intervalo: grava no máximo uma vez por intervalo
                        agora = time.monotonic()
                        if a.get("eh_tora", True) and agora - ultima_gravacao >= cfg.intervalo_captura_seg:
                            ultima_gravacao = agora
                            uuid_gerado = salvar_inspecao(
                                conexao, cfg, a["indicadores"], a["defeitos"], a["status"], a["confianca_saude"],
                                posicao=posicao_atual(),
                            )
                            print(resumir_analise(uuid_gerado, cfg, a))
                except Exception as e:
                    print(f"⚠️  Erro na análise (seguindo em frente): {e}")
                    time.sleep(0.05)
                time.sleep(0.03)  # ~10 fps é mais que suficiente para a parte clássica
        finally:
            conexao.close()

    threads = [
        threading.Thread(target=worker_yolo, daemon=True),
        threading.Thread(target=worker_analise, daemon=True),
    ]
    # Quarta thread, opcional: câmera ao vivo no dashboard (stream_url no
    # config.json + STREAM_TOKEN no .env). Sem rede ela só espera.
    if cfg.stream_url and cfg.stream_url.strip():
        threads.append(threading.Thread(target=loop_stream, args=(cfg, estado, lock, parar), daemon=True))
    for t in threads:
        t.start()

    mostrar = args.simulado or args.gui
    pasta_capturas = Path("./capturas")

    # Fundo de referência automático: só com CÂMERA (num vídeo/pasta o
    # primeiro frame já tem tora). Junta os frames dos primeiros segundos e
    # usa a mediana pixel a pixel — robusta a ruído e a alguém passando.
    fonte_e_camera = not (args.fonte and Path(args.fonte).exists())
    auto_fundo = fonte_e_camera and cfg.fundo_auto_seg > 0
    frames_fundo: list = []
    inicio_captura = time.monotonic()

    miniatura = {"img": None}

    def definir_fundo(imagem: np.ndarray, origem: str) -> None:
        with lock:
            estado["fundo"] = imagem
        # Miniatura para a janela (160 px de largura, proporção do frame)
        esc = 160.0 / imagem.shape[1]
        miniatura["img"] = cv2.resize(imagem, (160, max(1, int(imagem.shape[0] * esc))), interpolation=cv2.INTER_AREA)
        if cfg.fundo_salvar_em:
            try:
                Path(cfg.fundo_salvar_em).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(cfg.fundo_salvar_em, imagem)
            except Exception:
                pass
        print(f"🖼️  Fundo de referência definido ({origem}). A tora passa a ser 'o que difere do fundo'. Tecle 'b' para recapturar.")
        # Aviso se o quadro parece conter uma peça: o fundo com a peça dentro
        # faz ela NUNCA ser detectada. Confira na miniatura do canto da janela.
        try:
            m_, c_, o_ = segmentar_tora_origem(imagem)
            parece, _, _ = validar_tora(imagem, m_, c_, o_, cfg)
        except Exception:
            parece = False
        if parece:
            print("⚠️  ATENÇÃO: parece haver uma tora no quadro capturado como fundo. Se a miniatura no canto mostrar a peça, TIRE a peça e tecle 'b' de novo.")

    if auto_fundo:
        print(f"🖼️  Capturando o fundo de referência nos primeiros {cfg.fundo_auto_seg:g} s — deixe a garra VAZIA.")
    elif fonte_e_camera:
        print("🖼️  Sem fundo de referência. Com a mesa/garra VAZIA na frente da câmera, tecle 'b' na janela (recomendado: funciona em qualquer fundo).")
    print(
        "🚀 Rodando em tempo real. "
        + ("Tecle 'q' na janela para sair, 'b' para capturar o fundo (garra vazia), 's' para salvar o frame em capturas/, ou " if mostrar else "")
        + "Ctrl+C para parar.\n"
    )

    try:
        while not parar.is_set():
            sucesso, frame = captura.read()
            if not sucesso:
                if fonte_e_video:
                    # Fim do vídeo: volta ao início (demo em loop).
                    captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("⚠️  Falha ao capturar frame. Tentando de novo...")
                time.sleep(0.1)
                continue

            cru = frame  # o que a câmera entregou (tecla 's' salva este)
            frame = realce.processar(frame)
            if realce.ligado:
                # Console: avisa só quando o nível MUDA e fica estável por 1,5 s.
                # Mesa preta com a garra vazia é uma cena escura mesmo com a sala
                # clara — sem isso o console piscava "critica/baixa" a cada
                # tora que entra e sai. (A medição usa o nível na hora, sempre.)
                agora_luz = time.monotonic()
                if realce.nivel != ultimo_nivel_luz.get("candidato"):
                    ultimo_nivel_luz["candidato"] = realce.nivel
                    ultimo_nivel_luz["desde"] = agora_luz
                elif realce.nivel != ultimo_nivel_luz["v"] and agora_luz - ultimo_nivel_luz["desde"] >= 1.5:
                    e = realce.estado
                    print(f"💡 Luz {e['nivel'].upper()} (brilho {e['brilho']:.0f}, ruído {e['ruido']:.1f})"
                          + (" — realce ligado: empilhamento + ganho" if e["nivel"] != "boa" else "")
                          + (" — toras medidas com BAIXA CONFIANÇA até iluminar" if e["nivel"] == "critica" else ""))
                    ultimo_nivel_luz["v"] = realce.nivel

            # Publica o frame mais recente para o worker e pega o último resultado.
            with lock:
                estado["frame"] = frame
                estado["luz"] = realce.estado
                analise = estado["analise"]

            if auto_fundo:
                if time.monotonic() - inicio_captura <= cfg.fundo_auto_seg:
                    if len(frames_fundo) < 30:
                        frames_fundo.append(frame.copy())
                else:
                    auto_fundo = False
                    if frames_fundo:
                        definir_fundo(np.median(np.stack(frames_fundo), axis=0).astype(np.uint8), f"automático, mediana de {len(frames_fundo)} frames")
                    frames_fundo = []

            if mostrar:
                cv2.imshow("Omni-Root | John Deere Wood Inspection", desenhar_analise(frame, analise, miniatura["img"]))
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord('q'):
                    break
                if tecla == ord('b'):
                    # Fundo de referência manual: garra/mesa VAZIA na frente da câmera.
                    definir_fundo(frame.copy(), "manual, tecla b")
                if tecla == ord('s'):
                    # Frame CRU (sem desenho e SEM realce de luz): serve para
                    # dataset e para testar o pipeline offline com
                    # `--fonte ./capturas` — que aplica o realce de novo.
                    pasta_capturas.mkdir(exist_ok=True)
                    nome = pasta_capturas / f"frame_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(nome), cru)
                    print(f"💾 Frame salvo em {nome}")
            else:
                # Sem janela: só mantém o worker alimentado sem ocupar 100% da CPU.
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n🛑 Encerrando por solicitação do usuário...")
    finally:
        parar.set()
        for t in threads:
            t.join(timeout=2.0)
        if leitor_gnss is not None:
            leitor_gnss.parar()
        captura.release()
        cv2.destroyAllWindows()
        print("✅ Recursos liberados. Até a próxima inspeção!")


if __name__ == "__main__":
    main()
