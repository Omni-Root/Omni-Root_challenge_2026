"""
analise.py — Pipeline de UM quadro: da imagem aos indicadores (função pura, usada pelo loop ao vivo e pelo simulador).

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

from typing import TYPE_CHECKING

import cv2
import numpy as np

from omniroot.classificacao import calcular_confianca_saude, classificar_qualidade, severidade_do_defeito
from omniroot.config import Config
from omniroot.deteccao import (
    FiltroPersistencia,
    deslocar_defeitos,
    detectar_yolo,
    filtrar_dentro_do_contorno,
    filtrar_por_area_minima,
)
from omniroot.inventario import calcular_densidade_estimada
from omniroot.medidas import (
    calcular_cm_por_px,
    calcular_dimensao_real_cm,
    calcular_massa_seca_kg,
    calcular_porcentagem_casca,
    calcular_tortuosidade,
    calcular_volume_m3,
    detectar_rachaduras_secao,
    escala_por_marcador,
)
from omniroot.segmentacao import (
    balancear_branco,
    classificar_vista,
    mascara_interior,
    recortar_roi,
    segmentar_por_fundo,
    segmentar_tora_origem,
    validar_tora,
)

if TYPE_CHECKING:  # só para a anotação de tipo; quem carrega o modelo é omniroot.deteccao
    from ultralytics import YOLO

# ============================================================
# PIPELINE DE ANÁLISE DE UM FRAME
# ============================================================
# Tudo o que acontece entre "chegou um frame" e "temos status +
# indicadores" vive aqui, numa função pura (sem thread, sem banco, sem
# janela). O loop ao vivo (main) e o simulador sem câmera
# (tests/simular_cenario.py) chamam a MESMA função -- o que a demo mostra
# é exatamente o que o teste testa.
# ============================================================


def analisar_frame(
    frame: np.ndarray,
    modelo: "YOLO | None",
    cfg: Config,
    persistencia: "FiltroPersistencia | None" = None,
    defeitos_yolo: "list | None" = None,
    yolo_novo: bool = True,
    fundo: "np.ndarray | None" = None,
) -> dict:
    """
    Executa o pipeline completo num frame BGR e devolve um dicionário com:
      status, confianca_saude, defeitos (filtrados, coords do frame inteiro),
      descartados (o que o modelo viu mas os filtros removeram), indicadores
      (prontos para salvar_inspecao), contorno, roi (x, y, w, h) e hud.

    Duas formas de uso:
      - `modelo` dado: roda o YOLO aqui mesmo (simulador, testes offline).
      - `defeitos_yolo` dado (coords do recorte da ROI): reaproveita um
        resultado do YOLO calculado em outra thread. É assim que o loop ao
        vivo mantém a parte clássica (~70 ms) a ~10 fps enquanto o modelo
        (~0,5-2 s em CPU) roda no ritmo dele. `yolo_novo=False` avisa que
        são as mesmas caixas da chamada anterior, para o filtro de
        persistência não contar o mesmo resultado duas vezes.
    """
    # --- 1. ROI: a câmera é fixa, a tora sempre aparece na mesma região ---
    recorte, (rx, ry, rw, rh) = recortar_roi(frame, cfg)

    # --- 2. Defeitos do modelo (no recorte pré-processado) ---
    # As medições por OpenCV mais abaixo usam o recorte COLORIDO original:
    # contorno e porcentagem de casca funcionam melhor com cor.
    if defeitos_yolo is None:
        if modelo is None:
            raise ValueError("analisar_frame precisa de `modelo` ou de `defeitos_yolo`.")
        defeitos_yolo = detectar_yolo(recorte, modelo, cfg)
    defeitos_brutos = list(defeitos_yolo)

    # --- 3. Segmentação da tora (dentro da ROI) + portão "é madeira?" ---
    # Precisa vir ANTES dos filtros: é o que permite descartar detecção
    # fora da madeira. Por cor (madeira x fundo), com Otsu de brilho como
    # reserva. O portão decide se o blob achado é mesmo uma tora; se não
    # for, o frame é tratado como "sem tora": nenhum defeito conta, nada
    # é medido e o evento de tora não abre.
    # Cor corrigida pela luz do ambiente só para a parte clássica.
    recorte_cor = balancear_branco(recorte, cfg.balanco_branco_borda)

    # Marcador ArUco antes da segmentação: além da escala, o cartão branco
    # do marcador é EXCLUÍDO da máscara (numa mesa preta ele seria o objeto
    # mais claro do quadro e viraria "tora pálida").
    cm_por_px, marcador = escala_por_marcador(frame, cfg.marcador_aruco_cm)
    ignorar = None
    if marcador is not None:
        ignorar = np.zeros(recorte.shape[:2], dtype=np.uint8)
        pts = (marcador.astype(np.int32) - np.array([rx, ry], dtype=np.int32)).reshape(-1, 1, 2)
        cv2.fillConvexPoly(ignorar, pts, 255)
        lado = float(np.mean([np.linalg.norm(marcador[i] - marcador[(i + 1) % 4]) for i in range(4)]))
        k = max(3, int(0.35 * lado)) | 1
        ignorar = cv2.dilate(ignorar, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    # Com fundo de referência (câmera fixa), ele é a ÚNICA fonte: "nada
    # difere do fundo" significa garra vazia, e cair na cor aqui devolveria
    # o fundo inteiro quando madeira e fundo têm a mesma cor. Sem fundo
    # capturado, segmenta por cor (madeira quente x fundo neutro).
    if fundo is not None:
        fundo_rec, _ = recortar_roi(fundo, cfg)
        mask, contorno, motivo_fundo = segmentar_por_fundo(recorte, fundo_rec, cfg.fundo_limiar, ignorar)
        origem_seg = "fundo" if mask is not None else "nenhuma"
        eh_tora, motivo_sem_tora, _ = validar_tora(recorte_cor, mask, contorno, origem_seg, cfg)
        if not eh_tora and motivo_fundo:
            motivo_sem_tora = motivo_fundo
    else:
        mask, contorno, origem_seg = segmentar_tora_origem(recorte_cor, ignorar)
        eh_tora, motivo_sem_tora, _ = validar_tora(recorte_cor, mask, contorno, origem_seg, cfg)
    if not eh_tora:
        mask, contorno = None, None
    interior = mascara_interior(mask) if mask is not None else None
    vista, lado_menor_px, lado_maior_px, _ = classificar_vista(contorno)

    # Rachadura radial na face de corte: o modelo não cobre; visão clássica cobre.
    if vista == "secao" and mask is not None:
        defeitos_brutos = defeitos_brutos + detectar_rachaduras_secao(recorte, mask)

    # --- 4. Filtros de inferência, em cascata, do mais barato ao mais caro ---
    defeitos = defeitos_brutos
    if not eh_tora:
        defeitos = []  # sem madeira no frame não existe defeito de madeira
    if cfg.filtro_area_minima > 0:
        defeitos = filtrar_por_area_minima(defeitos, cfg.filtro_area_minima)
    if cfg.filtro_dentro_contorno:
        defeitos = filtrar_dentro_do_contorno(defeitos, contorno, mascara_interior=interior)
    if persistencia is not None and cfg.filtro_persistencia > 1:
        # Persistência só se aplica ao YOLO (é ele que "pisca"); a rachadura
        # por visão clássica é determinística no frame. E só avança o
        # histórico quando há resultado NOVO do modelo.
        so_yolo = [d for d in defeitos if d.get("origem") != "opencv"]
        so_cv = [d for d in defeitos if d.get("origem") == "opencv"]
        if yolo_novo:
            so_yolo = persistencia.filtrar(so_yolo)
        else:
            so_yolo = persistencia.ultimo_confirmado(so_yolo)
        defeitos = so_yolo + so_cv

    mantidos_ids = {id(d) for d in defeitos}
    descartados = [d for d in defeitos_brutos if id(d) not in mantidos_ids]

    # --- 5. Saúde da tora, ponderada pela GRAVIDADE de cada defeito ---
    confianca_saude = calcular_confianca_saude(defeitos)

    # --- 6. Escala px -> cm: marcador ArUco (já detectado acima) > config > GSD ---
    if cm_por_px is not None:
        metodo_escala = "marcador_aruco"
    else:
        cm_por_px = calcular_cm_por_px(cfg, frame.shape[1])
        metodo_escala = "montagem_calibrada" if cfg.cm_por_px and cfg.cm_por_px > 0 else "gsd_optica_generica"

    # --- 7. Geometria conforme a VISTA ---
    # Seção (rodela): diâmetro pela área (robusto a borda irregular);
    # comprimento não é visível -> usa o comprimento de traçamento do talhão.
    # Lateral: lado menor = diâmetro, lado maior = comprimento visível;
    # tortuosidade só faz sentido aqui.
    comprimento_medido = False
    if vista == "secao" and mask is not None:
        area_px = float(np.count_nonzero(mask))
        diametro_cm = calcular_dimensao_real_cm((4.0 * area_px / np.pi) ** 0.5, cm_por_px)
        comprimento_cm = float(cfg.comprimento_corte_cm)
        tortuosidade = 0.0
    elif vista == "lateral":
        diametro_cm = calcular_dimensao_real_cm(lado_menor_px, cm_por_px)
        comprimento_cm = calcular_dimensao_real_cm(lado_maior_px, cm_por_px)
        comprimento_medido = True
        tortuosidade = calcular_tortuosidade(contorno)
    else:
        # Nada segmentado: usa a ROI como estimativa grosseira e sinaliza.
        diametro_cm = calcular_dimensao_real_cm(min(rw, rh), cm_por_px)
        comprimento_cm = float(cfg.comprimento_corte_cm)
        tortuosidade = 0.0

    densidade = calcular_densidade_estimada(cfg.clone_id)
    porcentagem_casca = calcular_porcentagem_casca(recorte_cor, contorno)
    volume_util_m3 = calcular_volume_m3(diametro_cm, comprimento_cm, confianca_saude)
    massa_seca_kg = calcular_massa_seca_kg(volume_util_m3, densidade)

    status = classificar_qualidade(confianca_saude, defeitos, cfg) if eh_tora else "sem_tora"

    metodo_diam = f"imagem_{vista}_{metodo_escala}"
    metodo_compr = f"imagem_lateral_{metodo_escala}" if comprimento_medido else "comprimento_tracamento_config"
    metodo_tort = "opencv_eixo_flecha" if vista == "lateral" else "nao_aplicavel_secao"
    indicadores = {
        "densidade": {"valor": densidade, "unidade": "kg/m3", "metodo": f"lookup_clone_{cfg.clone_id}"},
        "altura": {"valor": comprimento_cm, "unidade": "cm", "metodo": metodo_compr},
        "diametro": {"valor": diametro_cm, "unidade": "cm", "metodo": metodo_diam},
        "tortuosidade": {"valor": tortuosidade, "unidade": "indice", "metodo": metodo_tort},
        "porcentagem_casca": {"valor": porcentagem_casca, "unidade": "%", "metodo": "opencv_otsu_casca_residual"},
        "volume_util": {"valor": volume_util_m3, "unidade": "m3", "metodo": "cilindro_" + ("medido" if comprimento_medido else "diam_medido_compr_config")},
        "massa_seca": {"valor": massa_seca_kg, "unidade": "kg", "metodo": f"volume_x_densidade_clone_{cfg.clone_id}"},
        "apodrecimento_pragas": {"valor": round(confianca_saude * 100, 2), "unidade": "%", "metodo": "yolo_severidade"},
    }

    # Coordenadas de volta ao frame inteiro (é assim que vão para o banco e
    # para a tela — a ROI é detalhe de implementação, não do dado).
    defeitos = deslocar_defeitos(defeitos, rx, ry)
    descartados = deslocar_defeitos(descartados, rx, ry)
    contorno_frame = None
    if contorno is not None:
        contorno_frame = np.asarray(contorno, dtype=np.int32).reshape(-1, 1, 2) + np.array([rx, ry], dtype=np.int32)

    rotulo_vista = {"secao": "Secao", "lateral": "Lateral", "desconhecida": "Sem tora"}[vista]
    compr_txt = f"Compr {comprimento_cm:.0f}cm" + ("" if comprimento_medido else "*")
    tort_txt = f"Tort {tortuosidade:.1f}%" if vista == "lateral" else "Tort n/a"
    hud = [
        (f"Status: {status.upper()} | Clone: {cfg.clone_id}" if eh_tora
         else f"SEM TORA ({motivo_sem_tora}) | Clone: {cfg.clone_id}"),
        f"Saude {confianca_saude:.0%} | Casca {porcentagem_casca:.1f}% | Defeitos {len(defeitos)}"
        + (f" (+{len(descartados)} ign.)" if descartados else ""),
        f"{rotulo_vista} | Diam {diametro_cm:.1f}cm | {compr_txt} | Dens {densidade:.0f}kg/m3 | {tort_txt}",
        f"Escala: {metodo_escala} ({cm_por_px * 10:.2f} mm/px) | Seg: {origem_seg if eh_tora else '-'}"
        + ("" if fundo is not None else " | sem fundo de ref. (tecle b com a garra vazia)")
        + ("" if comprimento_medido else "  *comprimento de tracamento"),
    ]

    return {
        "status": status,
        "segmentacao": origem_seg,
        "eh_tora": eh_tora,
        "motivo_sem_tora": motivo_sem_tora,
        "confianca_saude": confianca_saude,
        "defeitos": defeitos,
        "descartados": descartados,
        "indicadores": indicadores,
        "contorno": contorno_frame,
        "roi": (rx, ry, rw, rh),
        "vista": vista,
        "marcador": marcador,
        "hud": hud,
    }


def resumir_analise(uuid_gerado: str, cfg: Config, a: dict) -> str:
    """Linha de log de uma inspeção gravada — o que o operador (e a banca) lê no console."""
    ind = a["indicadores"]
    if a["defeitos"]:
        pior = max(a["defeitos"], key=lambda d: d["confianca"] * severidade_do_defeito(d["tipo_defeito"]))
        resumo_defeito = f"pior={pior['tipo_defeito']}({pior['confianca']:.2f})"
    else:
        resumo_defeito = "sem defeito"
    emoji = {"aprovado": "✅", "quarentena": "⚠️", "reprovado": "❌"}[a["status"]]
    evento = ""
    if "numero" in a:
        evento = f"Tora #{a['numero']} ({a['quadros']} quadros, {a['duracao_s']}s, vista {a['vista']}) "
    return (
        f"{emoji} [{uuid_gerado[:8]}] {evento}Clone={cfg.clone_id} status={a['status']} "
        f"saúde={a['confianca_saude']:.2%} compr={ind['altura']['valor']}cm diam={ind['diametro']['valor']}cm "
        f"densidade={ind['densidade']['valor']}kg/m3 tortuosidade={ind['tortuosidade']['valor']} "
        f"casca={ind['porcentagem_casca']['valor']}% volume={ind['volume_util']['valor']}m3 "
        f"massa={ind['massa_seca']['valor']}kg defeitos={len(a['defeitos'])} {resumo_defeito}"
        + (f" [ignorados={len(a['descartados'])}]" if a["descartados"] else "")
    )


def descrever_posicao(posicao: dict | None) -> str:
    """Sufixo do log do console com a posição gravada na tora (vazio sem posição)."""
    if posicao is None:
        return " [sem posição]"
    return f" [pos {posicao['lat']:.5f},{posicao['lon']:.5f} {posicao['fonte']}]"
