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
import hashlib
import json
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

@dataclass
class Config:
    # --- Identidade da máquina (deve bater com o numero_serie no Postgres) ---
    maquina_id: str = "RASPI-DEMO-01"
    talhao_id: str = "Talhão Demo"
    clone_id: str = "SP3108"             # Clone genético do eucalipto (ex: SP3108, SP2974, CO41H_TEST)
    idade_talhao_anos: float = 5.3       # Idade do plantio na data da colheita

    # --- Banco local ---
    sqlite_path: str = "./omni_root_local.db"

    # --- Modelo de IA ---
    # Ordem de busca: modelo_path (se existir) > OpenVINO INT8 > OpenVINO
    # FP32 > NCNN > .pt. OpenVINO é o runtime da Intel para CPU (o harvester
    # NÃO tem GPU). Medido neste notebook, mesmo estado térmico: .pt a 1024
    # 2.400 ms; .pt a 640 1.050 ms; OpenVINO FP32 640 ~1.000 ms; OpenVINO
    # INT8 640 ~500 ms. Gere com `python exportar_modelo.py` (uma vez por máquina).
    modelo_path: str = "./models/wood_best_int8_openvino_model"
    modelo_ncnn_path: str = "./models/wood_ncnn_model"   # compatibilidade
    conf_threshold: float = 0.40
    iou_threshold: float = 0.55
    # A webcam entrega 640x480: inferir a 1024 só amplia pixel, sem informação
    # nova, e custa 3x. O export OpenVINO/NCNN é gerado com tamanho FIXO --
    # tem que ser o mesmo daqui (exportar_modelo.py lê este valor).
    imgsz: int = 640

    # --- Câmera ---
    camera_index: int = 0
    intervalo_captura_seg: float = 2.0  # tempo entre análises

    # --- Região de interesse (ROI), em fração do frame [x, y, w, h] ---
    # A câmera é fixa na garra: a tora SEMPRE aparece na mesma região do
    # frame. Tudo fora dela (fundo, mão de quem segura a peça na demo,
    # mesa, teclado...) é cortado ANTES da inferência — o modelo nem vê.
    # É o filtro mais barato e mais eficaz contra detecção fora da madeira.
    # null/None desliga (usa o frame inteiro). Ex.: [0.2, 0.05, 0.6, 0.9]
    roi: list | None = None

    # --- Regra de negócio (limiar de aprovação) ---
    limiar_aprovacao: float = 0.85       # 85% de confiança mínima
    limiar_quarentena: float = 0.60      # abaixo disso já é reprovado direto

    # --- Filtros de inferência (reduzem falso positivo na demo) ---
    # Ver as funções filtrar_* e FiltroPersistencia.
    filtro_area_minima: float = 0.0008   # fração da área do frame; 0 desliga
    filtro_dentro_contorno: bool = True  # descarta defeito fora da madeira
    filtro_persistencia: int = 2         # análises consecutivas p/ confirmar; 1 desliga

    # --- Distância câmera-tora, FIXA e calibrada (sem sensor) ---
    # A câmera é montada numa posição fixa em relação à tora (no braço do
    # harvester ou na maquete), então a distância não muda entre leituras.
    # CALIBREM ESSE VALOR com a montagem real antes da apresentação: meçam
    # a distância física câmera-tora uma vez, com fita métrica, e coloquem
    # o número aqui. É o único "hardware" que isso substitui — nenhum
    # sensor físico é necessário.
    distancia_camera_tora_cm: float = 45.0

    # --- Conversão pixel -> cm ---
    # Duas formas, em ordem de preferência:
    #
    #   (a) cm_por_px > 0: calibração DIRETA com régua. Com a câmera na
    #       posição final, coloquem uma régua onde a tora fica, meçam
    #       quantos pixels correspondem a 10 cm no frame e dividam:
    #       cm_por_px = 10 / pixels. É o método mais preciso e o mais
    #       fácil de explicar para a banca. Quando definido, ignora (b).
    #
    #   (b) cm_por_px = 0: fórmula de GSD a partir da óptica da câmera
    #       (distância x largura do sensor / foco x largura do frame).
    #       Os valores abaixo são genéricos de webcam — só servem como
    #       ordem de grandeza até calibrar com a régua.
    #
    # A largura do frame em pixels vem do próprio frame capturado
    # (frame.shape[1]); NÃO é o imgsz do modelo — o Ultralytics devolve
    # as caixas já nas coordenadas do frame original.
    cm_por_px: float = 0.0
    distancia_focal_mm: float = 4.0      # foco da lente
    largura_sensor_mm: float = 6.3       # largura física do sensor da câmera

    # --- Escala AUTOMÁTICA por marcador ArUco (ver escala_por_marcador) ---
    # Lado do marcador impresso, em cm. Se o marcador aparecer no frame, a
    # escala vem dele e ignora cm_por_px/GSD. 0 desliga.
    marcador_aruco_cm: float = 5.0

    # --- Comprimento de traçamento do talhão, em cm ---
    # O harvester corta a tora em comprimento FIXO (setpoint da operação:
    # 6 m é o padrão de celulose no Brasil). Quando a câmera vê só a SEÇÃO
    # (rodela), o comprimento não é mensurável na imagem e o volume usa
    # este valor — é o mesmo que o cabeçote usa para traçar.
    comprimento_corte_cm: float = 600.0

    # --- Portão "isso é madeira?" (ver validar_tora) ---
    # A segmentação SEMPRE acha algum blob no centro (é o trabalho dela);
    # quem decide se aquilo é uma tora é este portão. Sem ele, teclado,
    # notebook e mão viravam "Tora #N". Todos os critérios são baratos e
    # explicáveis; qualquer um reprovado = "Sem tora" (nada é gravado).
    tora_exigir_cor: bool = True        # só aceita segmentação pela COR (fallback de brilho não vale)
    tora_solidez_min: float = 0.80      # área / área do casco convexo: tora é convexa; mão/objetos irregulares não
    tora_fracao_pele_max: float = 0.45  # fração de pixels com matiz de pele; acima disso é mão, não madeira
    tora_razao_max: float = 6.0         # lado maior / lado menor acima disso = cabo, borda, ruído
    tora_fracao_madeira_min: float = 0.55  # fração de pixels da máscara com cor de madeira (matiz quente + saturação)

    # --- Gravação por EVENTO DE TORA (ver RastreadorTora) ---
    # "evento": uma tora = um registro. O evento abre quando a tora aparece
    # e se mantém por `evento_frames_abrir` análises seguidas, fecha quando
    # some por `evento_frames_fechar` (garra vazia) ou quando o contorno
    # "pula" (outra tora entrou sem gap), e grava UM registro consolidando
    # todos os quadros: mediana dos indicadores, união dos defeitos.
    # "intervalo": comportamento antigo — grava a cada intervalo_captura_seg,
    # tenha tora ou não (mantido só para comparação/depuração).
    # Em máquina real o gatilho natural seria o ciclo de corte do cabeçote.
    modo_gravacao: str = "evento"
    evento_frames_abrir: int = 3         # análises seguidas com tora para abrir
    evento_frames_fechar: int = 6        # análises seguidas sem tora para fechar (~0,6 s a 10 fps)
    evento_frames_minimo: int = 4        # evento mais curto que isso é ruído: descarta
    evento_iou_troca: float = 0.25       # IoU do contorno entre quadros abaixo disso = tora diferente
    evento_frames_troca: int = 3         # ...por tantos quadros seguidos
    evento_defeito_min_quadros: int = 2  # defeito precisa aparecer em N quadros do evento

    # --- Câmera ao vivo no dashboard (ver worker_stream) ---
    # A máquina EMPURRA o frame anotado (caixas + HUD) em JPEG para o
    # servidor do dashboard, a poucos fps, enquanto houver rede — mesma
    # direção do sync_daemon.py (em campo a máquina está atrás de 4G/NAT,
    # a central não consegue "puxar" dela). Sem rede, só recua e tenta de
    # novo; a inspeção nunca espera por isso. URL vazia desliga.
    # O token vai em STREAM_TOKEN no .env (é segredo; config.json vai pro Git).
    stream_url: str = ""                 # ex: http://192.168.1.50:3001/api/camera/frame
    stream_fps: float = 4.0              # quadros por segundo enviados (banda ~ 40 KB x fps)
    stream_largura_px: int = 640         # reduz o frame antes de codificar (0 = tamanho original)
    stream_jpeg_qualidade: int = 70      # 1-100


# ============================================================
# PROCESSADOR DO INVENTÁRIO FLORESTAL (John Deere / Suzano)
# ============================================================
def carregar_inventario_florestal(
    caminho_densidade: str = "./data/clones_densidade.json",
    caminho_dendrometria: str = "./data/inventario_johndeere.json",
) -> dict:
    """
    Monta o dicionário de referência por clone usado pela IA, a partir de
    DUAS fontes com papéis bem diferentes — importante não misturar:

      - clones_densidade.json: densidade básica (kg/m3) por clone/material
        genético. É esse o dado que, na prática do setor (confirmado por
        e-mail com o contato da John Deere), varia por genética e não por
        medição em campo.
        O arquivo é um CACHE gravado pelo sync_daemon.py a partir da tabela
        clones_densidade do Postgres (fonte de verdade); cada clone traz
        'tipo_dado' e 'fonte'. Não editar à mão. A recarga em tempo de
        execução é feita por inventario_atual().

      - inventario_johndeere.json: amostras de DAP por árvore, agrupadas por
        clone — isso sim vem da planilha real de inventário que a JD
        passou. NÃO tem densidade. Serve só como contexto dendrométrico
        (DAP médio, idade do talhão) para exibir no dashboard/relatório e,
        opcionalmente, conferir o diâmetro medido pela câmera contra a
        média do talhão. NÃO é usado para calcular densidade — DAP não é
        um bom preditor de densidade básica, e tentar derivar um a partir
        do outro foi o erro da versão anterior deste arquivo.
    """
    estatisticas_clones: dict = {}

    p_dens = Path(caminho_densidade)
    if p_dens.exists():
        with open(p_dens, "r", encoding="utf-8") as f:
            dados_densidade = json.load(f)
        for clone_key, info in dados_densidade.items():
            if not isinstance(info, dict):
                continue
            # densidade_base pode vir None: o clone está cadastrado no
            # Postgres mas ainda aguarda laudo de laboratório. Nesse caso
            # deixamos None de propósito -- calcular_densidade_estimada
            # avisa no console em vez de mascarar com um número inventado.
            dens_bruta = info.get("densidade_base")
            estatisticas_clones[str(clone_key).upper()] = {
                "densidade_base": float(dens_bruta) if dens_bruta is not None else None,
                "taxa_maturacao": info.get("taxa_maturacao"),  # guardado, NÃO aplicado (ver calcular_densidade_estimada)
                "especie": info.get("especie", info.get("descricao", "Eucalyptus sp.")),
                "dap_medio_inventario": None,
                "idade_anos": None,
            }

    p_dendro = Path(caminho_dendrometria)
    if p_dendro.exists():
        with open(p_dendro, "r", encoding="utf-8") as f:
            dados_dendro = json.load(f)
        for clone_key, info in dados_dendro.get("clones", dados_dendro).items():
            if not isinstance(info, dict):
                continue
            key_upper = str(clone_key).upper()
            amostras = info.get("dap_amostras_cm", [])
            dap_medio = float(np.mean(amostras)) if amostras else None

            entrada = estatisticas_clones.setdefault(key_upper, {
                "densidade_base": None,  # sem referência de densidade cadastrada para este clone
                "taxa_maturacao": None,
                "especie": info.get("especie", "Eucalyptus sp."),
                "dap_medio_inventario": None,
                "idade_anos": None,
            })
            entrada["dap_medio_inventario"] = round(dap_medio, 2) if dap_medio is not None else None
            entrada["idade_anos"] = info.get("idade_anos")

    return estatisticas_clones


def carregar_configuracao_json(caminho_json: str = "./config.json") -> Config:
    """Carrega as configurações operacionais da máquina a partir de um arquivo JSON estático."""
    cfg = Config()
    p = Path(caminho_json)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                dados = json.load(f)
                if isinstance(dados, dict):
                    for k, v in dados.items():
                        if hasattr(cfg, k):
                            setattr(cfg, k, v)
                        elif k == "largura_imagem_px":
                            print(
                                "ℹ️  'largura_imagem_px' não é mais usado: a largura vem do "
                                "próprio frame da câmera. Pode remover do config.json."
                            )
        except Exception as e:
            print(f"⚠️ Erro ao carregar {caminho_json}: {e}")

    # ROI precisa ser [x, y, w, h] em fração do frame, tudo em (0, 1].
    if cfg.roi is not None:
        ok = (
            isinstance(cfg.roi, (list, tuple)) and len(cfg.roi) == 4
            and all(isinstance(v, (int, float)) for v in cfg.roi)
            and 0.0 <= cfg.roi[0] < 1.0 and 0.0 <= cfg.roi[1] < 1.0
            and 0.0 < cfg.roi[2] <= 1.0 - cfg.roi[0]
            and 0.0 < cfg.roi[3] <= 1.0 - cfg.roi[1]
        )
        if not ok:
            print(f"⚠️ 'roi' inválida ({cfg.roi}); esperado [x, y, w, h] em fração do frame. Ignorando.")
            cfg.roi = None
    return cfg


# ------------------------------------------------------------
# RECARGA AUTOMÁTICA DA TABELA DE DENSIDADE
# ------------------------------------------------------------
# O sync_daemon.py regrava data/clones_densidade.json toda vez que baixa a
# tabela do Postgres central. Antes, o main.py lia o arquivo UMA vez no
# import: cadastrar a densidade de um clone no banco só valia depois de
# reiniciar a máquina -- justamente a demonstração que a JD pediu. Agora
# `inventario_atual()` confere o mtime/tamanho dos JSONs (um stat, barato)
# no máximo 1x por segundo e recarrega quando mudaram. O sync grava por
# arquivo temporário + replace (atômico), então nunca lemos JSON pela metade;
# se mesmo assim a leitura falhar, fica o cache anterior.
# ------------------------------------------------------------
_ARQUIVOS_INVENTARIO = ("./data/clones_densidade.json", "./data/inventario_johndeere.json")
_INVENTARIO = {"dados": {}, "assinatura": None, "checado_em": -1e9}
_INVENTARIO_LOCK = threading.Lock()
_AVISOS_CLONE_SEM_DENSIDADE: set = set()


def _assinatura_inventario() -> tuple:
    saida = []
    for caminho in _ARQUIVOS_INVENTARIO:
        p = Path(caminho)
        try:
            st = p.stat()
            saida.append((st.st_mtime_ns, st.st_size))
        except OSError:
            saida.append(None)
    return tuple(saida)


def inventario_atual(intervalo_checagem_s: float = 1.0) -> dict:
    """Dicionário por clone, recarregado sozinho quando os JSONs mudam em disco."""
    global INVENTARIO_FLORESTAL
    with _INVENTARIO_LOCK:
        agora = time.monotonic()
        if agora - _INVENTARIO["checado_em"] < intervalo_checagem_s:
            return _INVENTARIO["dados"]
        _INVENTARIO["checado_em"] = agora

        assinatura = _assinatura_inventario()
        if assinatura == _INVENTARIO["assinatura"]:
            return _INVENTARIO["dados"]
        try:
            novo = carregar_inventario_florestal(*_ARQUIVOS_INVENTARIO)
        except Exception as e:
            print(f"⚠️  Não consegui reler a tabela de densidade ({e}); mantendo a anterior.")
            return _INVENTARIO["dados"]

        antigo = _INVENTARIO["dados"]
        if _INVENTARIO["assinatura"] is not None:
            # Não é a primeira carga: diz o que mudou, para a demo ficar visível
            mudancas = []
            for clone, info in novo.items():
                d_novo = info.get("densidade_base")
                d_antigo = (antigo.get(clone) or {}).get("densidade_base")
                if clone not in antigo:
                    mudancas.append(f"{clone}: novo ({d_novo} kg/m3)")
                elif d_novo != d_antigo:
                    mudancas.append(f"{clone}: {d_antigo} -> {d_novo} kg/m3")
            for clone in antigo.keys() - novo.keys():
                mudancas.append(f"{clone}: removido")
            resumo = "; ".join(mudancas[:6]) + (" ..." if len(mudancas) > 6 else "")
            print(f"📥 Tabela de densidade recarregada do disco ({len(novo)} clones). " + (f"Mudanças: {resumo}" if mudancas else "Sem mudança de valor."))
            _AVISOS_CLONE_SEM_DENSIDADE.clear()  # se o clone foi cadastrado, o aviso some; se não, avisa de novo

        _INVENTARIO["dados"] = novo
        _INVENTARIO["assinatura"] = assinatura
        INVENTARIO_FLORESTAL = novo
        return novo


INVENTARIO_FLORESTAL = inventario_atual()
CONFIG = carregar_configuracao_json()


# ============================================================
# CÁLCULO DOS INDICADORES DE QUALIDADE
# ============================================================

def calcular_cm_por_px(cfg: Config, largura_frame_px: int) -> float:
    """
    Fator de conversão pixel -> cm para o frame capturado.

    Se o config tiver `cm_por_px` calibrado com régua, usa direto. Senão,
    cai na fórmula de GSD (Ground Sample Distance) — a mesma lógica usada
    em fotografia aérea, adaptada para a distância câmera-tora fixa:

        GSD (cm/px) = (distância_cm * largura_sensor_mm) / (foco_mm * largura_frame_px)

    `largura_frame_px` é a largura REAL do frame da câmera (frame.shape[1]),
    porque é nesse sistema de coordenadas que as caixas do YOLO e o contorno
    do OpenCV são medidos.
    """
    if cfg.cm_por_px and cfg.cm_por_px > 0:
        return float(cfg.cm_por_px)
    return (cfg.distancia_camera_tora_cm * cfg.largura_sensor_mm) / (
        cfg.distancia_focal_mm * max(1, int(largura_frame_px))
    )


def calcular_dimensao_real_cm(tamanho_px: float, cm_por_px: float) -> float:
    """Converte um tamanho em pixels para centímetros reais."""
    return round(float(tamanho_px) * float(cm_por_px), 2)


def calcular_volume_m3(diametro_cm: float, comprimento_cm: float, confianca_saude: float) -> float:
    """
    Estima o volume da tora a partir de medidas REAIS (diâmetro e
    comprimento, ambos vindos do bounding box do contorno + GSD) —
    não mais de uma proporção arbitrária entre os dois.

    Fórmula do cilindro (aproximação de Huber, sem afilamento):
        volume_bruto = pi * raio^2 * comprimento

    Para volume comercial mais preciso (considerando o afilamento
    natural da tora), a indústria usa a fórmula de Smalian, que pede
    o diâmetro nas DUAS pontas:
        volume = (pi/8) * (diametro_fino^2 + diametro_grosso^2) * comprimento
    Isso exigiria medir o diâmetro nas duas extremidades da tora (hoje
    só medimos uma vez, no centro do frame) — fica como próximo passo
    se quiserem refinar depois da apresentação.

    volume_util desconta, de forma proporcional, o volume equivalente
    à severidade dos defeitos detectados (confianca_saude = 1.0 = tora
    limpa, sem desconto).
    """
    raio_m = (diametro_cm / 100.0) / 2.0
    comprimento_m = comprimento_cm / 100.0
    volume_bruto_m3 = float(np.pi * (raio_m ** 2) * comprimento_m)
    return round(float(volume_bruto_m3 * confianca_saude), 3)


def calcular_massa_seca_kg(volume_util_m3: float, densidade_kg_m3: float) -> float:
    """
    Massa seca estimada = volume útil medido pela câmera (m3) x densidade
    de referência do clone (kg/m3).

    Cruza um dado MEDIDO (volume, vindo da visão computacional) com um dado
    ESTIMADO (densidade, vindo do lookup por clone — ver
    calcular_densidade_estimada). O resultado herda a incerteza da
    densidade: é uma estimativa de massa, não uma pesagem real. Isso é
    exatamente o que a apresentação deve deixar claro ao citar este número.
    """
    return round(float(volume_util_m3) * float(densidade_kg_m3), 1)


def calcular_densidade_estimada(clone_id: str, inventario_stats: dict | None = None) -> float:
    """
    Retorna a densidade básica de referência (kg/m3) para o clone/material
    genético informado — um lookup simples, não um modelo derivado.

    De onde vem o valor (ver data/clones_densidade.json para as fontes
    completas): buscamos os códigos de clone do inventário (SP3108, SP2974
    etc.) em literatura científica e não há correspondência pública — são
    códigos internos proprietários, o que bate com o que o contato da John
    Deere já tinha explicado por e-mail. A própria planilha de inventário
    também não permite identificar a espécie botânica real por trás de cada
    código (a coluna "Espécie" está preenchida com o próprio código do
    clone). Por isso, TODOS os clones cadastrados hoje usam o mesmo valor:
    a densidade básica média de híbridos comerciais de Eucalyptus grandis x
    E. urophylla — o material genético mais comum em plantios industriais
    de celulose no Brasil — medida em 3 fontes técnicas/acadêmicas reais
    (tese USP, boletim técnico e artigo Revista Árvore/SciELO, citados no
    JSON). Isso é uma aproximação de literatura no nível de espécie/híbrido,
    não o dado de laboratório do clone específico (que é proprietário).

    Por quê lookup e não fórmula: densidade básica da madeira é, na prática
    do setor, majoritariamente definida pelo material genético, não algo
    que se calcule a partir de DAP/altura/idade medidos em campo — uma
    versão anterior deste arquivo tentava "derivar" densidade com uma
    fórmula sem base científica (e com um bug que contava o efeito da
    idade duas vezes); foi removida.

    Se o clone não tiver densidade cadastrada em data/clones_densidade.json,
    cai num valor genérico de eucalipto e avisa no console — isso é
    intencional, para não mascarar dado faltante com um número inventado.
    """
    inv = inventario_stats if inventario_stats is not None else inventario_atual()
    key = str(clone_id).upper()
    info = inv.get(key)

    if not info or info.get("densidade_base") is None:
        # Avisa UMA vez por clone (a análise roda ~10x/s); volta a avisar se
        # a tabela for recarregada e o clone continuar sem densidade.
        if key not in _AVISOS_CLONE_SEM_DENSIDADE:
            _AVISOS_CLONE_SEM_DENSIDADE.add(key)
            print(
                f"⚠️  Sem densidade cadastrada para o clone '{clone_id}'. Usando média "
                f"genérica de eucalipto (500.0 kg/m3). Para corrigir, cadastre o valor "
                f"na tabela 'clones_densidade' do PostgreSQL central (fonte de verdade) "
                f"— o sync_daemon.py traz a atualização automaticamente na próxima "
                f"sincronização, e o main.py recarrega sozinho, sem reiniciar."
            )
        return 500.0

    return round(float(info["densidade_base"]), 1)


def _mascara_do_contorno(shape: tuple, contorno) -> np.ndarray:
    """Máscara binária (255 dentro) a partir de um contorno, no tamanho do frame."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.asarray(contorno, dtype=np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [pts], -1, 255, -1)
    return mask


_ULTIMO_AVISO_CASCA = 0.0
CASCA_FRACAO_BRILHO = 0.60   # pixel mais escuro que 60% da madeira exposta = casca


def calcular_porcentagem_casca(frame: np.ndarray | None, contorno: np.ndarray | None) -> float:
    """
    Casca RESIDUAL: % da superfície visível da tora ainda coberta por casca.

    É a grandeza que interessa à fábrica no enunciado do desafio ("casca
    excessiva influencia na qualidade da celulose"): o cabeçote do harvester
    descasca o eucalipto na colheita, e o que chega à fábrica é a casca que
    SOBROU. Uma câmera lateral vê exatamente isso.

    Método (OpenCV clássico, sem modelo):
      1. Só olha os pixels DENTRO do contorno da tora (suavizados, para
         medir casca x madeira e não veio x veio).
      2. Referência de "madeira exposta" = percentil 90 do brilho dentro da
         tora (o creme/amarelo claro da madeira descascada).
      3. Pixel é casca se for mais escuro que CASCA_FRACAO_BRILHO x essa
         referência: casca (marrom/cinza) tem tipicamente menos da metade
         do brilho da madeira exposta.
      4. casca % = pixels de casca / pixels da tora.

    Por que não Otsu: Otsu SEMPRE divide em dois grupos, mesmo quando não há
    casca nenhuma -- num disco limpo com sombra de um lado ele devolvia 30-50%
    de "casca". Com o limiar relativo, uma superfície uniforme dá ~0%, e o
    número só sobe quando existe região realmente escura em relação à madeira.
    Limitação conhecida: tora INTEIRA com casca (não descascada) tem a própria
    casca como referência e sai baixa -- para a demo o esperado é peça
    descascada com casca residual.
    """
    if frame is None or contorno is None or len(contorno) < 3:
        return 0.0  # sem dados suficientes para medir — retorna 0 (desconhecido)

    try:
        mask = _mascara_do_contorno(frame.shape, contorno)
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cinza = cv2.GaussianBlur(cinza, (9, 9), 0)
        pixels = cinza[mask > 0]
        if pixels.size < 100:
            return 0.0

        referencia = float(np.percentile(pixels, 90))
        if referencia < 40:
            return 0.0  # tora inteira escura: sem madeira exposta para comparar
        limiar = CASCA_FRACAO_BRILHO * referencia
        escuros = int(np.count_nonzero(pixels < limiar))
        pct = round(100.0 * escuros / float(pixels.size), 1)

        # Aviso de outlier, no máximo a cada 15 s (não bloqueia, só informa)
        global _ULTIMO_AVISO_CASCA
        if pct > 60.0 and time.monotonic() - _ULTIMO_AVISO_CASCA > 15.0:
            _ULTIMO_AVISO_CASCA = time.monotonic()
            print(f"⚠️  Casca residual muito alta ({pct}%): tora não descascada ou segmentação pegando fundo escuro — confira na tela.")
        return pct
    except Exception:
        return 0.0


# ------------------------------------------------------------
# PRÉ-PROCESSAMENTO PARA O MODELO
# ------------------------------------------------------------
# O modelo foi TREINADO com as imagens convertidas para escala de
# cinza e equalizadas com CLAHE (ver a célula de preparação do
# dataset no notebook de treino). Se a inferência mandar o frame
# BGR cru da câmera, o modelo recebe uma distribuição de pixels
# diferente da que aprendeu -- isso degrada a precisão sem gerar
# erro nenhum, o tipo de problema que passa despercebido.
#
# Esta função replica EXATAMENTE o pré-processamento do treino:
#     gray = cvtColor(img, BGR2GRAY)
#     gray = CLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
#     img  = cvtColor(gray, GRAY2BGR)
#
# ⚠️ Se mudarem clipLimit/tileGridSize no notebook de treino,
#    mudem AQUI TAMBÉM -- os dois precisam andar juntos sempre.
# ------------------------------------------------------------
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocessar_para_modelo(frame: np.ndarray) -> np.ndarray:
    """Aplica grayscale + CLAHE, igual ao pré-processamento do treino."""
    if frame is None:
        return frame
    try:
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cinza = _CLAHE.apply(cinza)
        return cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
    except Exception:
        # Em caso de falha, devolve o frame original: melhor inferir
        # com qualidade degradada do que derrubar o loop inteiro.
        return frame


# ============================================================
# FILTROS DE INFERÊNCIA — reduzem falso positivo na demo ao vivo
# ============================================================
# O modelo tem Precision ~0.72: cerca de 3 em cada 10 detecções são
# falsas. Numa demonstração ao vivo isso aparece como "detectou um
# defeito no nada". Retreinar não resolve a tempo — mas três filtros
# baratos, aplicados DEPOIS da inferência, cortam a maior parte:
#
#   1. ÁREA MÍNIMA   -> caixa minúscula quase sempre é ruído
#   2. DENTRO DA TORA-> defeito fora do contorno da madeira é impossível
#   3. PERSISTÊNCIA  -> falso positivo pisca, defeito real permanece
#
# Nenhum deles mexe no modelo; todos são reversíveis por config.
# ============================================================

def filtrar_por_area_minima(defeitos: list, area_min_relativa: float = 0.0008) -> list:
    """
    Descarta caixas muito pequenas. Num frame de 1024x1024, 0.0008
    equivale a ~840 px² (uma caixa de ~29x29). Defeito real que
    aparece menor que isso também não seria confiável na prática.

    Detecções de origem "opencv" (rachadura radial) são ISENTAS: a área
    delas é a da LINHA, não da caixa — uma rachadura de ponta a ponta tem
    área minúscula e era descartada aqui por engano.
    """
    return [
        d for d in defeitos
        if d.get("origem") == "opencv" or d.get("area_relativa", 0.0) >= area_min_relativa
    ]


def filtrar_dentro_do_contorno(defeitos: list, contorno, margem_px: int = 20, mascara_interior=None) -> list:
    """
    Mantém apenas defeitos cujo CENTRO cai dentro da tora.

    É o filtro mais eficaz para a demonstração: elimina detecção na
    mesa, na mão de quem segura a peça, no fundo da sala — coisas que
    o modelo nunca viu no treino e onde ele "alucina" com mais
    frequência.

    Se `mascara_interior` for dada (tora SEM a faixa de borda/casca), o
    teste é feito nela: o modelo, treinado em madeira serrada, confunde o
    anel de casca escuro de uma seção com "Knot_missing" — defeito de
    fibra fica no miolo, não na borda. Sem máscara, usa o polígono do
    contorno com uma margem.

    Se nada foi segmentado, não filtra (melhor deixar passar do que
    esconder tudo por falha de segmentação).
    """
    try:
        mantidos = []
        if mascara_interior is not None:
            h, w = mascara_interior.shape[:2]
            for d in defeitos:
                cx = int(d["pos_x"] + d["largura"] / 2.0)
                cy = int(d["pos_y"] + d["altura"] / 2.0)
                if 0 <= cx < w and 0 <= cy < h and mascara_interior[cy, cx] > 0:
                    mantidos.append(d)
            return mantidos

        # 3 pontos já formam um polígono válido para pointPolygonTest.
        # Usar 5 aqui seria um bug: o CHAIN_APPROX_SIMPLE do OpenCV comprime
        # um contorno retangular (uma tora vista de lado, por exemplo) para
        # exatamente 4 pontos -- e o filtro deixaria de funcionar justo no
        # caso mais comum.
        if contorno is None or len(contorno) < 3:
            return defeitos
        pts = np.array(contorno, dtype=np.int32).reshape(-1, 1, 2)
        for d in defeitos:
            cx = d["pos_x"] + d["largura"] / 2.0
            cy = d["pos_y"] + d["altura"] / 2.0
            # distância positiva = dentro; negativa = fora
            dist = cv2.pointPolygonTest(pts, (float(cx), float(cy)), True)
            if dist >= -margem_px:
                mantidos.append(d)
        return mantidos
    except Exception:
        return defeitos


class FiltroPersistencia:
    """
    Exige que um defeito apareça em N análises consecutivas, na mesma
    região aproximada, antes de ser considerado real.

    Por que funciona: falso positivo do modelo é instável -- aparece
    num frame e some no próximo. Defeito de verdade, com a peça
    parada na frente da câmera, é detectado repetidamente no mesmo
    lugar. Com intervalo de 2s entre análises e min_ocorrencias=2,
    custa ~2 segundos a mais para confirmar, e corta a maior parte
    do ruído.
    """

    def __init__(self, min_ocorrencias: int = 2, tolerancia_px: float = 80.0, memoria: int = 4):
        self.min_ocorrencias = min_ocorrencias
        self.tolerancia_px = tolerancia_px
        self.memoria = memoria
        self.historico: list[list[dict]] = []

    def _mesma_regiao(self, a: dict, b: dict) -> bool:
        ax = a["pos_x"] + a["largura"] / 2.0
        ay = a["pos_y"] + a["altura"] / 2.0
        bx = b["pos_x"] + b["largura"] / 2.0
        by = b["pos_y"] + b["altura"] / 2.0
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= self.tolerancia_px

    def filtrar(self, defeitos: list) -> list:
        self.historico.append(defeitos)
        if len(self.historico) > self.memoria:
            self.historico.pop(0)

        if len(self.historico) < self.min_ocorrencias:
            return []  # ainda aquecendo: não afirma nada nos primeiros frames

        confirmados = []
        for d in defeitos:
            vistas = sum(
                1 for frame_ant in self.historico[:-1]
                if any(self._mesma_regiao(d, ant) and ant["tipo_defeito"] == d["tipo_defeito"]
                       for ant in frame_ant)
            )
            if vistas + 1 >= self.min_ocorrencias:
                confirmados.append(d)
        return confirmados

    def ultimo_confirmado(self, defeitos: list) -> list:
        """Reaplica a decisão da última chamada a `filtrar` sem avançar o histórico (mesmo resultado do modelo)."""
        if len(self.historico) < self.min_ocorrencias:
            return []
        confirmados = []
        for d in defeitos:
            vistas = sum(
                1 for frame_ant in self.historico[:-1]
                if any(self._mesma_regiao(d, ant) and ant["tipo_defeito"] == d["tipo_defeito"]
                       for ant in frame_ant)
            )
            if vistas + 1 >= self.min_ocorrencias:
                confirmados.append(d)
        return confirmados

    def reset(self) -> None:
        self.historico.clear()


def extrair_contorno_tora(frame: np.ndarray) -> np.ndarray | None:
    """
    Segmenta a região da tora no frame usando visão computacional clássica
    (Otsu thresholding + operações morfológicas). Retorna o contorno
    principal da tora ou None.

    Otsu separa o frame em "claro" e "escuro", mas não sabe qual dos dois
    é a madeira: tora clara em fundo escuro e tora escura em fundo claro
    (parede branca, mesa clara) são igualmente comuns. Por isso testamos
    as DUAS polaridades. Blobs que cobrem >95% da área são fundo, não tora.

    Entre os candidatos, preferimos o que CONTÉM O CENTRO do frame: a
    câmera é fixa e apontada para a tora, então a tora está no meio. Isso
    resolve o caso mais comum na garra — a tora atravessa o frame de ponta
    a ponta e divide o fundo em duas faixas do mesmo tamanho que ela;
    "pegar o maior blob" viraria cara-ou-coroa. Se nenhum candidato contém
    o centro, fica o de centroide mais próximo dele.

    Use junto com a ROI do config: quanto menos fundo entra aqui, mais
    confiável fica o contorno.
    """
    if frame is None:
        return None

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        h, w = frame.shape[:2]
        area_frame = float(h * w)
        area_minima = area_frame * 0.05
        area_maxima = area_frame * 0.95
        centro = (w / 2.0, h / 2.0)

        candidatos = []
        for modo in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            _, thresh = cv2.threshold(blurred, 0, 255, modo + cv2.THRESH_OTSU)
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contornos, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contornos:
                a = cv2.contourArea(c)
                if area_minima <= a <= area_maxima:
                    candidatos.append((a, c))

        if not candidatos:
            return None

        # 1º critério: contém o centro (distância >= 0); 2º: centroide mais perto do centro
        def pontuacao(item):
            a, c = item
            contem = cv2.pointPolygonTest(c, centro, False) >= 0
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            else:
                cx, cy = centro
            dist = ((cx - centro[0]) ** 2 + (cy - centro[1]) ** 2) ** 0.5
            return (contem, -dist, a)

        return max(candidatos, key=pontuacao)[1]
    except Exception:
        return None


# ------------------------------------------------------------
# SEGMENTAÇÃO DA TORA POR COR (madeira x fundo)
# ------------------------------------------------------------
# Otsu em escala de cinza separa "claro" de "escuro" — mas madeira
# descascada é CLARA e uma mesa cinza/branca também é. Foi o que
# aconteceu nas primeiras demos: o contorno seguia sombras e a borda da
# mesa, e todos os indicadores herdavam o erro.
#
# O que distingue madeira (e casca) de mesa, parede, chão de concreto ou
# metal da garra não é o brilho, é a SATURAÇÃO: madeira é creme/amarela/
# marrom (colorida), o fundo industrial é acinzentado. Então:
#   1. Otsu na SATURAÇÃO (adapta-se à iluminação) + matiz na faixa
#      amarelo/laranja/marrom;
#   2. fecha buracos (o anel de casca envolve o miolo);
#   3. fica com o componente que contém o centro do frame/ROI.
# Se o fundo também for colorido (mesa de madeira, terra), a saturação
# não separa -- aí cai no Otsu de brilho de extrair_contorno_tora.
# ------------------------------------------------------------

def segmentar_tora(frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Devolve (mascara, contorno) da tora, ou (None, None). Ver segmentar_tora_origem."""
    mask, contorno, _ = segmentar_tora_origem(frame)
    return mask, contorno


def segmentar_tora_origem(frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Devolve (mascara, contorno, origem). `origem` diz QUAL caminho achou a
    tora: "cor" (saturação + matiz de madeira — confiável) ou "brilho"
    (Otsu de brilho, reserva para fundo colorido — acha blob em qualquer
    coisa, inclusive num teclado). O portão validar_tora usa isso.
    """
    if frame is None or frame.size == 0:
        return None, None, "nenhuma"
    try:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        sat_b = cv2.GaussianBlur(sat, (9, 9), 0)

        limiar_s, _ = cv2.threshold(sat_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Fundo colorido demais (limiar alto) ou frame sem cor nenhuma:
        # saturação não separa -> deixa o Otsu de brilho decidir.
        if limiar_s < 18 or limiar_s > 140:
            c = extrair_contorno_tora(frame)
            return (_mascara_do_contorno(frame.shape, c), c, "brilho") if c is not None else (None, None, "nenhuma")

        colorido = sat_b >= max(limiar_s, 25)
        # Madeira/casca: amarelo, laranja, marrom, vermelho-escuro (OpenCV H em 0..180)
        tom_madeira = (hue <= 35) | (hue >= 165)
        nao_preto = val >= 20

        # Qual das duas classes de saturação é a tora? A que está no CENTRO do
        # frame (a câmera aponta para a tora). Normalmente é a saturada
        # (madeira creme sobre mesa cinza); mas madeira pálida sobre mesa
        # marrom ou terra inverte -- e aí o fundo é que é "colorido".
        cy0, cx0 = h // 2, w // 2
        dy, dx = max(2, h // 20), max(2, w // 20)
        centro_colorido = np.mean(colorido[cy0 - dy:cy0 + dy, cx0 - dx:cx0 + dx]) >= 0.5
        if centro_colorido:
            mask = colorido & tom_madeira & nao_preto
        else:
            mask = (~colorido) & nao_preto
        mask = mask.astype(np.uint8) * 255


        k_pequeno = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_pequeno)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_grande)

        # Se "cor de madeira" cobre quase o frame inteiro, o fundo também é
        # madeira/terra: a cor não separa nada -> Otsu de brilho.
        if np.count_nonzero(mask) > 0.85 * h * w:
            c = extrair_contorno_tora(frame)
            return (_mascara_do_contorno(frame.shape, c), c, "brilho") if c is not None else (None, None, "nenhuma")

        # Componente conectado que contém o centro (é para onde a câmera aponta);
        # senão, o de centroide mais próximo do centro. Exige área >= 3% do frame.
        n, rotulos, stats, centroides = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None, None, "nenhuma"
        cx0, cy0 = w / 2.0, h / 2.0
        area_min = 0.03 * h * w
        escolhido = None
        rotulo_centro = rotulos[int(cy0), int(cx0)]
        if rotulo_centro > 0 and stats[rotulo_centro, cv2.CC_STAT_AREA] >= area_min:
            escolhido = rotulo_centro
        else:
            melhor = None
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < area_min:
                    continue
                d = (centroides[i][0] - cx0) ** 2 + (centroides[i][1] - cy0) ** 2
                if melhor is None or d < melhor[0]:
                    melhor = (d, i)
            if melhor is not None:
                escolhido = melhor[1]
        if escolhido is None:
            return None, None, "nenhuma"

        mask = (rotulos == escolhido).astype(np.uint8) * 255

        # Preenche buracos internos (miolo claro cercado pela casca, nós escuros)
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None, None, "nenhuma"
        contorno = max(contornos, key=cv2.contourArea)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contorno], -1, 255, -1)
        return mask, contorno, "cor"
    except Exception:
        return None, None, "nenhuma"


# ------------------------------------------------------------
# PORTÃO "ISSO É MADEIRA?"
# ------------------------------------------------------------
# A segmentação acha "o blob do centro" -- num teclado, num notebook, numa
# mão, ela também acha. Antes, isso bastava para o sistema medir diâmetro,
# casca e abrir um evento de tora em cima de um teclado. Este portão é a
# segunda opinião, com critérios baratos e explicáveis para a banca:
#   1. ORIGEM: só vale segmentação pela cor (saturação + matiz quente). O
#      fallback de brilho acha blob em qualquer coisa -- não vale para
#      dizer "tem tora".
#   2. COR DE MADEIRA: a maior parte da máscara tem matiz quente e alguma
#      saturação (creme, amarelo, marrom). Teclado/notebook/mesa cinza: não.
#   3. SOLIDEZ (área / casco convexo): tora, de lado ou de frente, é
#      convexa (>= 0,9). Mão aberta, cabos, objetos irregulares: ~0,6-0,75.
#   4. PELE: fração de pixels com matiz de pele (vermelho-rosado) alta =
#      mão, não madeira. Casca marrom e madeira creme ficam fora dessa faixa.
#   5. PROPORÇÃO: lado maior / lado menor > 6 é borda de mesa ou cabo.
# Qualquer critério reprovado => "Sem tora": nada é medido, nada é gravado.
# ------------------------------------------------------------

def validar_tora(frame: np.ndarray, mask: np.ndarray | None, contorno, origem: str, cfg: Config) -> tuple[bool, str, dict]:
    """Devolve (eh_tora, motivo_se_nao, metricas)."""
    met: dict = {"origem": origem}
    if mask is None or contorno is None or len(contorno) < 3:
        return False, "sem contorno", met
    try:
        if cfg.tora_exigir_cor and origem != "cor":
            return False, "sem cor de madeira", met

        (_, _), (a, b), _ = cv2.minAreaRect(np.asarray(contorno, dtype=np.float32).reshape(-1, 1, 2))
        menor, maior = max(1.0, min(a, b)), max(a, b)
        met["razao"] = round(maior / menor, 2)
        if maior / menor > cfg.tora_razao_max:
            return False, f"proporcao {maior / menor:.1f}", met

        area = float(cv2.contourArea(contorno))
        casco = float(cv2.contourArea(cv2.convexHull(contorno)))
        solidez = area / casco if casco > 0 else 0.0
        met["solidez"] = round(solidez, 3)
        if solidez < cfg.tora_solidez_min:
            return False, f"solidez {solidez:.2f}", met

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dentro = mask > 0
        hue = hsv[..., 0][dentro].astype(np.int32)
        sat = hsv[..., 1][dentro].astype(np.int32)
        val = hsv[..., 2][dentro].astype(np.int32)
        if hue.size < 100:
            return False, "mascara minuscula", met

        # Pele: matiz 0-10 (ou 170-180), saturação média, claro. Casca é
        # mais escura e mais laranja; madeira exposta é mais amarela.
        pele = ((hue <= 10) | (hue >= 172)) & (sat >= 35) & (sat <= 170) & (val >= 90)
        fracao_pele = float(np.mean(pele))
        met["pele"] = round(fracao_pele, 3)
        if fracao_pele > cfg.tora_fracao_pele_max:
            return False, f"pele {fracao_pele:.0%}", met

        # Madeira = matiz quente com alguma saturação (casca, madeira amarelada)
        # OU pálida e clara (madeira exposta creme: nesses pixels a saturação
        # é baixa e o matiz vira ruído -- não dá para exigir "quente").
        quente = ((hue <= 35) | (hue >= 165)) & (sat >= 20) & (val >= 20)
        palida = (sat < 45) & (val >= 110)
        madeira = quente | palida
        fracao_madeira = float(np.mean(madeira))
        met["madeira"] = round(fracao_madeira, 3)
        if fracao_madeira < cfg.tora_fracao_madeira_min:
            return False, f"cor de madeira {fracao_madeira:.0%}", met

        return True, "", met
    except Exception as e:
        return False, f"erro {type(e).__name__}", met


def classificar_vista(contorno) -> tuple[str, float, float, float]:
    """
    Diz se a câmera está vendo a SEÇÃO (face cortada, rodela) ou a
    LATERAL da tora, e devolve (vista, lado_menor_px, lado_maior_px, angulo).

    Usa o retângulo de área mínima (rotacionado), então tora inclinada não
    vira "mais larga". Razão lado_maior/lado_menor < 1.5 => seção (é
    aproximadamente redonda); acima => lateral (alongada).

    Importa porque os indicadores mudam de significado: numa seção mede-se
    diâmetro e casca (anel); comprimento e tortuosidade só existem na
    lateral. O sistema diz qual é qual em vez de inventar número.
    """
    # 3 pontos bastam: CHAIN_APPROX_SIMPLE reduz uma tora que atravessa o
    # frame a um retângulo de 4 pontos — o caso mais comum na garra.
    if contorno is None or len(contorno) < 3:
        return "desconhecida", 0.0, 0.0, 0.0
    (_, _), (a, b), ang = cv2.minAreaRect(np.asarray(contorno, dtype=np.float32).reshape(-1, 1, 2))
    menor, maior = (a, b) if a <= b else (b, a)
    if menor < 1:
        return "desconhecida", 0.0, 0.0, 0.0
    vista = "secao" if maior / menor < 1.5 else "lateral"
    return vista, float(menor), float(maior), float(ang)


def mascara_interior(mask: np.ndarray, fracao: float = 0.09) -> np.ndarray:
    """Tora sem a faixa de borda (casca): erosão proporcional ao tamanho da peça."""
    x, y, w, h = cv2.boundingRect(mask)
    k = max(3, int(fracao * min(w, h)))
    k += (k % 2 == 0)
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


# ------------------------------------------------------------
# ESCALA AUTOMÁTICA — marcador ArUco de tamanho conhecido
# ------------------------------------------------------------
# Em máquina de verdade ninguém vai clicar em régua. A escala px->cm vem,
# em ordem de preferência:
#   1. do próprio harvester (o cabeçote já mede diâmetro/comprimento com
#      sensores nas facas e no rolo — o dado sai no StanForD .hpr);
#   2. de um MARCADOR de tamanho conhecido fixo no campo de visão (na
#      garra ou na maquete): OpenCV detecta o ArUco em cada frame e
#      calcula cm/px sozinho, inclusive se a câmera mudar de posição;
#   3. de cm_por_px fixo no config (calibração de fábrica da montagem);
#   4. da fórmula de GSD com óptica genérica (ordem de grandeza).
# Gere o marcador com `python calibrar.py --gerar-marcador` e imprima.
# ------------------------------------------------------------
_ARUCO_DETECTOR = None


def _detector_aruco():
    global _ARUCO_DETECTOR
    if _ARUCO_DETECTOR is None:
        dicionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        _ARUCO_DETECTOR = cv2.aruco.ArucoDetector(dicionario, cv2.aruco.DetectorParameters())
    return _ARUCO_DETECTOR


def escala_por_marcador(frame: np.ndarray, lado_marcador_cm: float) -> tuple[float | None, np.ndarray | None]:
    """
    Procura um marcador ArUco (DICT_4X4_50) no frame. Se achar, devolve
    (cm_por_px, cantos) usando a média dos 4 lados em pixels; senão (None, None).
    """
    if not lado_marcador_cm or lado_marcador_cm <= 0:
        return None, None
    try:
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cantos, ids, _ = _detector_aruco().detectMarkers(cinza)
        if ids is None or len(cantos) == 0:
            return None, None
        # Se houver mais de um, usa o maior (mais perto do plano da tora / mais confiável)
        melhor = max(cantos, key=lambda c: cv2.contourArea(c.reshape(-1, 1, 2).astype(np.float32)))
        pts = melhor.reshape(4, 2)
        lados = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
        lado_px = float(np.mean(lados))
        if lado_px < 8:
            return None, None
        return lado_marcador_cm / lado_px, pts.astype(np.int32)
    except Exception:
        return None, None


# ------------------------------------------------------------
# RACHADURA RADIAL NA SEÇÃO — detector clássico (complementa o YOLO)
# ------------------------------------------------------------
# O modelo foi treinado em madeira serrada: a classe "Crack" dele é
# rachadura longitudinal em tábua aplainada. Rachadura RADIAL numa face
# de corte (rodela) é outra imagem — e ele não a vê nem com confiança
# 0.15. Até o fine-tuning com fotos de eucalipto, este detector cobre o
# caso com visão clássica, que aqui é até mais adequada:
#   1. black-hat morfológico realça estruturas finas MAIS ESCURAS que a
#      vizinhança (rachadura, não anel de crescimento suave);
#   2. fica só com o 1% mais forte, dentro do miolo (sem a casca);
#   3. cada componente vira candidato se for LONGO (>= 12% do diâmetro),
#      FINO (alongamento >= 6), RETO (resíduo do ajuste de reta pequeno)
#      e a reta passar PERTO DO CENTRO — rachadura de secagem é radial;
#      anel de crescimento é um arco tangencial, longe do centro e curvo.
# Sai como defeito "Crack" (mesmo nome da classe do modelo), com
# origem="opencv" para a tela distinguir.
# ------------------------------------------------------------

def detectar_rachaduras_secao(frame: np.ndarray, mask: np.ndarray) -> list:
    try:
        x, y, w, h = cv2.boundingRect(mask)
        diam = float(max(w, h))
        if diam < 40:
            return []
        m = cv2.moments(mask)
        if m["m00"] <= 0:
            return []
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        interior = mascara_interior(mask, 0.08)
        k = max(7, int(0.05 * diam))
        k += (k % 2 == 0)
        relevo = cv2.morphologyEx(cinza, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        relevo[interior == 0] = 0
        valores = relevo[interior > 0]
        if valores.size < 500:
            return []
        limiar = max(18.0, float(np.percentile(valores, 99.0)))
        binario = (relevo >= limiar).astype(np.uint8) * 255
        binario = cv2.morphologyEx(binario, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        area_frame = float(frame.shape[0] * frame.shape[1])
        saida = []
        contornos, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contornos:
            if len(c) < 10:
                continue
            (_, _), (a, b), _ = cv2.minAreaRect(c)
            comprimento, largura = max(a, b), max(1.0, min(a, b))
            if comprimento < 0.12 * diam or comprimento / largura < 6.0:
                continue
            pts = c.reshape(-1, 2).astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            # resíduo do ajuste de reta (retidão) e distância do centro à reta
            residuo = float(np.mean(np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)))
            dist_centro = abs((cx - x0) * vy - (cy - y0) * vx)
            # Medido nas capturas: rachadura real residuo/L ~0.012, dist/diam ~0.02;
            # arco de anel de crescimento residuo/L ~0.042, dist/diam ~0.22.
            if residuo > 0.025 * comprimento or dist_centro > 0.12 * diam:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            extensao = min(1.0, comprimento / diam)
            saida.append({
                "tipo_defeito": "Crack",
                "pos_x": float(bx), "pos_y": float(by),
                "largura": float(bw), "altura": float(bh),
                # área da LINHA, não da caixa: rachadura é fina, não "extensa"
                "area_relativa": round(float(cv2.contourArea(c)) / area_frame, 4),
                "confianca": round(0.55 + 0.4 * extensao, 4),
                "origem": "opencv",
            })
        return saida
    except Exception:
        return []


# ------------------------------------------------------------
# ROI — recorte fixo do frame antes da inferência
# ------------------------------------------------------------
def roi_em_pixels(frame_shape: tuple, cfg: Config) -> tuple[int, int, int, int]:
    """Converte a ROI do config (frações) para (x, y, w, h) em pixels do frame."""
    h, w = frame_shape[:2]
    if not cfg.roi:
        return 0, 0, int(w), int(h)
    rx, ry, rw, rh = cfg.roi
    x = int(round(rx * w))
    y = int(round(ry * h))
    ww = max(1, int(round(rw * w)))
    hh = max(1, int(round(rh * h)))
    return x, y, min(ww, w - x), min(hh, h - y)


def recortar_roi(frame: np.ndarray, cfg: Config) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Devolve (recorte, (x, y, w, h)). Sem ROI configurada, devolve o frame inteiro."""
    x, y, w, h = roi_em_pixels(frame.shape, cfg)
    if (x, y, w, h) == (0, 0, frame.shape[1], frame.shape[0]):
        return frame, (x, y, w, h)
    return frame[y:y + h, x:x + w], (x, y, w, h)


def deslocar_defeitos(defeitos: list, dx: int, dy: int) -> list:
    """Leva caixas medidas no recorte da ROI de volta às coordenadas do frame inteiro."""
    if not dx and not dy:
        return defeitos
    return [
        {**d, "pos_x": round(d["pos_x"] + dx, 2), "pos_y": round(d["pos_y"] + dy, 2)}
        for d in defeitos
    ]


def calcular_tortuosidade(contorno, frame_shape: tuple | None = None) -> float:
    """
    Tortuosidade da tora em % — "flecha" máxima do EIXO da tora em relação
    à corda que liga suas duas pontas, dividida pelo comprimento:

        tortuosidade = (desvio máximo do eixo / comprimento) x 100

    É a definição usada em campo na avaliação de fuste (flecha/comprimento):
    0 = perfeitamente reta; 5 já é uma curvatura visível; > 10 é tora
    torta de verdade. A Embrapa registra, por exemplo, que o clone GG100
    tende a tortuosidade de fuste — é esse tipo de coisa que o índice deve
    pegar.

    Como o eixo é obtido: preenche o contorno, percorre a tora ao longo
    do seu lado mais comprido e, em cada fatia transversal, marca o ponto
    médio entre as duas bordas. Esses pontos médios formam a linha central
    (eixo). Uma tora reta tem eixo reto; uma tora curva, eixo curvo.

    A versão anterior media a distância dos pontos do CONTORNO ao eixo —
    ou seja, media meio diâmetro, não curvatura: uma tora reta e grossa
    saía com índice ~60. Foi substituída.
    """
    if contorno is None or len(contorno) < 3:
        return 0.0

    try:
        pts = np.asarray(contorno, dtype=np.int32).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(pts)
        if w < 5 or h < 5:
            return 0.0
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [pts - np.array([x, y], dtype=np.int32)], -1, 255, -1)

        # Percorre ao longo do lado mais comprido; transpõe se a tora
        # estiver "deitada" para o código ser um só.
        if w > h:
            mask = mask.T
        comprimento = mask.shape[0]

        eixo = []
        for i in range(comprimento):
            cols = np.flatnonzero(mask[i])
            if cols.size:
                eixo.append((i, (cols[0] + cols[-1]) / 2.0))
        if len(eixo) < 10:
            return 0.0

        eixo_arr = np.asarray(eixo, dtype=np.float64)
        # Ignora 5% de cada ponta: as extremidades do contorno (corte da
        # tora, garra) distorcem o ponto médio sem serem curvatura.
        corte = max(1, int(0.05 * len(eixo_arr)))
        eixo_arr = eixo_arr[corte:-corte] if len(eixo_arr) > 2 * corte + 10 else eixo_arr

        p0, p1 = eixo_arr[0], eixo_arr[-1]
        corda = p1 - p0
        norma = float(np.hypot(*corda))
        if norma < 1.0:
            return 0.0
        # Distância perpendicular de cada ponto do eixo à corda p0->p1
        desvios = np.abs((eixo_arr[:, 0] - p0[0]) * corda[1] - (eixo_arr[:, 1] - p0[1]) * corda[0]) / norma
        indice = float(desvios.max()) / norma * 100.0
        return round(min(indice, 100.0), 2)
    except Exception:
        return 0.0


def extrair_defeitos_yolo(resultado, cfg: Config, frame_shape: tuple | None = None) -> list:
    """Converte as detecções do YOLO em uma lista de defeitos estruturados com área relativa."""
    defeitos = []
    nomes_classes = resultado.names

    img_h, img_w = frame_shape[:2] if frame_shape is not None else (640, 640)
    area_total_img = float(img_h * img_w)

    for box in resultado.boxes:
        classe_id = int(box.cls[0])
        tipo_defeito = nomes_classes.get(classe_id, "wood_defect")
        confianca = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        largura = max(0.0, x2 - x1)
        altura = max(0.0, y2 - y1)
        area_box = largura * altura
        area_relativa = round(area_box / area_total_img, 4)

        defeitos.append({
            "tipo_defeito": tipo_defeito,
            "pos_x": round(x1, 2),
            "pos_y": round(y1, 2),
            "largura": round(largura, 2),
            "altura": round(altura, 2),
            "area_relativa": area_relativa,
            "confianca": round(confianca, 4),
        })

    return defeitos


# ============================================================
# REGRA DE NEGÓCIO — CLASSIFICAÇÃO FINAL DA TORA
# ============================================================

# ------------------------------------------------------------
# SEVERIDADE POR TIPO DE DEFEITO
# ------------------------------------------------------------
# Nem todo defeito pesa igual para a indústria de celulose. Esta
# graduação é o que diferencia a triagem de um simples "detectou
# algo = reprovado".
#
# IMPORTANTE: os nomes aqui são EXATAMENTE as classes do modelo
# treinado (data.yaml, nc=8). A versão anterior desta função
# procurava por "apodrecimento"/"praga"/"rot", que NÃO existem no
# modelo — ou seja, aquela regra nunca disparava.
#
# Justificativa de cada nível:
#   GRAVE
#     Dead_Knot       nó morto: não está integrado à fibra ao redor,
#                     pode soltar e virar buraco no processamento
#     Knot_missing    nó já ausente: o buraco existe
#     knot_with_crack nó + rachadura: combina dois problemas
#     resin           bolsa de resina: resina é extrativo, e o próprio
#                     enunciado do desafio liga teor de extrativos ao
#                     consumo de químicos para separar celulose da
#                     lignina
#   MODERADO
#     Crack           rachadura: gravidade varia com extensão/profundidade
#     Marrow          medula: tecido mole, fibra de baixa qualidade,
#                     mas área pequena
#     Quartzity       inclusão mineral: danifica lâmina, não a fibra
#   LEVE
#     Live_Knot       nó vivo: integrado à madeira ao redor; é o defeito
#                     mais comum e o de menor impacto para celulose
#
# Estes pesos são uma DECISÃO DE PROJETO baseada em características
# conhecidas da madeira, não uma norma técnica publicada. Se a
# Suzano/JD fornecer o critério de triagem oficial deles, é só
# ajustar os conjuntos abaixo — nada mais no código muda.
# ------------------------------------------------------------
DEFEITOS_GRAVES = ("Dead_Knot", "Knot_missing", "knot_with_crack", "resin")
DEFEITOS_MODERADOS = ("Crack", "Marrow", "Quartzity")
DEFEITOS_LEVES = ("Live_Knot",)

PESO_SEVERIDADE = {
    **{d: 1.0 for d in DEFEITOS_GRAVES},
    **{d: 0.5 for d in DEFEITOS_MODERADOS},
    **{d: 0.2 for d in DEFEITOS_LEVES},
}
PESO_PADRAO = 0.5  # classe desconhecida: trata como moderada, não ignora


def severidade_do_defeito(tipo_defeito: str) -> float:
    """Peso de 0.2 (leve) a 1.0 (grave) para um tipo de defeito."""
    return PESO_SEVERIDADE.get(tipo_defeito, PESO_PADRAO)


def calcular_confianca_saude(defeitos: list) -> float:
    """
    Score de saúde da tora (1.0 = limpa), ponderado por severidade.

    Antes, qualquer defeito descontava igual: um nó vivo detectado com
    90% de confiança derrubava a saúde tanto quanto um nó morto. Agora
    o desconto é proporcional à gravidade do defeito para o processo
    de celulose.
    """
    if not defeitos:
        return 1.0

    # O defeito mais crítico define o desconto principal...
    impacto_principal = max(
        d["confianca"] * severidade_do_defeito(d["tipo_defeito"]) for d in defeitos
    )
    # ...e a quantidade de defeitos adiciona um desconto menor,
    # limitado, para que uma tora cheia de defeitos leves ainda seja
    # penalizada — só que menos que uma com um defeito grave.
    impacto_acumulado = min(
        0.25,
        sum(d["confianca"] * severidade_do_defeito(d["tipo_defeito"]) for d in defeitos) * 0.05,
    )

    saude = 1.0 - (impacto_principal * 0.5) - impacto_acumulado
    return round(max(0.0, min(1.0, saude)), 4)


def classificar_qualidade(confianca_saude: float, defeitos: list, cfg: Config) -> str:
    """
    Triagem da tora, ponderada pela gravidade do defeito para a
    indústria de celulose:

      1. Defeito GRAVE com confiança >= 70%              -> reprovado
      2. Defeito extenso (>= 5% da área) com conf >= 65% -> reprovado
      3. Saúde abaixo de limiar_quarentena (60%)         -> reprovado
      4. Sem defeito E saúde >= limiar_aprovacao (85%)   -> aprovado
      5. Só defeitos LEVES e saúde alta                  -> aprovado
      6. Qualquer outro caso                             -> quarentena

    A regra 5 é a diferença prática do modelo ponderado: uma tora com
    apenas nós vivos (o defeito mais comum e de menor impacto) não
    precisa ir para revisão manual — antes ia, o que geraria fila de
    quarentena desnecessária em operação real.
    """
    # 1. Defeito grave com confiança alta
    tem_grave = any(
        d["tipo_defeito"] in DEFEITOS_GRAVES and d["confianca"] >= 0.70
        for d in defeitos
    )
    if tem_grave:
        return "reprovado"

    # 2. Defeito extenso (ignora leves: um nó vivo grande não reprova a tora)
    tem_defeito_extenso = any(
        d.get("area_relativa", 0.0) >= 0.05
        and d["confianca"] >= 0.65
        and d["tipo_defeito"] not in DEFEITOS_LEVES
        for d in defeitos
    )
    if tem_defeito_extenso:
        return "reprovado"

    # 3. Saúde muito baixa
    if confianca_saude < cfg.limiar_quarentena:
        return "reprovado"

    # 4. Tora limpa
    if confianca_saude >= cfg.limiar_aprovacao and len(defeitos) == 0:
        return "aprovado"

    # 5. Apenas defeitos leves, com saúde alta -> aprovado
    so_leves = defeitos and all(d["tipo_defeito"] in DEFEITOS_LEVES for d in defeitos)
    if so_leves and confianca_saude >= cfg.limiar_aprovacao:
        return "aprovado"

    # 6. Zona cinza -> revisão manual
    return "quarentena"



def gerar_hash_sha256(dados: dict) -> str:
    """Gera um hash de integridade do registro (rastreabilidade)."""
    dados_serializados = json.dumps(dados, sort_keys=True, default=str)
    return hashlib.sha256(dados_serializados.encode("utf-8")).hexdigest()


# ============================================================
# CAMADA DE BANCO DE DADOS (SQLite local)
# ============================================================

def conectar_banco(cfg: Config) -> sqlite3.Connection:
    conexao = sqlite3.connect(cfg.sqlite_path)
    conexao.execute("PRAGMA foreign_keys = ON")

    # Verifica se a tabela toras_local já existe, caso contrário carrega o schema
    tabelas = conexao.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='toras_local'").fetchall()
    if not tabelas:
        schema_path = Path(cfg.sqlite_path).parent / "Banco de dados" / "schema_sqlite.sql"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                conexao.executescript(f.read())
    return conexao


def salvar_inspecao(
    conexao: sqlite3.Connection,
    cfg: Config,
    indicadores: dict,
    defeitos: list,
    status_classificacao: str,
    confianca_media: float,
) -> str:
    """Grava a tora + indicadores + defeitos nas 3 tabelas locais."""
    cursor = conexao.cursor()

    uuid_local = str(uuid.uuid4())
    log_id = f"LOG-{datetime.now():%Y%m%d%H%M%S}-{uuid_local[:4]}"
    data_inspecao = datetime.now().isoformat(timespec="seconds")

    hash_dados = gerar_hash_sha256({
        "uuid_local": uuid_local,
        "log_id": log_id,
        "indicadores": indicadores,
        "defeitos": defeitos,
        "status": status_classificacao,
    })

    # --- 1. Insere a tora ---
    cursor.execute(
        """
        INSERT INTO toras_local
            (uuid_local, maquina_id, talhao_id, log_id, data_inspecao,
             confianca_ia, status_classificacao, hash_sha256, sync_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (uuid_local, cfg.maquina_id, cfg.talhao_id, log_id, data_inspecao,
         confianca_media, status_classificacao, hash_dados),
    )
    tora_id_local = cursor.lastrowid

    # --- 2. Insere os indicadores de qualidade ---
    for tipo, dados_indicador in indicadores.items():
        cursor.execute(
            """
            INSERT INTO indicadores_qualidade_local
                (tora_id, tipo_indicador, valor, unidade, metodo_medicao, sync_status)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (tora_id_local, tipo, dados_indicador["valor"],
             dados_indicador["unidade"], dados_indicador["metodo"]),
        )

    # --- 3. Insere os defeitos detectados ---
    for defeito in defeitos:
        cursor.execute(
            """
            INSERT INTO defeitos_detectados_local
                (tora_id, tipo_defeito, pos_x, pos_y, largura, altura, confianca, sync_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (tora_id_local, defeito["tipo_defeito"], defeito["pos_x"], defeito["pos_y"],
             defeito["largura"], defeito["altura"], defeito["confianca"]),
        )

    conexao.commit()
    return uuid_local


# ============================================================
# PIPELINE DE ANÁLISE DE UM FRAME
# ============================================================
# Tudo o que acontece entre "chegou um frame" e "temos status +
# indicadores" vive aqui, numa função pura (sem thread, sem banco, sem
# janela). O loop ao vivo (main) e o simulador sem câmera
# (tests/simular_cenario.py) chamam a MESMA função -- o que a demo mostra
# é exatamente o que o teste testa.
# ============================================================

CORES_STATUS = {"aprovado": (0, 255, 0), "quarentena": (0, 255, 255), "reprovado": (0, 0, 255), "sem_tora": (200, 200, 200)}
COR_IGNORADO = (140, 140, 140)


def carregar_modelo(cfg: Config) -> YOLO:
    """Mesma ordem de busca em todo lugar: modelo_path > OpenVINO INT8 > OpenVINO FP32 > NCNN > .pt treinado > yolov8n genérico."""
    caminho_modelo = None
    candidatos = [
        Path(cfg.modelo_path),
        Path("./models/wood_best_int8_openvino_model"),
        Path("./models/wood_best_openvino_model"),
        Path(cfg.modelo_ncnn_path),
        Path("./models/wood_best.pt"),
        Path("./yolov8n.pt"),
    ]
    for p in candidatos:
        if p.exists():
            caminho_modelo = str(p)
            break
    if caminho_modelo is None:
        caminho_modelo = "yolov8n.pt"
        print("⚠️  Nenhum modelo treinado encontrado em models/ — usando yolov8n.pt genérico (NÃO detecta defeito de madeira).")
    elif caminho_modelo.endswith(".pt"):
        print("ℹ️  Rodando o .pt direto no PyTorch (lento em CPU). Rode `python exportar_modelo.py` para gerar a versão OpenVINO (~5x mais rápida).")
    modelo = YOLO(caminho_modelo, task="detect")
    print(f"✅ Modelo carregado ({caminho_modelo})")
    return modelo


def detectar_yolo(recorte: np.ndarray, modelo: YOLO, cfg: Config) -> list:
    """Roda o modelo no recorte PRÉ-PROCESSADO (grayscale + CLAHE, igual ao treino) e devolve os defeitos brutos, em coordenadas do recorte."""
    resultados = modelo.predict(
        source=preprocessar_para_modelo(recorte),
        conf=cfg.conf_threshold,
        iou=cfg.iou_threshold,
        imgsz=cfg.imgsz,
        device="cpu",
        verbose=False,
    )
    return extrair_defeitos_yolo(resultados[0], cfg, recorte.shape)


def analisar_frame(
    frame: np.ndarray,
    modelo: "YOLO | None",
    cfg: Config,
    persistencia: "FiltroPersistencia | None" = None,
    defeitos_yolo: "list | None" = None,
    yolo_novo: bool = True,
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
    mask, contorno, origem_seg = segmentar_tora_origem(recorte)
    eh_tora, motivo_sem_tora, _ = validar_tora(recorte, mask, contorno, origem_seg, cfg)
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

    # --- 6. Escala px -> cm: marcador ArUco (automático) > config > GSD ---
    cm_por_px, marcador = escala_por_marcador(frame, cfg.marcador_aruco_cm)
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
    porcentagem_casca = calcular_porcentagem_casca(recorte, contorno)
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
        f"Escala: {metodo_escala} ({cm_por_px * 10:.2f} mm/px)" + ("" if comprimento_medido else "  *comprimento de tracamento"),
    ]

    return {
        "status": status,
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


def desenhar_analise(frame: np.ndarray, a: dict | None) -> np.ndarray:
    """
    Desenha o resultado sobre uma cópia do frame. Caixas MANTIDAS na cor do
    status; caixas IGNORADAS pelos filtros em cinza fino, com o motivo
    visível -- na demo isso mostra que o sistema viu o teclado/a mão e
    decidiu, de propósito, não contar.
    """
    vis = frame.copy()
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    if a is None:
        cv2.putText(vis, "Analisando...", (20, 40), fonte, 0.9, (255, 255, 255), 2)
        return vis

    cor = CORES_STATUS[a["status"]]
    rx, ry, rw, rh = a["roi"]
    if (rw, rh) != (frame.shape[1], frame.shape[0]):
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 1)
        cv2.putText(vis, "ROI", (rx + 4, ry + 16), fonte, 0.5, (255, 200, 0), 1)
    if a["contorno"] is not None and len(a["contorno"]) >= 3:
        cv2.drawContours(vis, [a["contorno"]], -1, (255, 255, 255), 1)
    if a.get("marcador") is not None:
        cv2.polylines(vis, [a["marcador"].reshape(-1, 1, 2)], True, (255, 0, 255), 2)
        mx, my = a["marcador"][0]
        cv2.putText(vis, "escala", (int(mx), max(15, int(my) - 6)), fonte, 0.5, (255, 0, 255), 1)

    for d in a["descartados"]:
        x1, y1 = int(d["pos_x"]), int(d["pos_y"])
        x2, y2 = int(d["pos_x"] + d["largura"]), int(d["pos_y"] + d["altura"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), COR_IGNORADO, 1)
        cv2.putText(vis, f"{d['tipo_defeito']} {d['confianca']:.2f} (ignorado)", (x1, max(15, y1 - 6)), fonte, 0.45, COR_IGNORADO, 1)
    for d in a["defeitos"]:
        x1, y1 = int(d["pos_x"]), int(d["pos_y"])
        x2, y2 = int(d["pos_x"] + d["largura"]), int(d["pos_y"] + d["altura"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), cor, 2)
        rotulo = f"{d['tipo_defeito']} {d['confianca']:.2f}" + (" (cv)" if d.get("origem") == "opencv" else "")
        cv2.putText(vis, rotulo, (x1, max(15, y1 - 6)), fonte, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, rotulo, (x1, max(15, y1 - 6)), fonte, 0.5, cor, 1, cv2.LINE_AA)

    # HUD sobre uma faixa escura semitransparente: legível sobre mesa clara e
    # sobre madeira. Fonte proporcional à largura do frame (webcam 640 px x
    # câmera industrial 1920 px).
    fator = max(0.75, min(1.0, frame.shape[1] / 1280.0))
    escalas = [0.85 * fator, 0.62 * fator, 0.52 * fator, 0.48 * fator]
    passos = [int(34 * fator), int(26 * fator), int(24 * fator), int(24 * fator)]
    # Linhas extras (ex.: estado do evento de tora) usam o estilo da última.
    while len(escalas) < len(a["hud"]):
        escalas.append(escalas[-1])
        passos.append(passos[-1])
    altura_faixa = 8 + sum(passos[: len(a["hud"])])
    faixa = vis.copy()
    cv2.rectangle(faixa, (0, 0), (frame.shape[1], altura_faixa), (0, 0, 0), -1)
    cv2.addWeighted(faixa, 0.55, vis, 0.45, 0, vis)
    y = 4
    for i, linha in enumerate(a["hud"]):
        y += passos[i] - 6
        cor_txt = cor if i == 0 else (255, 255, 255)
        cv2.putText(vis, linha, (12, y), fonte, escalas[i], cor_txt, 2 if i <= 1 else 1, cv2.LINE_AA)
        y += 6
    return vis


# ============================================================
# EVENTO DE TORA — uma tora, um registro
# ============================================================
# Antes, o loop gravava uma inspeção a cada 2 s, tivesse tora no frame ou
# não: uma tora parada 10 s na garra virava 5 registros e garra vazia
# também gerava linha. O dashboard contava "toras", mas eram janelas de
# 2 s. Aqui a unidade passa a ser a TORA:
#
#   vazio ──(tora estável por N análises)──► aberto ──(sem tora por M)──► fecha
#                                              │
#                                              └─(contorno "pulou": outra tora)─► fecha e reabre
#
# Ao fechar, consolida todos os quadros do evento:
#   - indicadores geométricos: MEDIANA (robusta a quadro ruim);
#   - vista predominante decide diâmetro; comprimento/tortuosidade só
#     dos quadros laterais, se houve algum;
#   - defeitos: UNIÃO por tipo+posição, exigindo presença em >= K quadros;
#   - saúde e status recalculados sobre os defeitos consolidados (mesmas
#     funções do quadro único — a regra de negócio não muda);
#   - volume e massa recalculados das medianas.
# ============================================================

def _bbox_contorno(contorno) -> tuple[int, int, int, int] | None:
    if contorno is None or len(contorno) < 3:
        return None
    x, y, w, h = cv2.boundingRect(np.asarray(contorno, dtype=np.int32).reshape(-1, 1, 2))
    return int(x), int(y), int(w), int(h)


def _iou_bbox(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = float(ix * iy)
    uniao = float(aw * ah + bw * bh) - inter
    return inter / uniao if uniao > 0 else 0.0


def consolidar_defeitos(quadros: list[list[dict]], min_quadros: int, tolerancia_px: float = 80.0) -> list:
    """
    União dos defeitos de todos os quadros de um evento: agrupa por tipo e
    posição (centro a <= tolerancia_px), fica com a detecção de maior
    confiança de cada grupo e descarta grupos vistos em menos de
    `min_quadros` quadros (um defeito real reaparece; ruído não).
    """
    grupos: list[dict] = []  # {"rep": defeito, "quadros": set(idx)}
    for i, defeitos in enumerate(quadros):
        for d in defeitos:
            cx = d["pos_x"] + d["largura"] / 2.0
            cy = d["pos_y"] + d["altura"] / 2.0
            alvo = None
            for g in grupos:
                r = g["rep"]
                if r["tipo_defeito"] != d["tipo_defeito"]:
                    continue
                rx = r["pos_x"] + r["largura"] / 2.0
                ry = r["pos_y"] + r["altura"] / 2.0
                if ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 <= tolerancia_px:
                    alvo = g
                    break
            if alvo is None:
                grupos.append({"rep": dict(d), "quadros": {i}})
            else:
                alvo["quadros"].add(i)
                if d["confianca"] > alvo["rep"]["confianca"]:
                    alvo["rep"] = dict(d)
    minimo = min(min_quadros, max(1, len(quadros)))
    return [g["rep"] for g in grupos if len(g["quadros"]) >= minimo]


def consolidar_evento(analises: list[dict], cfg: Config) -> dict:
    """
    Reduz as análises de um evento (uma tora) a UM resultado com a mesma
    forma de `analisar_frame` (status, confianca_saude, defeitos,
    indicadores, vista) — é o que vai para salvar_inspecao.
    """
    vistas = [a["vista"] for a in analises]
    vista = max(("secao", "lateral", "desconhecida"), key=vistas.count)
    da_vista = [a for a in analises if a["vista"] == vista] or analises
    laterais = [a for a in analises if a["vista"] == "lateral"]

    def mediana(lista: list[dict], chave: str) -> float:
        return float(np.median([a["indicadores"][chave]["valor"] for a in lista]))

    def metodo_mais_comum(lista: list[dict], chave: str) -> str:
        ms = [a["indicadores"][chave]["metodo"] for a in lista]
        return max(set(ms), key=ms.count)

    diametro_cm = round(mediana(da_vista, "diametro"), 2)
    metodo_diam = metodo_mais_comum(da_vista, "diametro")
    if laterais:
        comprimento_cm = round(mediana(laterais, "altura"), 2)
        metodo_compr = metodo_mais_comum(laterais, "altura")
        tortuosidade = round(mediana(laterais, "tortuosidade"), 2)
        metodo_tort = "opencv_eixo_flecha"
        comprimento_medido = True
    else:
        comprimento_cm = float(cfg.comprimento_corte_cm)
        metodo_compr = "comprimento_tracamento_config"
        tortuosidade = 0.0
        metodo_tort = "nao_aplicavel_secao"
        comprimento_medido = False
    porcentagem_casca = round(mediana(analises, "porcentagem_casca"), 1)
    densidade = float(analises[-1]["indicadores"]["densidade"]["valor"])

    defeitos = consolidar_defeitos([a["defeitos"] for a in analises], cfg.evento_defeito_min_quadros)
    confianca_saude = calcular_confianca_saude(defeitos)
    status = classificar_qualidade(confianca_saude, defeitos, cfg)
    volume_util_m3 = calcular_volume_m3(diametro_cm, comprimento_cm, confianca_saude)
    massa_seca_kg = calcular_massa_seca_kg(volume_util_m3, densidade)

    indicadores = {
        "densidade": dict(analises[-1]["indicadores"]["densidade"]),
        "altura": {"valor": comprimento_cm, "unidade": "cm", "metodo": metodo_compr},
        "diametro": {"valor": diametro_cm, "unidade": "cm", "metodo": metodo_diam},
        "tortuosidade": {"valor": tortuosidade, "unidade": "indice", "metodo": metodo_tort},
        "porcentagem_casca": {"valor": porcentagem_casca, "unidade": "%", "metodo": "opencv_otsu_casca_residual"},
        "volume_util": {"valor": volume_util_m3, "unidade": "m3", "metodo": "cilindro_" + ("medido" if comprimento_medido else "diam_medido_compr_config")},
        "massa_seca": {"valor": massa_seca_kg, "unidade": "kg", "metodo": f"volume_x_densidade_clone_{cfg.clone_id}"},
        "apodrecimento_pragas": {"valor": round(confianca_saude * 100, 2), "unidade": "%", "metodo": "yolo_severidade"},
    }
    return {
        "status": status,
        "confianca_saude": confianca_saude,
        "defeitos": defeitos,
        "descartados": [],
        "indicadores": indicadores,
        "vista": vista,
        "quadros": len(analises),
        "quadros_laterais": len(laterais),
    }


class RastreadorTora:
    """
    Máquina de estados que transforma a sequência de análises (uma por
    quadro) em EVENTOS de tora. `atualizar(a)` devolve o evento consolidado
    quando um fecha, senão None. Sem modelo, sem banco: só decide "é a
    mesma tora?" e agrega — testável com análises sintéticas.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.numero = 0                 # toras fechadas até agora
        self.analises: list[dict] = []  # quadros do evento aberto (ou candidatos a abrir)
        self.aberto = False
        self.inicio = 0.0
        self.ausentes = 0
        self.trocas = 0
        self.bbox_anterior: tuple | None = None
        self.descartados_ruido = 0

    @staticmethod
    def _presente(a: dict) -> bool:
        return a.get("eh_tora", True) and a.get("contorno") is not None and a.get("vista") != "desconhecida"

    def _fechar(self) -> dict | None:
        analises, self.analises = self.analises, []
        self.aberto = False
        self.ausentes = 0
        self.trocas = 0
        self.bbox_anterior = None
        if len(analises) < self.cfg.evento_frames_minimo:
            self.descartados_ruido += 1
            return None
        self.numero += 1
        evento = consolidar_evento(analises, self.cfg)
        evento["numero"] = self.numero
        evento["duracao_s"] = round(time.monotonic() - self.inicio, 2)
        return evento

    def atualizar(self, a: dict) -> dict | None:
        cfg = self.cfg
        presente = self._presente(a)
        fechado = None

        if not self.aberto:
            if presente:
                self.analises.append(a)
                if len(self.analises) == 1:
                    self.inicio = time.monotonic()
                if len(self.analises) >= cfg.evento_frames_abrir:
                    self.aberto = True
                    self.bbox_anterior = _bbox_contorno(a["contorno"])
            else:
                self.analises = []  # candidato não estabilizou
            return None

        if not presente:
            self.ausentes += 1
            if self.ausentes >= cfg.evento_frames_fechar:
                fechado = self._fechar()
            return fechado

        self.ausentes = 0
        bbox = _bbox_contorno(a["contorno"])
        if self.bbox_anterior is not None and bbox is not None and _iou_bbox(bbox, self.bbox_anterior) < cfg.evento_iou_troca:
            self.trocas += 1
        else:
            self.trocas = 0
        if self.trocas >= cfg.evento_frames_troca:
            # Contorno pulou por vários quadros: é outra tora. Os quadros da
            # "troca" pertencem à nova; tira-os da antiga antes de fechar.
            novos = self.analises[-(cfg.evento_frames_troca - 1):] if cfg.evento_frames_troca > 1 else []
            self.analises = self.analises[: len(self.analises) - len(novos)]
            fechado = self._fechar()
            self.analises = novos + [a]
            self.inicio = time.monotonic()
            self.aberto = len(self.analises) >= cfg.evento_frames_abrir
            self.bbox_anterior = bbox
            return fechado

        self.analises.append(a)
        # A referência só acompanha quadros que BATERAM com a tora atual; num
        # quadro suspeito ela fica parada, senão a "troca" nunca acumula.
        if self.trocas == 0:
            self.bbox_anterior = bbox
        return None

    def descricao(self) -> str:
        """Linha de HUD: estado atual do rastreador."""
        if self.aberto:
            return f"Tora #{self.numero + 1} em analise | {len(self.analises)} quadros | {time.monotonic() - self.inicio:.1f}s"
        if self.analises:
            return f"Tora entrando... ({len(self.analises)}/{self.cfg.evento_frames_abrir})"
        return f"Aguardando tora | {self.numero} gravadas"


# ============================================================
# CÂMERA AO VIVO — empurra o frame anotado para o dashboard
# ============================================================
# Direção campo -> central, como o sync. Cada quadro é um POST HTTP com o
# JPEG no corpo (urllib, sem dependência nova); o servidor do dashboard
# guarda só o último quadro em memória e o serve como MJPEG ao navegador.
# Nada disso passa pelo Postgres. Falhou a rede: espera 5 s e tenta de
# novo, avisando UMA vez no console (e uma vez quando volta) — nunca
# bloqueia captura, modelo ou gravação.
# ============================================================

def _token_stream() -> str:
    """STREAM_TOKEN do ambiente/.env — o mesmo valor configurado no dashboard."""
    import os
    try:
        from dotenv import load_dotenv  # já está no requirements (sync_daemon)
        load_dotenv()
    except ImportError:
        pass
    return os.getenv("STREAM_TOKEN", "").strip()


def codificar_quadro_stream(frame: np.ndarray, analise: dict | None, cfg: Config) -> bytes | None:
    """Frame anotado (mesmo desenho da janela --gui), reduzido e em JPEG."""
    vis = desenhar_analise(frame, analise)
    if cfg.stream_largura_px and vis.shape[1] > cfg.stream_largura_px:
        escala = cfg.stream_largura_px / float(vis.shape[1])
        vis = cv2.resize(vis, (cfg.stream_largura_px, int(round(vis.shape[0] * escala))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, cfg.stream_jpeg_qualidade)))])
    return buf.tobytes() if ok else None


def enviar_quadro_stream(jpeg: bytes, cfg: Config, token: str, analise: dict | None, timeout_s: float = 2.0) -> None:
    """Um POST; levanta exceção em qualquer falha (quem chama decide o recuo)."""
    import urllib.request
    cabecalhos = {
        "Content-Type": "image/jpeg",
        "X-Stream-Token": token,
        "X-Maquina-Id": cfg.maquina_id,
        "X-Status": (analise or {}).get("status", "") or "",
        "X-Vista": (analise or {}).get("vista", "") or "",
    }
    req = urllib.request.Request(cfg.stream_url, data=jpeg, method="POST", headers=cabecalhos)
    with urllib.request.urlopen(req, timeout=timeout_s):
        pass


def loop_stream(cfg: Config, estado: dict, lock: threading.Lock, parar: threading.Event) -> None:
    """
    Corpo da thread de transmissão. Lê `estado["frame"]`/`estado["analise"]`
    (os mesmos que a janela usa), codifica e envia a `stream_fps`.
    """
    import urllib.error

    token = _token_stream()
    if not token:
        print("⚠️  stream_url configurado mas STREAM_TOKEN vazio (.env) — câmera ao vivo desligada.")
        return

    intervalo = 1.0 / max(0.5, float(cfg.stream_fps))
    recuo_falha = 5.0
    proxima = 0.0
    online: bool | None = None  # None = ainda não tentou
    print(f"📡 Câmera ao vivo: enviando {cfg.stream_fps:g} fps para {cfg.stream_url}")

    while not parar.is_set():
        agora = time.monotonic()
        if agora < proxima:
            time.sleep(min(0.05, proxima - agora))
            continue
        with lock:
            frame = None if estado["frame"] is None else estado["frame"].copy()
            analise = estado["analise"]
        if frame is None:
            time.sleep(0.05)
            continue
        try:
            jpeg = codificar_quadro_stream(frame, analise, cfg)
            if jpeg is None:
                proxima = agora + intervalo
                continue
            enviar_quadro_stream(jpeg, cfg, token, analise)
            if online is not True:
                print("📡 Câmera ao vivo: conectada ao dashboard.")
                online = True
            proxima = agora + intervalo
        except urllib.error.HTTPError as e:
            # Servidor respondeu, mas recusou: é configuração, não rede.
            # 401 = token diferente nos dois .env; 503 = STREAM_TOKEN ausente lá.
            if online is not False:
                print(f"📡 Dashboard recusou o quadro (HTTP {e.code}) — confira STREAM_TOKEN nos dois .env. Tentando de novo a cada 30 s.")
                online = False
            proxima = agora + 30.0
        except Exception as e:
            if online is not False:
                print(f"📡 Dashboard inacessível ({type(e).__name__}) — inspeção continua normalmente; tentando de novo a cada {recuo_falha:.0f} s.")
                online = False
            proxima = agora + recuo_falha


# ============================================================
# FONTES DE VÍDEO — câmera, arquivo de vídeo ou pasta de imagens
# ============================================================
# `--fonte` aceita as três. As duas últimas existem para (a) testar sem a
# tora física em mãos e (b) ter um PLANO B na apresentação: se a webcam
# falhar no palco, `--fonte demo.mp4` roda o pipeline inteiro, de
# verdade, sobre um vídeo gravado -- não é tela mocada, é o mesmo código.
# ============================================================

class FonteImagens:
    """Emula cv2.VideoCapture sobre uma pasta de imagens, em loop, segurando cada uma por alguns segundos."""

    EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, pasta: Path, segundos_por_imagem: float = 4.0, segundos_vazio: float = 1.0):
        self.arquivos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in self.EXTENSOES)
        self.segundos = segundos_por_imagem
        # Entre uma imagem e a próxima, um trecho de frame PRETO: simula a
        # garra vazia entre duas toras, para o evento de tora fechar e abrir
        # como faria com a câmera de verdade.
        self.segundos_vazio = segundos_vazio
        self.inicio = time.monotonic()
        self._vazio = None

    def isOpened(self) -> bool:
        return bool(self.arquivos)

    def read(self):
        if not self.arquivos:
            return False, None
        t = time.monotonic() - self.inicio
        periodo = self.segundos + self.segundos_vazio
        i = int(t / periodo) % len(self.arquivos)
        frame = cv2.imread(str(self.arquivos[i]))
        time.sleep(0.03)  # ~30 fps de "vídeo" parado, sem fritar a CPU
        if frame is None:
            return False, None
        if (t % periodo) >= self.segundos:
            if self._vazio is None or self._vazio.shape != frame.shape:
                self._vazio = np.zeros_like(frame)
            return True, self._vazio
        return True, frame

    def set(self, *_):
        return True

    def release(self):
        pass


def abrir_fonte(fonte: str | None, cfg: Config):
    """Abre a câmera do config, um índice de câmera, um arquivo de vídeo ou uma pasta de imagens."""
    if fonte is None or fonte.strip() == "":
        fonte = str(cfg.camera_index)

    caminho = Path(fonte)
    if caminho.is_dir():
        print(f"🖼️  Fonte: pasta de imagens {caminho} (loop)")
        return FonteImagens(caminho)
    if caminho.is_file():
        print(f"🎞️  Fonte: vídeo {caminho}")
        return cv2.VideoCapture(str(caminho))

    indice = int(fonte)
    print(f"📷 Abrindo câmera (índice {indice})...")
    if sys.platform == "win32":
        # No Windows, o backend padrão do OpenCV às vezes demora ou falha
        # pra abrir a webcam. DirectShow é mais rápido e confiável lá.
        captura = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    else:
        captura = cv2.VideoCapture(indice)
    # Buffer de 1 frame: por padrão o OpenCV enfileira frames e o read()
    # devolve os ANTIGOS quando o processamento não acompanha — é isso que dá
    # a sensação de atraso. Com buffer 1, sempre pegamos o frame mais recente.
    captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return captura


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
    args = parser.parse_args()

    cfg = carregar_configuracao_json(args.config_file)
    if args.clone is not None:
        cfg.clone_id = args.clone
    if args.idade is not None:
        cfg.idade_talhao_anos = args.idade

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
    estado = {"frame": None, "analise": None, "yolo": None, "yolo_id": 0}
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
                if frame is None or defeitos_yolo is None:
                    time.sleep(0.01)
                    continue
                try:
                    a = analisar_frame(
                        frame, None, cfg, persistencia,
                        defeitos_yolo=defeitos_yolo, yolo_novo=(yolo_id != ultimo_yolo_id),
                    )
                    ultimo_yolo_id = yolo_id

                    if rastreador is not None:
                        evento = rastreador.atualizar(a)
                        a["hud"].append(rastreador.descricao())
                        with lock:
                            estado["analise"] = a
                        if evento is not None:
                            uuid_gerado = salvar_inspecao(
                                conexao, cfg, evento["indicadores"], evento["defeitos"],
                                evento["status"], evento["confianca_saude"],
                            )
                            print(resumir_analise(uuid_gerado, cfg, evento))
                    else:
                        with lock:
                            estado["analise"] = a
                        # Modo intervalo: grava no máximo uma vez por intervalo
                        agora = time.monotonic()
                        if a.get("eh_tora", True) and agora - ultima_gravacao >= cfg.intervalo_captura_seg:
                            ultima_gravacao = agora
                            uuid_gerado = salvar_inspecao(
                                conexao, cfg, a["indicadores"], a["defeitos"], a["status"], a["confianca_saude"]
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
    print(
        "🚀 Rodando em tempo real. "
        + ("Tecle 'q' na janela para sair, 's' para salvar o frame em capturas/, ou " if mostrar else "")
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

            # Publica o frame mais recente para o worker e pega o último resultado.
            with lock:
                estado["frame"] = frame
                analise = estado["analise"]

            if mostrar:
                cv2.imshow("Omni-Root | John Deere Wood Inspection", desenhar_analise(frame, analise))
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord('q'):
                    break
                if tecla == ord('s'):
                    # Frame CRU (sem desenho): serve para dataset e para testar
                    # o pipeline offline com `--fonte ./capturas`.
                    pasta_capturas.mkdir(exist_ok=True)
                    nome = pasta_capturas / f"frame_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(nome), frame)
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
        captura.release()
        cv2.destroyAllWindows()
        print("✅ Recursos liberados. Até a próxima inspeção!")


if __name__ == "__main__":
    main()
