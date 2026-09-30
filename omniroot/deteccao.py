"""
deteccao.py — Modelo de defeitos (YOLO): carga, inferência, pré-processamento (cinza + CLAHE) e filtros de falso positivo.

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from omniroot.config import Config
from omniroot.segmentacao import mascara_interior


CONF_POR_CLASSE_PADRAO = {"resin": 0.80, "Live_Knot": 0.70, "Marrow": 0.65, "Quartzity": 0.70}


def limiar_da_classe(cfg: Config, tipo_defeito: str) -> float:
    """Limiar de confiança para uma classe: o específico se houver, senão o geral."""
    tabela = cfg.conf_por_classe if isinstance(cfg.conf_por_classe, dict) else CONF_POR_CLASSE_PADRAO
    return float(tabela.get(tipo_defeito, cfg.conf_threshold))


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


def deslocar_defeitos(defeitos: list, dx: int, dy: int) -> list:
    """Leva caixas medidas no recorte da ROI de volta às coordenadas do frame inteiro."""
    if not dx and not dy:
        return defeitos
    return [
        {**d, "pos_x": round(d["pos_x"] + dx, 2), "pos_y": round(d["pos_y"] + dy, 2)}
        for d in defeitos
    ]


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
        # O predict roda com o limiar GERAL; aqui aplica o da classe, que
        # pode ser mais exigente (ver conf_por_classe).
        if confianca < limiar_da_classe(cfg, tipo_defeito):
            continue
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
