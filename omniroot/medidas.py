"""
medidas.py — Medidas físicas da tora: escala px->cm (ArUco/calibração/GSD), diâmetro, volume, massa, casca, tortuosidade, rachadura.

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

import time

import cv2
import numpy as np

from omniroot.config import Config
from omniroot.segmentacao import _mascara_do_contorno, mascara_interior


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
