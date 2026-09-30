"""
segmentacao.py — Segmentação da tora e portão "isso é madeira?": máscaras, fundo escuro/claro, fundo de referência, vista e ROI.

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

import cv2
import numpy as np

from omniroot.config import Config


def _mascara_do_contorno(shape: tuple, contorno) -> np.ndarray:
    """Máscara binária (255 dentro) a partir de um contorno, no tamanho do frame."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.asarray(contorno, dtype=np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [pts], -1, 255, -1)
    return mask


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
# BALANÇO DE BRANCO — estimado pela borda do frame
# ------------------------------------------------------------
# Gray-world clássico assume que a MÉDIA da cena é cinza; com uma tora
# marrom ocupando o centro isso desbotaria justamente a madeira. Aqui o
# cinza é estimado só na moldura externa do frame (fundo: mesa, garra,
# parede), e o ganho por canal fica preso em [0.8, 1.25]. Resultado: mesa
# azulada ou amarelada vira neutra, a madeira mantém a cor, e a segmentação
# por saturação + o portão de cor ficam estáveis entre ambientes.
# ------------------------------------------------------------

def balancear_branco(frame: np.ndarray, borda: float = 0.10) -> np.ndarray:
    if frame is None or not borda or borda <= 0:
        return frame
    try:
        h, w = frame.shape[:2]
        by, bx = max(2, int(h * borda)), max(2, int(w * borda))
        moldura = np.zeros((h, w), dtype=bool)
        moldura[:by, :] = True
        moldura[-by:, :] = True
        moldura[:, :bx] = True
        moldura[:, -bx:] = True
        pixels = frame[moldura].astype(np.float32)
        # Só pixels de superfície NEUTRA entram na estimativa: nem muito
        # escuros/claros (sombra, estouro) nem saturados (madeira, mão,
        # objeto colorido que invade a borda). O tom que sobra num cinza é
        # a cor da luz — é isso que se corrige.
        brilho = pixels.mean(axis=1)
        sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1][moldura].astype(np.float32)
        neutro = (brilho > 40) & (brilho < 235) & (sat < 60)
        # Se a borda NÃO é majoritariamente neutra (tinte forte demais, ou
        # tora/mão ocupando a moldura), a estimativa não é confiável: não
        # corrige. Corrigir errado é pior que não corrigir.
        if neutro.mean() < 0.5:
            return frame
        validos = pixels[neutro]
        if validos.shape[0] < 200:
            return frame
        medias = validos.mean(axis=0)  # B, G, R
        alvo = float(medias.mean())
        ganhos = np.clip(alvo / np.maximum(medias, 1.0), 0.8, 1.25)
        if np.allclose(ganhos, 1.0, atol=0.02):
            return frame
        corrigido = frame.astype(np.float32) * ganhos.reshape(1, 1, 3)
        return np.clip(corrigido, 0, 255).astype(np.uint8)
    except Exception:
        return frame


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


def _componente_central(mask: np.ndarray, area_min_frac: float = 0.03) -> np.ndarray | None:
    """Máscara (255) do componente que contém o centro do frame, ou do mais próximo dele; None se nada grande."""
    h, w = mask.shape[:2]
    n, rotulos, stats, centroides = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    cx0, cy0 = w / 2.0, h / 2.0
    area_min = area_min_frac * h * w
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
        if melhor is None:
            return None
        escolhido = melhor[1]
    return (rotulos == escolhido).astype(np.uint8) * 255


def fundo_e_escuro(frame: np.ndarray, borda: float = 0.08) -> tuple[bool, float]:
    """(True, V_mediano_da_borda) se a moldura do frame é escura (mesa preta, garra escura)."""
    h, w = frame.shape[:2]
    by, bx = max(2, int(h * borda)), max(2, int(w * borda))
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
    moldura = np.concatenate([v[:by].ravel(), v[-by:].ravel(), v[:, :bx].ravel(), v[:, -bx:].ravel()])
    med = float(np.median(moldura))
    return med < 60.0, med


def segmentar_fundo_escuro(frame: np.ndarray, ignorar: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    MESA PRETA (o cenário da banca): madeira x preto é uma separação de
    BRILHO — não depende da cor da madeira, do balanço de branco nem de
    casca/sem casca. Limiar = brilho da borda (fundo) + margem, com piso.
    Pixels muito escuros têm saturação-ruído, por isso este caminho vem
    ANTES do de cor quando a borda é escura.
    """
    try:
        h, w = frame.shape[:2]
        v = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2], (7, 7), 0)
        by, bx = max(2, int(h * 0.08)), max(2, int(w * 0.08))
        moldura = np.concatenate([v[:by].ravel(), v[-by:].ravel(), v[:, :bx].ravel(), v[:, -bx:].ravel()])
        # Mediana, não percentil alto: um tronco lateral atravessa o quadro e
        # ocupa parte da borda; a mediana continua sendo o preto da mesa
        # enquanto a tora cobrir menos da metade da moldura.
        limiar = max(55.0, float(np.median(moldura)) + 35.0)
        mask = (v > limiar).astype(np.uint8) * 255
        if ignorar is not None:
            mask[ignorar > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        if np.count_nonzero(mask) > 0.9 * h * w:
            return None, None
        comp = _componente_central(mask)
        if comp is None:
            return None, None
        contornos, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None, None
        contorno = max(contornos, key=cv2.contourArea)
        # Casca muito escura pode ficar abaixo do limiar e "morder" a borda da
        # tora: tora é convexa, então fecha pelo casco quando é convexo mordido.
        casco = cv2.convexHull(contorno)
        a_c, a_h = cv2.contourArea(contorno), cv2.contourArea(casco)
        if a_h > 0 and a_c / a_h >= 0.6:
            contorno = casco
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(out, [contorno], -1, 255, -1)
        return out, contorno
    except Exception:
        return None, None


def segmentar_por_calor(frame: np.ndarray, ignorar: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Madeira é QUENTE (amarelo/laranja/marrom: Lab b* > 0); papel, parede,
    lençol branco e mesa cinza saem NEUTROS ou FRIOS numa webcam (o
    auto-balanço puxa o branco para o azul). Otsu no canal b* separa as
    duas classes; a tora é a classe quente. Só vale se as classes estão
    de fato separadas (diferença de médias >= 8) — senão devolve None e
    a segmentação por saturação assume. Medido nas capturas: papel b=106-112,
    miolo da madeira b=132-137, casca b=138-142.
    """
    try:
        h, w = frame.shape[:2]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        b = cv2.GaussianBlur(lab[..., 2], (9, 9), 0)
        val = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
        t, _ = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        quente = b > t
        if quente.mean() < 0.02 or quente.mean() > 0.98:
            return None, None
        sep = float(b[quente].mean()) - float(b[~quente].mean())
        if sep < 8.0 or t < 118:
            return None, None  # sem contraste quente x frio (ou tudo é quente: mesa de madeira)
        mask = (quente & (val >= 20)).astype(np.uint8) * 255
        if ignorar is not None:
            mask[ignorar > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        if np.count_nonzero(mask) > 0.9 * h * w:
            return None, None
        comp = _componente_central(mask)
        if comp is None:
            return None, None
        contornos, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None, None
        contorno = max(contornos, key=cv2.contourArea)
        casco = cv2.convexHull(contorno)
        a_c, a_h = cv2.contourArea(contorno), cv2.contourArea(casco)
        if a_h > 0 and a_c / a_h >= 0.6:
            contorno = casco  # tora é convexa; fecha "mordidas" de miolo claro
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(out, [contorno], -1, 255, -1)
        return out, contorno
    except Exception:
        return None, None


def segmentar_tora_origem(frame: np.ndarray, ignorar: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Devolve (mascara, contorno, origem). `origem` diz QUAL caminho achou a
    tora: "cor" (saturação + matiz de madeira — confiável) ou "brilho"
    (Otsu de brilho, reserva para fundo colorido — acha blob em qualquer
    coisa, inclusive num teclado). O portão validar_tora usa isso.
    """
    if frame is None or frame.size == 0:
        return None, None, "nenhuma"
    # Mesa/garra escura: brilho separa melhor que cor (e a saturação de
    # pixels pretos é ruído). É o cenário da banca (mesa preta).
    escuro, _ = fundo_e_escuro(frame)
    if escuro:
        m, c = segmentar_fundo_escuro(frame, ignorar)
        return (m, c, "fundo_escuro") if m is not None else (None, None, "nenhuma")
    # Fundo claro/neutro: quente x frio separa melhor que saturação (papel e
    # parede brancos saem azulados na webcam; madeira sai quente).
    m, c = segmentar_por_calor(frame, ignorar)
    if m is not None:
        return m, c, "cor"
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

        if ignorar is not None:
            mask[ignorar > 0] = 0

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
# SEGMENTAÇÃO POR FUNDO DE REFERÊNCIA (câmera fixa)
# ------------------------------------------------------------
# Diferença, em Lab, entre o frame atual e o frame de referência da garra
# vazia. Antes de comparar, o brilho (L) do frame é alinhado ao da
# referência pela BORDA (fundo): a auto-exposição da webcam muda quando uma
# peça clara entra no quadro, e sem isso o frame inteiro "difere". Pixels
# que diferem mais que `limiar` formam a máscara; o componente que contém o
# centro (ou o mais próximo) é a tora, e o contorno externo preenchido
# resolve o miolo (que pode ter cor parecida com o fundo — só a borda e os
# anéis diferem, mas o preenchimento fecha o disco).
# ------------------------------------------------------------

def segmentar_por_fundo(frame: np.ndarray, fundo: np.ndarray, limiar: float = 22.0, ignorar: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Devolve (mascara, contorno, motivo). motivo: "" ok, "sem diferenca do fundo", "fundo mudou (tecle b)"."""
    if frame is None or fundo is None or frame.shape != fundo.shape:
        return None, None, "fundo invalido"
    try:
        h, w = frame.shape[:2]
        lab_f = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_b = cv2.cvtColor(cv2.GaussianBlur(fundo, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

        by, bx = max(2, int(h * 0.08)), max(2, int(w * 0.08))
        moldura = np.zeros((h, w), dtype=bool)
        moldura[:by, :] = True
        moldura[-by:, :] = True
        moldura[:, :bx] = True
        moldura[:, -bx:] = True
        # Alinha o brilho global pela borda (fundo): compensa auto-exposição.
        d_l = float(np.median(lab_f[..., 0][moldura]) - np.median(lab_b[..., 0][moldura]))
        if abs(d_l) < 60:
            lab_f[..., 0] -= d_l

        dist = np.linalg.norm(lab_f - lab_b, axis=2)
        mask = (dist > limiar).astype(np.uint8) * 255
        if ignorar is not None:
            mask[ignorar > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        # Fundo inteiro "diferente" = luz mudou, câmera mexeu ou fundo trocou:
        # não é tora, e o operador precisa recapturar o fundo.
        if np.count_nonzero(mask) > 0.85 * h * w:
            return None, None, "fundo mudou (tecle b)"

        n, rotulos, stats, centroides = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None, None, "sem diferenca do fundo"
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
            return None, None, "sem diferenca do fundo"

        comp = (rotulos == escolhido).astype(np.uint8) * 255
        contornos, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None, None, "sem diferenca do fundo"
        contorno = max(contornos, key=cv2.contourArea)
        # Miolo claro pode ter a MESMA cor do fundo (madeira pálida x parede
        # branca): a diferença acha a casca e os anéis, mas deixa "mordidas"
        # no contorno. Tora é convexa (disco ou tronco), então, se o
        # componente é um convexo mordido (>= 60% do casco), a máscara vira
        # o casco convexo. Formas muito abertas (mão, cabos) ficam como
        # estão e caem no portão.
        casco = cv2.convexHull(contorno)
        area_c, area_h = cv2.contourArea(contorno), cv2.contourArea(casco)
        if area_h > 0 and area_c / area_h >= 0.6:
            contorno = casco
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(out, [contorno], -1, 255, -1)  # preenche o miolo
        return out, contorno, ""
    except Exception:
        return None, None, "erro"


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
        if cfg.tora_exigir_cor and origem not in ("cor", "fundo", "fundo_escuro"):
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

        # Pele x madeira rosada: o que separa é o VERMELHO (Lab a*): pele
        # a* >= 140 (OpenCV, +12 e acima); madeira clara/rosada fica em
        # 131-136 e casca em ~136-140. Matiz sozinho não separa (a madeira
        # desta câmera sai com matiz 9-13, igual à pele).
        lab_a = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 1][dentro].astype(np.int32)
        # Saturação limitada: pele fica em S 30-130; casca avermelhada muito
        # saturada (S > 130) não é pele. Ajuste fino: tora_fracao_pele_max.
        pele = (lab_a >= 140) & (sat >= 30) & (sat <= 130) & (val >= 80)
        fracao_pele = float(np.mean(pele))
        met["pele"] = round(fracao_pele, 3)
        if fracao_pele > cfg.tora_fracao_pele_max:
            return False, f"pele {fracao_pele:.0%}", met

        # Madeira = matiz quente com alguma saturação (casca, madeira amarelada)
        # OU pálida e clara (madeira exposta creme: nesses pixels a saturação
        # é baixa e o matiz vira ruído -- não dá para exigir "quente").
        lab_b = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 2][dentro].astype(np.int32)
        quente = ((hue <= 35) | (hue >= 165)) & (sat >= 20) & (val >= 20)
        # Pálida só conta como madeira se NÃO for fria: papel/parede brancos
        # saem azulados na webcam (b* ~95-112) e não podem passar por madeira.
        # O limite não é o neutro exato (128): o auto-balanço da câmera deixa
        # o creme em ~126 (ver tora_palida_b_min).
        palida = (sat < 45) & (val >= 110) & (lab_b >= cfg.tora_palida_b_min)
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
