"""
hud.py — Desenho do resultado sobre o quadro (caixas, contorno, marcador e HUD do operador).

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

import cv2
import numpy as np

# Cores BGR do contorno/caixas por status (o cinza é o que os filtros ignoraram).
CORES_STATUS = {"aprovado": (0, 255, 0), "quarentena": (0, 255, 255), "reprovado": (0, 0, 255), "sem_tora": (200, 200, 200)}
COR_IGNORADO = (140, 140, 140)


def desenhar_analise(frame: np.ndarray, a: dict | None, miniatura_fundo: "np.ndarray | None" = None) -> np.ndarray:
    """
    Desenha o resultado sobre uma cópia do frame. Caixas MANTIDAS na cor do
    status; caixas IGNORADAS pelos filtros em cinza fino, com o motivo
    visível -- na demo isso mostra que o sistema viu o teclado/a mão e
    decidiu, de propósito, não contar.
    """
    vis = frame.copy()
    fonte = cv2.FONT_HERSHEY_SIMPLEX

    # Miniatura do fundo de referência no canto inferior direito: o operador
    # VÊ o que foi capturado -- se a peça aparecer aqui, o fundo está errado.
    if miniatura_fundo is not None:
        th, tw = miniatura_fundo.shape[:2]
        H, W = vis.shape[:2]
        if th + 30 < H and tw + 10 < W:
            y0, x0 = H - th - 8, W - tw - 8
            vis[y0:y0 + th, x0:x0 + tw] = miniatura_fundo
            cv2.rectangle(vis, (x0 - 1, y0 - 1), (x0 + tw, y0 + th), (255, 0, 255), 1)
            cv2.putText(vis, "fundo de ref. (b)", (x0, y0 - 6), fonte, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

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
