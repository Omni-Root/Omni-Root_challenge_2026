"""
calibrar.py — Marcador de escala automática (+ ferramentas opcionais de maquete)

Em operação real ninguém calibra na mão. A escala px->cm é AUTOMÁTICA:
um marcador ArUco de tamanho conhecido fica fixo no campo de visão (na
garra, ou ao lado da peça na maquete) e o main.py o detecta em cada frame
e calcula cm/px sozinho — mesmo que a câmera mude de posição.

    python calibrar.py --gerar-marcador          # gera marcador_aruco_5cm.png
    python calibrar.py --gerar-marcador --cm 8   # outro tamanho

Imprima em 100% (sem "ajustar à página"), confira com régua que o lado
preto mede exatamente os cm informados, e deixe `marcador_aruco_cm` no
config.json igual a esse valor. Colado num pedaço de papelão ao lado da
tora, pronto.

Ferramentas OPCIONAIS para a maquete (não são o fluxo de produção):
    python calibrar.py --roi       # arrasta um retângulo -> grava "roi" no config
    python calibrar.py --regua     # 2 cliques numa régua -> grava "cm_por_px" (reserva, se não houver marcador)

Precisa do opencv-python COM GUI (não o headless) para --roi/--regua.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from main import abrir_fonte, carregar_configuracao_json


def pegar_frame(captura):
    # Descarta alguns frames: a webcam costuma entregar os primeiros escuros.
    frame = None
    for _ in range(10):
        ok, f = captura.read()
        if ok and f is not None:
            frame = f
    if frame is None:
        raise RuntimeError("Não consegui capturar um frame da fonte.")
    return frame


def calibrar_roi(captura) -> list | None:
    print("\n[1/2] ROI — arraste o retângulo em volta de onde a tora fica; ENTER confirma, ESC pula.")
    janela = "Calibrar ROI (arraste, ENTER confirma, ESC pula)"
    frame = pegar_frame(captura)
    x, y, w, h = cv2.selectROI(janela, frame, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(janela)
    if w == 0 or h == 0:
        print("   ROI não definida (pulado).")
        return None
    H, W = frame.shape[:2]
    roi = [round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4)]
    print(f"   ROI = {roi}  (frame {W}x{H}, recorte {w}x{h} px)")
    return roi


def calibrar_regua(captura, comprimento_cm: float) -> float | None:
    print(f"\n[2/2] RÉGUA — clique no '0' e depois no '{comprimento_cm:g} cm' da régua. ENTER confirma, R refaz, ESC pula.")
    janela = "Calibrar regua (2 cliques, ENTER confirma, R refaz, ESC pula)"
    pontos: list[tuple[int, int]] = []

    def on_mouse(evento, mx, my, *_):
        if evento == cv2.EVENT_LBUTTONDOWN and len(pontos) < 2:
            pontos.append((mx, my))

    cv2.namedWindow(janela)
    cv2.setMouseCallback(janela, on_mouse)
    resultado = None
    while True:
        ok, frame = captura.read()
        if not ok or frame is None:
            continue
        vis = frame.copy()
        for p in pontos:
            cv2.circle(vis, p, 5, (0, 255, 255), -1)
        if len(pontos) == 2:
            cv2.line(vis, pontos[0], pontos[1], (0, 255, 255), 2)
            dist_px = ((pontos[0][0] - pontos[1][0]) ** 2 + (pontos[0][1] - pontos[1][1]) ** 2) ** 0.5
            resultado = comprimento_cm / max(dist_px, 1e-6)
            cv2.putText(vis, f"{dist_px:.0f}px = {comprimento_cm:g}cm -> {resultado:.5f} cm/px",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(vis, f"Clique no 0 e no {comprimento_cm:g} cm da regua ({len(pontos)}/2)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(janela, vis)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32) and resultado is not None:   # ENTER / ESPAÇO
            break
        if k in (ord("r"), ord("R")):
            pontos.clear()
            resultado = None
        if k == 27:                                    # ESC
            resultado = None
            break
    cv2.destroyWindow(janela)
    if resultado is None:
        print("   Régua não calibrada (pulado).")
        return None
    print(f"   cm_por_px = {resultado:.5f}")
    return round(resultado, 5)


def gerar_marcador(lado_cm: float, saida: Path, dpi: int = 300, marcador_id: int = 7) -> None:
    """Gera o PNG do marcador ArUco (DICT_4X4_50) com borda branca, no tamanho físico certo para imprimir."""
    px_por_cm = dpi / 2.54
    lado_px = int(round(lado_cm * px_por_cm))
    dicionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marcador = cv2.aruco.generateImageMarker(dicionario, marcador_id, lado_px)
    borda = int(round(1.0 * px_por_cm))  # 1 cm de "quiet zone" branca, obrigatória para detectar bem
    canvas = np.full((lado_px + 2 * borda, lado_px + 2 * borda), 255, dtype=np.uint8)
    canvas[borda:borda + lado_px, borda:borda + lado_px] = marcador
    cv2.putText(canvas, f"ArUco 4x4 id={marcador_id}  lado={lado_cm:g} cm  imprimir a 100%",
                (borda, canvas.shape[0] - borda // 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
    cv2.imwrite(str(saida), canvas)
    print(f"✅ Marcador gerado: {saida} ({lado_cm:g} cm de lado a {dpi} dpi). Imprima em 100% e confira com régua.")
    print(f"   No config.json: \"marcador_aruco_cm\": {lado_cm:g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o marcador de escala automática; opcionalmente ROI/régua para maquete")
    parser.add_argument("--gerar-marcador", action="store_true", help="Gera o PNG do marcador ArUco para imprimir")
    parser.add_argument("--cm", type=float, default=5.0, help="Lado do marcador em cm (padrão 5)")
    parser.add_argument("--saida", default=None, help="Arquivo PNG de saída (padrão: marcador_aruco_<cm>cm.png)")
    parser.add_argument("--roi", dest="so_roi", action="store_true", help="(maquete) define a ROI arrastando um retângulo")
    parser.add_argument("--regua", dest="so_regua", action="store_true", help="(maquete) define cm_por_px com 2 cliques numa régua")
    parser.add_argument("--config-file", default="./config.json")
    parser.add_argument("--fonte", default=None, help="Índice de câmera, vídeo ou imagem (padrão: camera_index do config)")
    parser.add_argument("--regua-cm", type=float, default=10.0, help="Comprimento marcado na régua entre os 2 cliques (padrão 10)")
    args = parser.parse_args()

    if args.gerar_marcador or not (args.so_roi or args.so_regua):
        saida = Path(args.saida) if args.saida else Path(f"marcador_aruco_{args.cm:g}cm.png")
        gerar_marcador(args.cm, saida)
        if not (args.so_roi or args.so_regua):
            return

    cfg = carregar_configuracao_json(args.config_file)
    captura = abrir_fonte(args.fonte, cfg)
    if not captura.isOpened():
        raise SystemExit("Não foi possível abrir a fonte de vídeo.")

    novos: dict = {}
    try:
        if not args.so_regua:
            roi = calibrar_roi(captura)
            if roi is not None:
                novos["roi"] = roi
        if not args.so_roi:
            cm = calibrar_regua(captura, args.regua_cm)
            if cm is not None:
                novos["cm_por_px"] = cm
    finally:
        captura.release()
        cv2.destroyAllWindows()

    if not novos:
        print("\nNada alterado.")
        return

    p = Path(args.config_file)
    dados = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    dados.update(novos)
    p.write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n✅ Gravado em {p}: {novos}")
    print("   Rode `python main.py --gui` e confira o retângulo 'ROI' na tela.")


if __name__ == "__main__":
    main()
