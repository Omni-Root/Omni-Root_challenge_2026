"""
testar_portao_cor.py — portão "é madeira?": madeira creme neutralizada pela webcam

Problema (capturas de 30/09): com a rodela ocupando o quadro, o auto-balanço
de branco da webcam puxa o creme da madeira para o neutro — miolo em Lab
b* 126-128. O portão exigia b* >= 128 para madeira pálida, então metade dos
pixels falhava por 1-2 unidades e a mesma rodela alternava tora / sem tora.
O limite virou config (tora_palida_b_min = 122); papel branco sai em b* ~95.

Verifica, com cenas sintéticas sobre mesa preta:
  1. rodela creme neutralizada (b* 126) passa — e reprovava com o limite 128;
  2. cartão/papel branco frio (b* 100) continua barrado;
  3. mão (pele) continua barrada;
  4. (se existirem) as capturas reais de 30/09 em capturas/: a maioria passa.

USO (raiz do repositório):
    python tests/testar_portao_cor.py
"""

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from omniroot.config import Config  # noqa: E402
from omniroot.segmentacao import recortar_roi, segmentar_tora_origem, validar_tora  # noqa: E402


def bgr_de_lab(L: int, a: int, b: int) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[[L, a, b]]]), cv2.COLOR_LAB2BGR)[0, 0].astype(np.float32)


def disco(cor_bgr: np.ndarray, seed: int, raio: int = 150) -> np.ndarray:
    """Disco com textura leve (anéis/fibra) sobre mesa preta."""
    img = np.full((480, 640, 3), 16.0, np.float32)
    m = np.zeros((480, 640), np.uint8)
    cv2.circle(m, (320, 240), raio, 255, -1)
    textura = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 6, (480, 640)).astype(np.float32), (0, 0), 2)
    img[m > 0] = cor_bgr + textura[m > 0][:, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def portao(frame: np.ndarray, cfg: Config) -> tuple[bool, str, dict]:
    mask, contorno, origem = segmentar_tora_origem(frame)
    return validar_tora(frame, mask, contorno, origem, cfg)


def main() -> int:
    falhas = 0

    def checar(cond: bool, msg: str) -> None:
        nonlocal falhas
        print(("  ✅ " if cond else "  ❌ ") + msg)
        if not cond:
            falhas += 1

    cfg = Config()
    cfg_antigo = replace(cfg, tora_palida_b_min=128.0)

    print("1) madeira creme neutralizada pela webcam (b* 126)")
    creme = disco(bgr_de_lab(200, 130, 126), seed=1)
    ok, motivo, met = portao(creme, cfg)
    checar(ok, f"passa no portão (madeira {met.get('madeira', 0):.0%}{', ' + motivo if motivo else ''})")
    ok_ant, motivo_ant, _ = portao(creme, cfg_antigo)
    checar(not ok_ant, f"com o limite antigo (128) reprovava: {motivo_ant or 'passou?'} — era o bug")

    print("2) papel/cartão branco frio continua barrado")
    papel = disco(bgr_de_lab(235, 128, 100), seed=2)
    ok, motivo, _ = portao(papel, cfg)
    checar(not ok, f"papel b* 100: sem tora ({motivo or 'passou!'})")

    print("3) mão continua barrada")
    mao = disco(np.array([120, 150, 225], np.float32), seed=3)
    ok, motivo, _ = portao(mao, cfg)
    checar(not ok, f"mão: sem tora ({motivo or 'passou!'})")

    print("4) capturas reais de 30/09 (opcional: capturas/ fica fora do git)")
    fotos = sorted((RAIZ / "capturas").glob("frame_20260930_*.jpg"))
    if not fotos:
        print("     (sem capturas nesta máquina — pulado)")
    else:
        aprovadas = aprovadas_antes = 0
        for f in fotos:
            rec, _ = recortar_roi(cv2.imread(str(f)), cfg)
            aprovadas += portao(rec, cfg)[0]
            aprovadas_antes += portao(rec, cfg_antigo)[0]
        checar(aprovadas >= len(fotos) - 1,
               f"{aprovadas}/{len(fotos)} reconhecidas como tora (limite antigo: {aprovadas_antes}/{len(fotos)})")

    print()
    print("TUDO OK" if falhas == 0 else f"{falhas} FALHA(S)")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
