"""
fontes.py — Fontes de imagem: câmera, arquivo de vídeo ou pasta de imagens (planos B da demonstração).

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from omniroot.config import Config


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
