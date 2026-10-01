"""
stream.py — Câmera ao vivo no dashboard: codifica e empurra o quadro anotado (JPEG) para o servidor central.

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

import threading
import time

import cv2
import numpy as np

from omniroot.config import Config
from omniroot.hud import desenhar_analise


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
