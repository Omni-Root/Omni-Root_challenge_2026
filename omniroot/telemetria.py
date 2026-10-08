"""
telemetria.py — Frota: posição da máquina em tempo real (dashboard) e trajeto (SQLite -> Postgres).

Duas saídas, a partir do MESMO LeitorPosicao que carimba as toras (hoje a
Localização do Windows; com o módulo GNSS, só muda o gnss_porta para "COM5"):

  1. TRAJETO (offline-first): um ponto em rastro_local a cada poucos segundos
     enquanto a máquina anda (e um por minuto parada). Gravado SEMPRE, com ou
     sem rede; o sync_daemon.py manda os pendentes para rastro_maquinas no
     Postgres. O trecho feito sem internet aparece no mapa quando a rede volta.
  2. TEMPO REAL: a cada `telemetria_intervalo_s`, um POST JSON pequeno para o
     dashboard com o SN (maquina_id) e a posição atual — o mesmo caminho da
     câmera ao vivo: o servidor guarda só a última posição em memória e a
     repassa aos navegadores na hora. Sem rede, recua e tenta de novo; nunca
     atrasa a inspeção.

Sem posição (sem fix / sem receptor), o POST vai assim mesmo, dizendo o
estado: o painel mostra "online, sem posição" em vez de sumir com a máquina.
"""

import json
import math
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

from omniroot.banco_local import migrar_banco_local
from omniroot.config import Config
from omniroot.stream import _token_stream

RAIO_TERRA_M = 6_371_000.0
CAMINHO_POSICAO = "/api/maquinas/posicao"


# ============================================================
# FUNÇÕES PURAS (testadas sem rede nem receptor)
# ============================================================

def distancia_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância em metros entre dois pontos (haversine; sobra para trajeto de máquina)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * RAIO_TERRA_M * math.asin(min(1.0, math.sqrt(a)))


def deve_gravar_ponto(ultimo: dict | None, posicao: dict, agora_s: float, cfg: Config) -> bool:
    """
    Vale gravar um ponto novo do trajeto? `ultimo` = {"lat", "lon", "t"} do
    último ponto gravado (t em segundos monotônicos). Andando: no máximo um a
    cada `rastro_intervalo_s`, e só se andou `rastro_distancia_min_m` — ou a
    precisão informada da leitura, se for maior (Wi-Fi do Windows: centenas
    de metros). O GNSS parado "passeia" metros; isso não é trajeto. Parada:
    um a cada `rastro_parado_s`, para o mapa saber que ela continuou ali.
    """
    if ultimo is None:
        return True
    dt = agora_s - ultimo["t"]
    if dt < cfg.rastro_intervalo_s:
        return False
    if dt >= cfg.rastro_parado_s:
        return True
    limite = max(cfg.rastro_distancia_min_m, posicao.get("precisao_m") or 0.0)
    return distancia_m(ultimo["lat"], ultimo["lon"], posicao["lat"], posicao["lon"]) >= limite


def url_telemetria(cfg: Config) -> str:
    """URL do POST de posição: a configurada, ou a do mesmo servidor da câmera ao vivo."""
    if cfg.telemetria_url and cfg.telemetria_url.strip():
        return cfg.telemetria_url.strip()
    if cfg.stream_url and cfg.stream_url.strip():
        p = urlsplit(cfg.stream_url.strip())
        return urlunsplit((p.scheme, p.netloc, CAMINHO_POSICAO, "", ""))
    return ""


def montar_payload(maquina_id: str, posicao: dict | None, estado: str, fonte: str) -> dict:
    """Corpo do POST. Sem posição vai só o estado (a máquina está online, sem fix)."""
    payload = {"maquina": maquina_id, "fonte": fonte, "estado": "ok" if posicao is not None else estado}
    if posicao is not None:
        for k in ("lat", "lon", "precisao_m", "hdop", "satelites", "idade_s"):
            payload[k] = posicao.get(k)
        payload["fonte"] = posicao.get("fonte") or fonte
    return payload


# ============================================================
# TRAJETO (SQLite local) E ENVIO (HTTP)
# ============================================================

def gravar_ponto_rastro(conexao: sqlite3.Connection, maquina_id: str, posicao: dict,
                        agora: datetime | None = None) -> str:
    """Um ponto do trajeto, pendente de sync. A hora é a da LEITURA (agora - idade da posição)."""
    agora = agora or datetime.now()
    registrado_em = agora - timedelta(seconds=float(posicao.get("idade_s") or 0.0))
    uuid_local = str(uuid.uuid4())
    conexao.execute(
        """
        INSERT INTO rastro_local
            (uuid_local, maquina_id, registrado_em, lat, lon, precisao_m, hdop, satelites, fonte, sync_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            uuid_local, maquina_id, registrado_em.isoformat(timespec="seconds"),
            posicao["lat"], posicao["lon"], posicao.get("precisao_m"), posicao.get("hdop"),
            posicao.get("satelites"), posicao.get("fonte") or "",
        ),
    )
    conexao.commit()
    return uuid_local


def enviar_posicao(url: str, token: str, payload: dict, timeout_s: float = 2.0) -> None:
    """Um POST; levanta exceção em qualquer falha (quem chama decide o recuo)."""
    import urllib.request
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-Stream-Token": token},
    )
    with urllib.request.urlopen(req, timeout=timeout_s):
        pass


def _loop_envio(cfg: Config, leitor, url: str, token: str, parar: threading.Event, tick_s: float) -> None:
    """
    Posição ao vivo, em thread PRÓPRIA: sem rede, um POST pode travar segundos
    (no Windows, conectar numa porta fechada leva ~2 s; e o DNS sem internet
    não respeita o timeout). Aqui isso só atrasa o próximo envio — o trajeto,
    na outra thread, nunca espera pela rede.
    """
    import urllib.error

    proximo_envio = 0.0
    online: bool | None = None  # None = ainda não tentou
    while not parar.is_set():
        agora = time.monotonic()
        if agora >= proximo_envio:
            payload = montar_payload(cfg.maquina_id, leitor.atual(cfg.gnss_validade_s), leitor.estado(), leitor.fonte)
            try:
                enviar_posicao(url, token, payload)
                if online is not True:
                    print("🚜 Frota: posição ao vivo chegando ao dashboard.")
                    online = True
                proximo_envio = agora + cfg.telemetria_intervalo_s
            except urllib.error.HTTPError as e:
                # Servidor respondeu, mas recusou: é configuração, não rede.
                if online is not False:
                    print(f"🚜 Dashboard recusou a posição (HTTP {e.code}) — confira STREAM_TOKEN nos dois .env. Tentando de novo a cada 30 s.")
                    online = False
                proximo_envio = agora + 30.0
            except Exception as e:
                if online is not False:
                    print(f"🚜 Dashboard inacessível ({type(e).__name__}) — o trajeto segue no banco local; tentando de novo a cada 5 s.")
                    online = False
                proximo_envio = agora + 5.0
        parar.wait(tick_s)


def loop_telemetria(cfg: Config, leitor, parar: threading.Event, tick_s: float = 0.5) -> None:
    """
    Corpo da thread de frota: grava o trajeto e, se houver URL e token, sobe
    a thread de envio ao vivo. `leitor` é o LeitorPosicao do main.py (só lê
    `atual()`/`estado()`: nunca toca na porta serial). Conexão SQLite própria
    (cada thread a sua; WAL convive com o worker que grava as toras).
    """
    url = url_telemetria(cfg)
    token = _token_stream() if url else ""
    if url and not token:
        print("⚠️  Frota: STREAM_TOKEN vazio (.env) — posição em tempo real desligada (o trajeto segue gravado).")
        url = ""

    conexao = sqlite3.connect(cfg.sqlite_path)
    conexao.execute("PRAGMA journal_mode=WAL")
    migrar_banco_local(conexao)

    print(
        f"🚜 Frota: trajeto no banco local"
        + (f" + posição ao vivo a cada {cfg.telemetria_intervalo_s:g} s para {url}" if url else "")
        + f" (máquina {cfg.maquina_id})."
    )
    envio = None
    if url:
        envio = threading.Thread(
            target=_loop_envio, args=(cfg, leitor, url, token, parar, tick_s), daemon=True, name="frota-envio"
        )
        envio.start()

    ultimo_ponto: dict | None = None
    avisou_erro_banco = False
    try:
        while not parar.is_set():
            agora = time.monotonic()
            posicao = leitor.atual(cfg.gnss_validade_s)
            if posicao is not None and deve_gravar_ponto(ultimo_ponto, posicao, agora, cfg):
                try:
                    gravar_ponto_rastro(conexao, cfg.maquina_id, posicao)
                    ultimo_ponto = {"lat": posicao["lat"], "lon": posicao["lon"], "t": agora}
                    avisou_erro_banco = False
                except sqlite3.Error as e:
                    if not avisou_erro_banco:
                        print(f"⚠️  Frota: não consegui gravar o trajeto ({e}); tentando de novo.")
                        avisou_erro_banco = True
            parar.wait(tick_s)
    finally:
        conexao.close()
        if envio is not None:
            envio.join(timeout=3.0)
