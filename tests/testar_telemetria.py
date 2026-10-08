"""
testar_telemetria.py — frota: posição ao vivo + trajeto (sem receptor, sem dashboard)

Verifica:
  1. distância entre pontos (haversine) em casos conhecidos
  2. quando gravar um ponto do trajeto: intervalo mínimo, distância mínima
     (GNSS parado "passeando" não vira trajeto) e o ponto da máquina parada
  3. URL do POST: a configurada, ou deduzida da câmera ao vivo; vazia desliga
  4. corpo do POST com e sem posição (sem fix vai o estado, sem coordenada)
  5. SQLite: banco ANTIGO (sem rastro_local) ganha a tabela sem perder
     toras; o ponto gravado guarda os campos e a hora da LEITURA
  6. thread completa contra um servidor HTTP falso: posição chega com o SN e
     o token; trajeto gravado no SQLite; SEM rede o trajeto continua sendo
     gravado; sem fix o POST diz o estado

USO (raiz do repositório):
    python tests/testar_telemetria.py
"""

import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

os.environ["STREAM_TOKEN"] = "token-de-teste"  # antes de qualquer load_dotenv (que não sobrescreve)

from omniroot.banco_local import migrar_banco_local  # noqa: E402
from omniroot.config import Config  # noqa: E402
from omniroot.telemetria import (  # noqa: E402
    deve_gravar_ponto,
    distancia_m,
    gravar_ponto_rastro,
    loop_telemetria,
    montar_payload,
    url_telemetria,
)

SN = "SN-TESTE-01"
LAT0, LON0 = -23.5641, -46.6524  # FIAP Paulista, aproximadamente
GRAU_LAT_M = 111_195.0           # 1 grau de latitude em metros (raio médio 6371 km)

TORAS_LOCAL_ANTIGA = """
CREATE TABLE toras_local (
    id INTEGER PRIMARY KEY AUTOINCREMENT, uuid_local TEXT UNIQUE NOT NULL, maquina_id TEXT NOT NULL,
    talhao_id TEXT, log_id TEXT NOT NULL, data_inspecao TEXT NOT NULL, confianca_ia REAL NOT NULL,
    status_classificacao TEXT NOT NULL, hash_sha256 TEXT NOT NULL,
    sync_status INTEGER NOT NULL DEFAULT 0, criado_em TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def pos(lat: float, lon: float, **extra) -> dict:
    base = {"lat": lat, "lon": lon, "hdop": None, "satelites": None, "precisao_m": 35.0,
            "fonte": "windows_localizacao", "idade_s": 0.0}
    base.update(extra)
    return base


def metros_ao_norte(m: float) -> float:
    return LAT0 + m / GRAU_LAT_M


class LeitorFalso:
    """Imita o LeitorPosicao: anda 10 m para o norte a cada leitura (ou fica sem fix)."""

    fonte = "windows_localizacao"

    def __init__(self, sem_fix: bool = False):
        self.sem_fix = sem_fix
        self.passos = 0

    def atual(self, validade_s: float):
        if self.sem_fix:
            return None
        self.passos += 1
        return pos(metros_ao_norte(10.0 * self.passos), LON0)

    def estado(self) -> str:
        return "sem_fix" if self.sem_fix else "ok"


class Receptor(BaseHTTPRequestHandler):
    """O 'dashboard' falso: guarda o que chegou."""

    recebidos: list = []

    def do_POST(self):
        corpo = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        Receptor.recebidos.append(
            {"token": self.headers.get("X-Stream-Token"), "caminho": self.path, "corpo": json.loads(corpo)}
        )
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


def cfg_teste(sqlite_path: str, url: str) -> Config:
    cfg = Config()
    cfg.maquina_id = SN
    cfg.sqlite_path = sqlite_path
    cfg.stream_url = ""
    cfg.telemetria_url = url
    cfg.telemetria_intervalo_s = 0.2
    cfg.rastro_intervalo_s = 0.1
    cfg.rastro_distancia_min_m = 3.0
    cfg.rastro_parado_s = 60.0
    return cfg


def rodar_thread(cfg: Config, leitor, segundos: float) -> None:
    parar = threading.Event()
    t = threading.Thread(target=loop_telemetria, args=(cfg, leitor, parar, 0.05), daemon=True)
    t.start()
    time.sleep(segundos)
    parar.set()
    t.join(timeout=5)


def main() -> int:
    falhas = 0

    def checar(cond: bool, msg: str) -> None:
        nonlocal falhas
        print(("  ✅ " if cond else "  ❌ ") + msg)
        if not cond:
            falhas += 1

    print("1) distância (haversine)")
    checar(distancia_m(LAT0, LON0, LAT0, LON0) == 0.0, "mesmo ponto = 0 m")
    d = distancia_m(LAT0, LON0, LAT0 + 1.0, LON0)
    checar(abs(d - GRAU_LAT_M) < 50, f"1 grau de latitude ~ 111,2 km (deu {d / 1000:.2f} km)")
    d = distancia_m(LAT0, LON0, metros_ao_norte(10.0), LON0)
    checar(abs(d - 10.0) < 0.01, f"10 m ao norte = 10 m (deu {d:.3f} m)")

    print("2) quando gravar um ponto do trajeto")
    c = Config()  # padrões: 5 s, 20 m, 60 s
    p0 = pos(LAT0, LON0, precisao_m=None)
    ultimo = {"lat": LAT0, "lon": LON0, "t": 100.0}
    gnss = lambda m: pos(metros_ao_norte(m), LON0, precisao_m=None, hdop=1.4, fonte="gnss_serial")  # noqa: E731
    checar(deve_gravar_ponto(None, p0, 100.0, c), "primeiro ponto: grava")
    checar(not deve_gravar_ponto(ultimo, gnss(50), 102.0, c), "andou 50 m em 2 s (< 5 s): espera")
    checar(not deve_gravar_ponto(ultimo, gnss(12), 110.0, c), "10 s depois, 12 m (GNSS parado passeando): não grava")
    checar(deve_gravar_ponto(ultimo, gnss(25), 110.0, c), "10 s depois, 25 m: grava")
    checar(not deve_gravar_ponto(ultimo, pos(metros_ao_norte(100), LON0, precisao_m=279.0), 110.0, c),
           "Windows ±279 m: 'andar' 100 m é a estimativa oscilando, não grava")
    checar(deve_gravar_ponto(ultimo, pos(LAT0, LON0), 160.0, c), "parada há 60 s: grava 'estive aqui'")

    print("3) URL do POST de posição")
    c = Config()
    checar(url_telemetria(c) == "", "sem telemetria_url e sem stream_url: desligado")
    c.stream_url = "http://192.168.1.50:3001/api/camera/frame"
    checar(url_telemetria(c) == "http://192.168.1.50:3001/api/maquinas/posicao", "deduzida do servidor da câmera ao vivo")
    c.telemetria_url = "https://central.exemplo/api/maquinas/posicao"
    checar(url_telemetria(c) == "https://central.exemplo/api/maquinas/posicao", "a configurada tem prioridade")

    print("4) corpo do POST")
    corpo = montar_payload(SN, pos(LAT0, LON0, hdop=0.9, satelites=9, fonte="gnss_serial"), "ok", "gnss_serial")
    checar(corpo["maquina"] == SN and corpo["estado"] == "ok", "leva o SN e estado ok")
    checar(corpo["lat"] == LAT0 and corpo["hdop"] == 0.9 and corpo["satelites"] == 9 and corpo["fonte"] == "gnss_serial",
           "leva lat/lon, HDOP, satélites e a fonte")
    corpo = montar_payload(SN, None, "sem_fix", "gnss_serial")
    checar(corpo == {"maquina": SN, "fonte": "gnss_serial", "estado": "sem_fix"}, "sem posição: só SN, fonte e estado")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        print("5) SQLite: migração e ponto gravado")
        caminho = str(Path(tmp) / "antigo.db")
        con = sqlite3.connect(caminho)
        con.executescript(TORAS_LOCAL_ANTIGA)
        con.execute(
            "INSERT INTO toras_local (uuid_local, maquina_id, log_id, data_inspecao, confianca_ia, status_classificacao, hash_sha256) "
            "VALUES ('u1', ?, 'LOG-1', '2026-10-07T10:00:00', 0.9, 'aprovado', 'h')",
            (SN,),
        )
        con.commit()
        migrar_banco_local(con)
        tabelas = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        checar("rastro_local" in tabelas, "banco antigo ganhou rastro_local")
        checar(con.execute("SELECT COUNT(*) FROM toras_local").fetchone()[0] == 1, "a tora que já existia continua lá")
        migrar_banco_local(con)
        checar(True, "migrar de novo não quebra (idempotente)")
        agora = datetime(2026, 10, 7, 14, 30, 10)
        u = gravar_ponto_rastro(con, SN, pos(LAT0, LON0, idade_s=4.0), agora=agora)
        linha = con.execute(
            "SELECT maquina_id, registrado_em, lat, lon, precisao_m, fonte, sync_status FROM rastro_local WHERE uuid_local = ?", (u,)
        ).fetchone()
        checar(linha == (SN, "2026-10-07T14:30:06", LAT0, LON0, 35.0, "windows_localizacao", 0),
               "ponto guarda SN, lat/lon, precisão, fonte, pendente de sync e a hora da LEITURA (agora - 4 s)")
        con.close()

        print("6) thread completa (servidor falso no lugar do dashboard)")
        servidor = HTTPServer(("127.0.0.1", 0), Receptor)
        threading.Thread(target=servidor.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{servidor.server_address[1]}/api/maquinas/posicao"

        Receptor.recebidos = []
        caminho = str(Path(tmp) / "online.db")
        rodar_thread(cfg_teste(caminho, url), LeitorFalso(), 1.2)
        rec = Receptor.recebidos
        checar(len(rec) >= 3, f"posições chegaram ao dashboard ({len(rec)} POSTs em 1,2 s, um a cada 0,2 s)")
        checar(all(r["token"] == "token-de-teste" for r in rec), "todo POST leva o STREAM_TOKEN")
        checar(all(r["caminho"] == "/api/maquinas/posicao" for r in rec), "na rota de posição")
        checar(all(r["corpo"]["maquina"] == SN and r["corpo"]["lat"] is not None for r in rec), "com o SN e a coordenada")
        lats = [r["corpo"]["lat"] for r in rec]
        checar(lats == sorted(lats) and lats[-1] > lats[0], "a posição avança (a máquina está andando)")
        con = sqlite3.connect(caminho)
        n = con.execute("SELECT COUNT(*) FROM rastro_local WHERE maquina_id = ? AND sync_status = 0", (SN,)).fetchone()[0]
        con.close()
        checar(n >= 3, f"trajeto gravado no SQLite ({n} pontos pendentes de sync)")

        caminho = str(Path(tmp) / "offline.db")
        sem_rede = "http://127.0.0.1:9/api/maquinas/posicao"  # porta 9 (discard): ninguém escuta
        rodar_thread(cfg_teste(caminho, sem_rede), LeitorFalso(), 1.0)
        con = sqlite3.connect(caminho)
        n = con.execute("SELECT COUNT(*) FROM rastro_local").fetchone()[0]
        con.close()
        checar(n >= 3, f"SEM rede: o trajeto continua sendo gravado ({n} pontos)")

        Receptor.recebidos = []
        caminho = str(Path(tmp) / "semfix.db")
        rodar_thread(cfg_teste(caminho, url), LeitorFalso(sem_fix=True), 0.6)
        rec = Receptor.recebidos
        checar(len(rec) >= 1 and all(r["corpo"] == {"maquina": SN, "fonte": "windows_localizacao", "estado": "sem_fix"} for r in rec),
               "sem fix: o POST vai assim mesmo, dizendo 'sem_fix' (máquina online, sem posição)")
        con = sqlite3.connect(caminho)
        n = con.execute("SELECT COUNT(*) FROM rastro_local").fetchone()[0]
        con.close()
        checar(n == 0, "sem fix: nenhum ponto inventado no trajeto")
        servidor.shutdown()
        servidor.server_close()

    print()
    print("TUDO OK" if falhas == 0 else f"{falhas} FALHA(S)")
    return 0 if falhas == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
