"""
banco_local.py — definições do SQLite da máquina compartilhadas entre o
main.py (quem grava) e o sync_daemon.py (quem lê e envia).

Hoje: as colunas de posição (GNSS) de toras_local, a tabela do trajeto da
máquina (rastro_local) e a migração que acrescenta as duas num banco criado
antes delas. Os dois processos chamam a mesma migração ao abrir o banco —
tanto faz qual sobe primeiro.
"""

import sqlite3

# Colunas de posição em toras_local (ver omniroot/posicao.py). Vazias = tora
# sem posição — é permitido e normal. A ordem é a usada nos SELECTs.
COLUNAS_POSICAO = {
    "pos_lat": "REAL",
    "pos_lon": "REAL",
    "pos_hdop": "REAL",
    "pos_satelites": "INTEGER",
    "pos_fonte": "TEXT",
    "pos_idade_s": "REAL",
    "pos_precisao_m": "REAL",  # raio de incerteza em metros, quando a fonte informa (Localização do Windows)
}


# Trajeto da máquina (omniroot/telemetria.py): um ponto a cada poucos segundos
# enquanto ela anda, gravado SEMPRE no SQLite — com ou sem internet. O
# sync_daemon.py manda os pendentes para rastro_maquinas no Postgres, então o
# trecho percorrido sem rede aparece no mapa quando a rede volta.
DDL_RASTRO_LOCAL = (
    """
    CREATE TABLE IF NOT EXISTS rastro_local (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        uuid_local      TEXT UNIQUE NOT NULL,       -- gerado em Python (uuid4): idempotência do sync
        maquina_id      TEXT NOT NULL,              -- numero_serie da máquina (SN)
        registrado_em   TEXT NOT NULL,              -- hora local da máquina, ISO 8601
        lat             REAL NOT NULL,              -- graus decimais, WGS84
        lon             REAL NOT NULL,
        precisao_m      REAL,                       -- raio de incerteza informado (Localização do Windows)
        hdop            REAL,                       -- informado pelo receptor GNSS
        satelites       INTEGER,
        fonte           TEXT NOT NULL,              -- gnss_serial | gnss_log | windows_localizacao
        sync_status     INTEGER NOT NULL DEFAULT 0 CHECK (sync_status IN (0, 1))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rastro_local_sync ON rastro_local(sync_status)",
)


def migrar_banco_local(conexao: sqlite3.Connection) -> None:
    """
    O schema só é aplicado quando o banco é criado; um banco que já existia
    não ganha colunas novas sozinho. Aqui elas são acrescentadas, uma vez,
    sem mexer nos dados (ADD COLUMN de coluna vazia é seguro no SQLite).
    Idempotente: rodar de novo não faz nada.
    """
    # Tabela do trajeto: CREATE IF NOT EXISTS não depende de toras_local.
    # (execute, não executescript: este faria COMMIT do que estivesse aberto.)
    tinha_rastro = conexao.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rastro_local'").fetchone()
    for ddl in DDL_RASTRO_LOCAL:
        conexao.execute(ddl)
    if not tinha_rastro:
        conexao.commit()
    existentes = {linha[1] for linha in conexao.execute("PRAGMA table_info(toras_local)")}
    if not existentes:
        return  # sem tabela (banco ainda não criado): nada a migrar
    faltando = [(nome, tipo) for nome, tipo in COLUNAS_POSICAO.items() if nome not in existentes]
    for nome, tipo in faltando:
        try:
            conexao.execute(f"ALTER TABLE toras_local ADD COLUMN {nome} {tipo}")
        except sqlite3.OperationalError as e:
            # O outro processo (main.py ou sync) acrescentou a mesma coluna
            # entre o PRAGMA e o ALTER: o resultado é o mesmo, segue.
            if "duplicate column" not in str(e).lower():
                raise
    if faltando:
        conexao.commit()
        print(f"🗄️  Banco local atualizado: colunas de posição adicionadas ({', '.join(n for n, _ in faltando)}).")
