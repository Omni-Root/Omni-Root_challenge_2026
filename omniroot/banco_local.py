"""
banco_local.py — definições do SQLite da máquina compartilhadas entre o
main.py (quem grava) e o sync_daemon.py (quem lê e envia).

Hoje: as colunas de posição (GNSS) de toras_local e a migração que as
acrescenta num banco criado antes delas. Os dois processos chamam a mesma
migração ao abrir o banco — tanto faz qual sobe primeiro.
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


def migrar_banco_local(conexao: sqlite3.Connection) -> None:
    """
    O schema só é aplicado quando o banco é criado; um banco que já existia
    não ganha colunas novas sozinho. Aqui elas são acrescentadas, uma vez,
    sem mexer nos dados (ADD COLUMN de coluna vazia é seguro no SQLite).
    Idempotente: rodar de novo não faz nada.
    """
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
