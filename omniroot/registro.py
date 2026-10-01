"""
registro.py — Gravação local (SQLite): abrir/migrar o banco, salvar e atualizar a tora com indicadores, defeitos e posição.

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from omniroot.banco_local import migrar_banco_local
from omniroot.classificacao import gerar_hash_sha256
from omniroot.config import Config


# ============================================================
# CAMADA DE BANCO DE DADOS (SQLite local)
# ============================================================

def conectar_banco(cfg: Config) -> sqlite3.Connection:
    conexao = sqlite3.connect(cfg.sqlite_path)
    conexao.execute("PRAGMA foreign_keys = ON")
    # WAL: o sync_daemon lê (snapshot) enquanto esta thread escreve, sem
    # "database is locked" e sem misturar versões de uma mesma tora.
    conexao.execute("PRAGMA journal_mode=WAL")

    # Verifica se a tabela toras_local já existe, caso contrário carrega o schema
    tabelas = conexao.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='toras_local'").fetchall()
    if not tabelas:
        schema_path = Path(cfg.sqlite_path).parent / "Banco de dados" / "schema_sqlite.sql"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                conexao.executescript(f.read())
    # Banco criado antes das colunas de posição: acrescenta (ver omniroot/banco_local.py).
    migrar_banco_local(conexao)
    return conexao


# Campos da posição (dict de LeitorPosicao.atual) na ordem das colunas pos_*
# de toras_local: "lat" -> pos_lat etc.
CAMPOS_POSICAO = ("lat", "lon", "hdop", "satelites", "fonte", "idade_s", "precisao_m")


def _hash_inspecao(uuid_local: str, log_id: str, indicadores: dict, defeitos: list,
                   status: str, posicao: dict | None) -> str:
    """Hash de integridade da tora. A posição entra quando existe (sem posição, o hash é o de antes)."""
    dados = {
        "uuid_local": uuid_local,
        "log_id": log_id,
        "indicadores": indicadores,
        "defeitos": defeitos,
        "status": status,
    }
    if posicao is not None:
        dados["posicao"] = {k: posicao.get(k) for k in CAMPOS_POSICAO}
    return gerar_hash_sha256(dados)


def salvar_inspecao(
    conexao: sqlite3.Connection,
    cfg: Config,
    indicadores: dict,
    defeitos: list,
    status_classificacao: str,
    confianca_media: float,
    uuid_local: str | None = None,
    posicao: dict | None = None,
) -> str:
    """
    Grava a tora + indicadores + defeitos nas 3 tabelas locais. `uuid_local`
    fixo permite atualizar depois. `posicao` (de LeitorPosicao.atual) é a
    posição da máquina no momento da gravação; None = tora sem posição.
    """
    cursor = conexao.cursor()

    uuid_local = uuid_local or str(uuid.uuid4())
    log_id = f"LOG-{datetime.now():%Y%m%d%H%M%S}-{uuid_local[:4]}"
    data_inspecao = datetime.now().isoformat(timespec="seconds")

    hash_dados = _hash_inspecao(uuid_local, log_id, indicadores, defeitos, status_classificacao, posicao)
    p = posicao or {}

    # --- 1. Insere a tora ---
    cursor.execute(
        f"""
        INSERT INTO toras_local
            (uuid_local, maquina_id, talhao_id, log_id, data_inspecao,
             confianca_ia, status_classificacao, hash_sha256, sync_status,
             {", ".join("pos_" + c for c in CAMPOS_POSICAO)})
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, {", ".join("?" for _ in CAMPOS_POSICAO)})
        """,
        (uuid_local, cfg.maquina_id, cfg.talhao_id, log_id, data_inspecao,
         confianca_media, status_classificacao, hash_dados,
         *(p.get(c) for c in CAMPOS_POSICAO)),
    )
    tora_id_local = cursor.lastrowid

    # --- 2. Insere os indicadores de qualidade ---
    for tipo, dados_indicador in indicadores.items():
        cursor.execute(
            """
            INSERT INTO indicadores_qualidade_local
                (tora_id, tipo_indicador, valor, unidade, metodo_medicao, sync_status)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (tora_id_local, tipo, dados_indicador["valor"],
             dados_indicador["unidade"], dados_indicador["metodo"]),
        )

    # --- 3. Insere os defeitos detectados ---
    for defeito in defeitos:
        cursor.execute(
            """
            INSERT INTO defeitos_detectados_local
                (tora_id, tipo_defeito, pos_x, pos_y, largura, altura, confianca, sync_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (tora_id_local, defeito["tipo_defeito"], defeito["pos_x"], defeito["pos_y"],
             defeito["largura"], defeito["altura"], defeito["confianca"]),
        )

    conexao.commit()
    return uuid_local


def atualizar_inspecao(
    conexao: sqlite3.Connection,
    uuid_local: str,
    indicadores: dict,
    defeitos: list,
    status_classificacao: str,
    confianca_media: float,
) -> bool:
    """
    Atualiza uma tora já gravada (evento incremental): troca indicadores e
    defeitos pelos consolidados até agora, recalcula o hash e volta
    sync_status para 0 — o sync_daemon reenvia e o Postgres é atualizado
    (UPSERT por uuid_local). Devolve False se o uuid não existe.
    A posição NÃO muda: é a do momento em que a tora foi gravada (o corte).
    """
    cursor = conexao.cursor()
    linha = cursor.execute(
        f"SELECT id, log_id, {', '.join('pos_' + c for c in CAMPOS_POSICAO)} "
        "FROM toras_local WHERE uuid_local = ?",
        (uuid_local,),
    ).fetchone()
    if linha is None:
        return False
    tora_id, log_id = linha[0], linha[1]
    posicao = None
    if linha[2] is not None and linha[3] is not None:
        posicao = dict(zip(CAMPOS_POSICAO, linha[2:]))
    hash_dados = _hash_inspecao(uuid_local, log_id, indicadores, defeitos, status_classificacao, posicao)
    cursor.execute(
        "UPDATE toras_local SET confianca_ia = ?, status_classificacao = ?, hash_sha256 = ?, sync_status = 0 WHERE id = ?",
        (confianca_media, status_classificacao, hash_dados, tora_id),
    )
    cursor.execute("DELETE FROM indicadores_qualidade_local WHERE tora_id = ?", (tora_id,))
    cursor.execute("DELETE FROM defeitos_detectados_local WHERE tora_id = ?", (tora_id,))
    for tipo, d in indicadores.items():
        cursor.execute(
            "INSERT INTO indicadores_qualidade_local (tora_id, tipo_indicador, valor, unidade, metodo_medicao, sync_status) VALUES (?, ?, ?, ?, ?, 0)",
            (tora_id, tipo, d["valor"], d["unidade"], d["metodo"]),
        )
    for d in defeitos:
        cursor.execute(
            "INSERT INTO defeitos_detectados_local (tora_id, tipo_defeito, pos_x, pos_y, largura, altura, confianca, sync_status) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (tora_id, d["tipo_defeito"], d["pos_x"], d["pos_y"], d["largura"], d["altura"], d["confianca"]),
        )
    conexao.commit()
    return True
