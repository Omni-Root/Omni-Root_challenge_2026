"""
sync_daemon.py — Sincronizador SQLite (local) -> PostgreSQL (central)

Substitui o antigo sync.go. O motivo de ter sido escrito em Go originalmente
era compilar sem CGO pra rodar no Raspberry Pi ARM sem dor de cabeça de
driver nativo. Como o Raspberry saiu do escopo (a máquina roda Windows, que
já precisa de Python instalado por causa do main.py de qualquer forma), a
vantagem de ter um binário Go separado desapareceu -- então trouxemos essa
lógica pra Python, uma linguagem a menos pra manter.

A lógica é a MESMA do sync.go, linha por linha, só a sintaxe muda:
  - Checa se o PostgreSQL está acessível (proxy pra "tem internet?")
  - Busca toras com sync_status = 0 no SQLite local
  - Pra cada tora: insere no Postgres de forma IDEMPOTENTE (se uuid_local
    já existe, não duplica -- importante pra sobreviver a retry após queda
    de rede no meio da sincronização)
  - Só marca como sincronizado (sync_status = 1) depois que a tora INTEIRA
    (registro + indicadores + defeitos) foi inserida com sucesso -- nunca
    fica "meio sincronizado"
  - Se der erro no meio de uma tora, para o ciclo e tenta tudo de novo no
    próximo (não segue pra próxima tora deixando a atual quebrada)

USO:
    python sync_daemon.py

CONFIGURAÇÃO:
    Mesma ideia do .env do sync.go -- crie um arquivo .env (não vai pro
    Git, já está no .gitignore) com:

        SQLITE_PATH=./omni_root_local.db
        PG_HOST=localhost
        PG_PORT=5432
        PG_USER=postgres
        PG_PASSWORD=sua_senha
        PG_DBNAME=desafio_madeira
        SYNC_INTERVALO_SEG=30

    Se não existir .env, cai pros valores padrão abaixo (mesmos do Go).
"""

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv


# ============================================================
# CONFIGURAÇÃO
# ============================================================

@dataclass
class Config:
    sqlite_path: str
    pg_host: str
    pg_port: str
    pg_user: str
    pg_password: str
    pg_dbname: str
    intervalo_seg: int

    @property
    def pg_dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} user={self.pg_user} "
            f"password={self.pg_password} dbname={self.pg_dbname} sslmode=disable"
        )


def carregar_config() -> Config:
    load_dotenv()  # lê .env se existir; se não existir, não é erro fatal

    pg_password = os.getenv("PG_PASSWORD", "")
    if not pg_password:
        print("⚠️  PG_PASSWORD não definido (.env ausente ou vazio) — tentando conectar sem senha.")

    return Config(
        sqlite_path=os.getenv("SQLITE_PATH", str(Path(".") / "omni_root_local.db")),
        pg_host=os.getenv("PG_HOST", "localhost"),
        pg_port=os.getenv("PG_PORT", "5432"),
        pg_user=os.getenv("PG_USER", "postgres"),
        pg_password=pg_password,
        pg_dbname=os.getenv("PG_DBNAME", "desafio_madeira"),
        intervalo_seg=int(os.getenv("SYNC_INTERVALO_SEG", "30")),
    )


# ============================================================
# SINCRONIZAÇÃO
# ============================================================

def sincronizar_bancos(cfg: Config) -> None:
    # 1. Tenta conectar no PostgreSQL (proxy pra "tem rede agora?")
    try:
        pg_conn = psycopg2.connect(cfg.pg_dsn, connect_timeout=5)
    except psycopg2.OperationalError:
        print("📡 Servidor PostgreSQL não encontrado na rede. Tentando depois...")
        return

    try:
        pg_conn.autocommit = False

        # 2. Conecta no SQLite local
        import sqlite3
        sqlite_conn = sqlite3.connect(cfg.sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row

        try:
            toras = buscar_toras_pendentes(sqlite_conn)

            if not toras:
                print("✅ Tudo sincronizado. Nada a enviar.")
                return

            processadas = 0
            for tora in toras:
                try:
                    sincronizar_tora(pg_conn, sqlite_conn, tora)
                    processadas += 1
                except Exception as e:
                    print(f"❌ Erro ao sincronizar tora {tora['uuid_local']}: {e}")
                    # Para o loop e tenta tudo de novo no próximo ciclo --
                    # evita deixar dados "pela metade" se a rede cair no meio.
                    break

            print(f"🚀 {processadas}/{len(toras)} toras processadas com sucesso!")
        finally:
            sqlite_conn.close()
    finally:
        pg_conn.close()


def buscar_toras_pendentes(sqlite_conn) -> list:
    cursor = sqlite_conn.execute(
        """
        SELECT id, uuid_local, maquina_id, talhao_id, log_id,
               data_inspecao, confianca_ia, status_classificacao, hash_sha256
        FROM toras_local
        WHERE sync_status = 0
        """
    )
    return cursor.fetchall()


def sincronizar_tora(pg_conn, sqlite_conn, tora) -> None:
    """
    Sincroniza uma tora inteira (registro + indicadores + defeitos) numa
    única transação Postgres. Se qualquer parte falhar, dá rollback --
    nada fica marcado como sincronizado no SQLite (fica pendente pro
    próximo ciclo).
    """
    pg_cursor = pg_conn.cursor()
    try:
        tora_remota_id = inserir_ou_buscar_tora(pg_cursor, tora)

        # --- indicadores de qualidade ---
        indicadores = buscar_indicadores_pendentes(sqlite_conn, tora["id"])
        for ind in indicadores:
            pg_cursor.execute(
                """
                INSERT INTO indicadores_qualidade
                    (tora_id, tipo_indicador, valor, unidade, metodo_medicao)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (tora_remota_id, ind["tipo_indicador"], ind["valor"], ind["unidade"], ind["metodo_medicao"]),
            )

        # --- defeitos detectados ---
        defeitos = buscar_defeitos_pendentes(sqlite_conn, tora["id"])
        for d in defeitos:
            pg_cursor.execute(
                """
                INSERT INTO defeitos_detectados
                    (tora_id, tipo_defeito, pos_x, pos_y, largura, altura, confianca)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (tora_remota_id, d["tipo_defeito"], d["pos_x"], d["pos_y"], d["largura"], d["altura"], d["confianca"]),
            )

        # Tudo certo no Postgres -- confirma a transação
        pg_conn.commit()

        # Só agora marca como sincronizado no SQLite local (banco separado,
        # não entra na transação do Postgres -- por isso a ordem importa:
        # só marcamos local depois que o commit remoto já foi confirmado)
        sqlite_conn.execute(
            "UPDATE indicadores_qualidade_local SET sync_status = 1 WHERE tora_id = ? AND sync_status = 0",
            (tora["id"],),
        )
        sqlite_conn.execute(
            "UPDATE defeitos_detectados_local SET sync_status = 1 WHERE tora_id = ? AND sync_status = 0",
            (tora["id"],),
        )
        sqlite_conn.execute(
            "UPDATE toras_local SET sync_status = 1 WHERE id = ?",
            (tora["id"],),
        )
        sqlite_conn.commit()

    except Exception:
        pg_conn.rollback()
        raise
    finally:
        pg_cursor.close()


def inserir_ou_buscar_tora(pg_cursor, tora) -> int:
    """
    Insere a tora no Postgres. Se uuid_local já existir (retry após queda
    de rede no meio de uma sincronização anterior), busca o id já existente
    em vez de duplicar -- é isso que garante idempotência.
    """
    pg_cursor.execute(
        """
        INSERT INTO toras_inspecionadas
            (uuid_local, maquina_id, talhao_id, log_id, data_inspecao,
             confianca_ia, status_classificacao, hash_sha256)
        SELECT %s, m.id_maquina, t.id_talhao, %s, %s, %s, %s, %s
        FROM maquinas m
        LEFT JOIN talhoes t ON t.nome = %s
        WHERE m.numero_serie = %s
        ON CONFLICT (uuid_local) DO NOTHING
        RETURNING id
        """,
        (
            tora["uuid_local"], tora["log_id"], tora["data_inspecao"],
            tora["confianca_ia"], tora["status_classificacao"], tora["hash_sha256"],
            tora["talhao_id"], tora["maquina_id"],
        ),
    )
    linha = pg_cursor.fetchone()
    if linha is not None:
        return linha[0]

    # Já existia (ON CONFLICT) -- busca o id remoto pelo uuid
    pg_cursor.execute(
        "SELECT id FROM toras_inspecionadas WHERE uuid_local = %s",
        (tora["uuid_local"],),
    )
    linha = pg_cursor.fetchone()
    if linha is not None:
        return linha[0]

    # Chegou aqui: o SELECT de origem (maquinas/talhoes) não achou
    # nenhuma linha correspondente a maquina_id/talhao_id -- não é bug,
    # é dado de referência faltando no Postgres. Falha com mensagem clara
    # em vez de deixar um NoneType estourar mais na frente sem contexto.
    raise ValueError(
        f"Nenhuma máquina encontrada em 'maquinas' com numero_serie = "
        f"'{tora['maquina_id']}' (talhão pedido: '{tora['talhao_id']}'). "
        f"Confiram se o seed rodou (docker compose exec postgres psql "
        f"-U <PG_USER do .env> -d <PG_DBNAME do .env> -c \"SELECT * FROM "
        f"maquinas;\") e se o maquina_id do config.json bate com o "
        f"numero_serie cadastrado no Postgres."
    )


def buscar_indicadores_pendentes(sqlite_conn, tora_id_local: int) -> list:
    cursor = sqlite_conn.execute(
        """
        SELECT id, tipo_indicador, valor, unidade, metodo_medicao
        FROM indicadores_qualidade_local
        WHERE tora_id = ? AND sync_status = 0
        """,
        (tora_id_local,),
    )
    return cursor.fetchall()


def buscar_defeitos_pendentes(sqlite_conn, tora_id_local: int) -> list:
    cursor = sqlite_conn.execute(
        """
        SELECT id, tipo_defeito, pos_x, pos_y, largura, altura, confianca
        FROM defeitos_detectados_local
        WHERE tora_id = ? AND sync_status = 0
        """,
        (tora_id_local,),
    )
    return cursor.fetchall()


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def main() -> None:
    cfg = carregar_config()
    print("⚙️  Sincronizador Python iniciado! Rodando em background...")
    print(f"📁 SQLite local : {cfg.sqlite_path}")
    print(f"🐘 Postgres alvo: {cfg.pg_host}:{cfg.pg_port}/{cfg.pg_dbname}")

    try:
        while True:
            try:
                sincronizar_bancos(cfg)
            except Exception as e:
                print(f"⚠️  Ciclo de sincronização terminou com erro: {e}")
            time.sleep(cfg.intervalo_seg)
    except KeyboardInterrupt:
        print("\n🛑 Encerrando por solicitação do usuário...")


if __name__ == "__main__":
    main()
