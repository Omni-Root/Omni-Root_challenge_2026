-- ============================================================
-- SCHEMA SQLITE — Banco local no Raspberry Pi (stanford_local.db)
-- Projeto: Qualidade da Madeira — Challenge FIAP x John Deere/Suzano
-- ============================================================
-- Este banco roda offline na máquina florestal. O script Python
-- (Tarefa 2) grava aqui. O sync.go (Tarefa 3) lê os registros com
-- sync_status = 0 e envia para o PostgreSQL quando há rede.
--
-- Diferenças em relação ao Postgres:
--   - Sem RBAC / auditoria (não faz sentido offline sem login)
--   - Sem tabela MAQUINAS/TALHOES completas — só um id de referência,
--     já que a máquina "sabe quem ela é" e onde está operando
--   - uuid é gerado em Python (uuid.uuid4()) antes do INSERT,
--     não existe função nativa de UUID no SQLite
--   - Cada tabela tem sync_status (0 = pendente, 1 = sincronizado)
-- ============================================================

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- 1. TORAS_LOCAL — equivalente local de toras_inspecionadas
-- ------------------------------------------------------------
CREATE TABLE toras_local (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid_local            TEXT UNIQUE NOT NULL,      -- gerado em Python: str(uuid.uuid4())
    maquina_id            TEXT NOT NULL,             -- numero_serie da máquina (referência simples)
    talhao_id             TEXT,                      -- nome/código do talhão atual
    log_id                TEXT NOT NULL,             -- id no padrão StanForD
    data_inspecao         TEXT NOT NULL,              -- ISO 8601: '2026-07-03T14:32:00'
    confianca_ia          REAL NOT NULL,
    status_classificacao  TEXT NOT NULL
                          CHECK (status_classificacao IN ('aprovado', 'quarentena', 'reprovado')),
    hash_sha256           TEXT NOT NULL,
    sync_status           INTEGER NOT NULL DEFAULT 0 CHECK (sync_status IN (0, 1)),
    criado_em             TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_toras_local_sync ON toras_local(sync_status);

-- ------------------------------------------------------------
-- 2. INDICADORES_QUALIDADE_LOCAL
-- ------------------------------------------------------------
CREATE TABLE indicadores_qualidade_local (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    tora_id           INTEGER NOT NULL REFERENCES toras_local(id) ON DELETE CASCADE,
    tipo_indicador    TEXT NOT NULL
                      CHECK (tipo_indicador IN ('densidade', 'massa_seca', 'altura', 'diametro', 'tortuosidade', 'porcentagem_casca', 'volume_util', 'apodrecimento_pragas')),
    valor             REAL NOT NULL,
    unidade           TEXT,
    metodo_medicao    TEXT NOT NULL,
    sync_status       INTEGER NOT NULL DEFAULT 0 CHECK (sync_status IN (0, 1)),
    criado_em         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_indicadores_local_tora ON indicadores_qualidade_local(tora_id);
CREATE INDEX idx_indicadores_local_sync ON indicadores_qualidade_local(sync_status);

-- ------------------------------------------------------------
-- 3. DEFEITOS_DETECTADOS_LOCAL
-- ------------------------------------------------------------
CREATE TABLE defeitos_detectados_local (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tora_id         INTEGER NOT NULL REFERENCES toras_local(id) ON DELETE CASCADE,
    tipo_defeito    TEXT NOT NULL,
    pos_x           REAL NOT NULL,
    pos_y           REAL NOT NULL,
    largura         REAL NOT NULL,
    altura          REAL NOT NULL,
    confianca       REAL NOT NULL,
    sync_status     INTEGER NOT NULL DEFAULT 0 CHECK (sync_status IN (0, 1)),
    criado_em       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_defeitos_local_tora ON defeitos_detectados_local(tora_id);
CREATE INDEX idx_defeitos_local_sync ON defeitos_detectados_local(sync_status);

-- ============================================================
-- NOTA IMPORTANTE PARA A TAREFA 2 (script Python no Raspberry)
-- ============================================================
-- Ao inserir uma nova tora, sempre gerar o uuid_local em Python:
--
--   import uuid
--   novo_uuid = str(uuid.uuid4())
--
-- E usar esse mesmo uuid_local nas 3 tabelas relacionadas, para
-- que o sync.go consiga rastrear o que já foi enviado.
-- ============================================================
