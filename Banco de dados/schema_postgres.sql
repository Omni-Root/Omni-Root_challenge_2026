-- ============================================================
-- SCHEMA POSTGRESQL — Banco central (notebook / nuvem)
-- Projeto: Qualidade da Madeira — Challenge FIAP x John Deere/Suzano
-- ============================================================
-- Este banco recebe os dados sincronizados do Raspberry Pi
-- (via sync.go) e alimenta o dashboard.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto; -- necessário para gen_random_uuid()

-- ------------------------------------------------------------
-- 1. MAQUINAS — colhedoras/máquinas florestais com câmera acoplada
-- ------------------------------------------------------------
CREATE TABLE maquinas (
    id_maquina      SERIAL PRIMARY KEY,
    modelo          VARCHAR(100) NOT NULL,
    numero_serie    VARCHAR(100) UNIQUE NOT NULL,
    criado_em       TIMESTAMP NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- 2. TALHOES — áreas florestais monitoradas
-- ------------------------------------------------------------
CREATE TABLE talhoes (
    id_talhao       SERIAL PRIMARY KEY,
    nome            VARCHAR(100) NOT NULL,
    area_hectares   NUMERIC(10,2),
    especie         VARCHAR(100),        -- ex: "Eucalipto"
    criado_em       TIMESTAMP NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- 3. TORAS_INSPECIONADAS — registro central de cada inspeção
-- ------------------------------------------------------------
-- uuid_local é gerado NO RASPBERRY no momento da inspeção.
-- É a chave que garante sincronização idempotente (sem duplicar
-- registros se a rede cair no meio do envio).
-- ------------------------------------------------------------
CREATE TABLE toras_inspecionadas (
    id                    SERIAL PRIMARY KEY,
    uuid_local            UUID UNIQUE NOT NULL,          -- gerado na borda (Raspberry)
    maquina_id            INTEGER REFERENCES maquinas(id_maquina),
    talhao_id             INTEGER REFERENCES talhoes(id_talhao),
    log_id                VARCHAR(100) NOT NULL,          -- id no padrão StanForD
    data_inspecao         TIMESTAMP NOT NULL,
    confianca_ia          NUMERIC(5,4) NOT NULL,          -- ex: 0.9123 = 91.23%
    status_classificacao  VARCHAR(20) NOT NULL
                          CHECK (status_classificacao IN ('aprovado', 'quarentena', 'reprovado')),
    hash_sha256           VARCHAR(64) NOT NULL,           -- integridade do registro
    data_sincronizacao    TIMESTAMP NOT NULL DEFAULT now(),
    criado_em             TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_toras_talhao   ON toras_inspecionadas(talhao_id);
CREATE INDEX idx_toras_maquina  ON toras_inspecionadas(maquina_id);
CREATE INDEX idx_toras_data     ON toras_inspecionadas(data_inspecao);
CREATE INDEX idx_toras_status   ON toras_inspecionadas(status_classificacao);

-- ------------------------------------------------------------
-- 4. INDICADORES_QUALIDADE — os 4 indicadores medidos por tora
-- ------------------------------------------------------------
CREATE TABLE indicadores_qualidade (
    id                SERIAL PRIMARY KEY,
    tora_id           INTEGER NOT NULL REFERENCES toras_inspecionadas(id) ON DELETE CASCADE,
    tipo_indicador    VARCHAR(30) NOT NULL
                      CHECK (tipo_indicador IN ('densidade', 'altura', 'tortuosidade', 'apodrecimento_pragas')),
    valor             NUMERIC(10,4) NOT NULL,
    unidade           VARCHAR(20),             -- ex: "kg/m3", "m", "indice", "%"
    metodo_medicao    VARCHAR(50) NOT NULL,    -- ex: "fusao_sensores", "imagem_4k_ultrassom", "opencv_contorno", "yolo"
    criado_em         TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_indicadores_tora ON indicadores_qualidade(tora_id);
CREATE INDEX idx_indicadores_tipo ON indicadores_qualidade(tipo_indicador);

-- ------------------------------------------------------------
-- 5. DEFEITOS_DETECTADOS — bounding boxes das detecções do YOLO
-- ------------------------------------------------------------
CREATE TABLE defeitos_detectados (
    id              SERIAL PRIMARY KEY,
    tora_id         INTEGER NOT NULL REFERENCES toras_inspecionadas(id) ON DELETE CASCADE,
    tipo_defeito    VARCHAR(50) NOT NULL,      -- ex: "praga", "apodrecimento", "tortuosidade"
    pos_x           NUMERIC(8,2) NOT NULL,     -- coordenadas normalizadas da bounding box
    pos_y           NUMERIC(8,2) NOT NULL,
    largura         NUMERIC(8,2) NOT NULL,
    altura          NUMERIC(8,2) NOT NULL,
    confianca       NUMERIC(5,4) NOT NULL,
    criado_em       TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_defeitos_tora ON defeitos_detectados(tora_id);

-- ------------------------------------------------------------
-- 6. USUARIOS — controle de acesso do dashboard (RBAC)
-- ------------------------------------------------------------
CREATE TABLE usuarios (
    id              SERIAL PRIMARY KEY,
    nome            VARCHAR(150) NOT NULL,
    email           VARCHAR(150) UNIQUE NOT NULL,
    senha_hash      VARCHAR(255) NOT NULL,
    papel           VARCHAR(20) NOT NULL
                    CHECK (papel IN ('admin', 'engenheiro', 'visualizador')),
    ativo           BOOLEAN NOT NULL DEFAULT true,
    criado_em       TIMESTAMP NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- 7. LOGS_AUDITORIA — rastreabilidade de ações no sistema
-- ------------------------------------------------------------
CREATE TABLE logs_auditoria (
    id              SERIAL PRIMARY KEY,
    usuario_id      INTEGER REFERENCES usuarios(id),
    acao            VARCHAR(100) NOT NULL,     -- ex: "visualizou_dashboard", "editou_status"
    tabela_afetada  VARCHAR(50),
    registro_id     INTEGER,
    criado_em       TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_auditoria_usuario ON logs_auditoria(usuario_id);
CREATE INDEX idx_auditoria_data    ON logs_auditoria(criado_em);

-- ============================================================
-- EXEMPLO DE INSERT IDEMPOTENTE (usar isso no sync.go!)
-- ============================================================
-- INSERT INTO toras_inspecionadas
--     (uuid_local, maquina_id, talhao_id, log_id, data_inspecao,
--      confianca_ia, status_classificacao, hash_sha256)
-- VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
-- ON CONFLICT (uuid_local) DO NOTHING;
--
-- Isso é o que torna a sincronização segura: se o Go tentar
-- reenviar o mesmo registro (por causa de retry após queda de
-- rede), o Postgres simplesmente ignora, sem duplicar.
-- ============================================================
