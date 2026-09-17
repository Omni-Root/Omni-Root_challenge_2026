-- ============================================================
-- OMNI-ROOT — SETUP DO POSTGRESQL CENTRAL (arquivo único)
-- Challenge 2026 — FIAP x John Deere / Suzano / Eldorado
-- ============================================================
-- Este é o ÚNICO script do banco central. Roda inteiro, quantas vezes
-- for preciso (idempotente): todo CREATE usa IF NOT EXISTS, todo INSERT
-- é protegido contra duplicata, o trigger é recriado. Rodar de novo num
-- banco com dados não apaga nem duplica nada.
--
-- Ordem (importa: seed depende das tabelas):
--   1. SCHEMA     — tabelas e índices que o sync_daemon.py preenche e o
--                   dashboard lê
--   2. SEED       — máquina + talhão de teste (batem com o config.json)
--   3. DENSIDADE  — tabela clones_densidade + clones da JD (fonte de
--                   verdade da densidade; o sync baixa para a máquina)
--   4. TEMPO REAL — trigger NOTIFY para o dashboard atualizar sozinho
--   5. LIMPEZA    — remove objetos de versões anteriores que nada usa
--
-- O banco LOCAL da máquina (SQLite) tem schema próprio: schema_sqlite.sql.
--
-- ------------------------------------------------------------
-- COMO RODAR
-- ------------------------------------------------------------
-- Docker (primeira subida): o docker-compose.yml monta este arquivo em
-- docker-entrypoint-initdb.d e o Postgres roda sozinho.
--
-- Banco já de pé (aplicar mudanças):
--   PowerShell:
--     Get-Content "Banco de dados\setup_completo.sql" | docker compose exec -T postgres psql -U postgres -d desafio_madeira
--   Git Bash / Linux / Mac:
--     docker compose exec -T postgres psql -U postgres -d desafio_madeira < "Banco de dados/setup_completo.sql"
--
-- Conferir:
--   SELECT COUNT(*) FROM maquinas;          -- 1
--   SELECT COUNT(*) FROM talhoes;           -- 1
--   SELECT COUNT(*) FROM clones_densidade;  -- 17
--   SELECT tgname FROM pg_trigger WHERE tgname = 'trg_notificar_nova_tora';
-- ============================================================


-- ============================================================
-- PARTE 1 — SCHEMA
-- ============================================================

-- Máquinas de colheita. numero_serie é a chave que a máquina de campo
-- usa (maquina_id no config.json): o sync resolve o id por ela.
CREATE TABLE IF NOT EXISTS maquinas (
    id_maquina      SERIAL PRIMARY KEY,
    modelo          VARCHAR(100) NOT NULL,
    numero_serie    VARCHAR(100) UNIQUE NOT NULL,
    criado_em       TIMESTAMP NOT NULL DEFAULT now()
);

-- Talhões (áreas de plantio). A máquina grava o nome; o sync resolve o id.
CREATE TABLE IF NOT EXISTS talhoes (
    id_talhao       SERIAL PRIMARY KEY,
    nome            VARCHAR(100) NOT NULL,
    area_hectares   NUMERIC(10,2),
    especie         VARCHAR(100),        -- ex: "Eucalyptus grandis x urophylla"
    criado_em       TIMESTAMP NOT NULL DEFAULT now()
);

-- Uma linha por inspeção sincronizada do campo.
CREATE TABLE IF NOT EXISTS toras_inspecionadas (
    id                    SERIAL PRIMARY KEY,
    uuid_local            UUID UNIQUE NOT NULL,          -- gerado na máquina (uuid4); garante idempotência do sync
    maquina_id            INTEGER REFERENCES maquinas(id_maquina),
    talhao_id             INTEGER REFERENCES talhoes(id_talhao),
    log_id                VARCHAR(100) NOT NULL,          -- id legível (LOG-AAAAMMDDhhmmss-xxxx), usado no export StanForD
    data_inspecao         TIMESTAMP NOT NULL,             -- hora local da máquina
    confianca_ia          NUMERIC(5,4) NOT NULL,          -- saúde da tora, 0-1 (ex: 0.9123)
    status_classificacao  VARCHAR(20) NOT NULL
                          CHECK (status_classificacao IN ('aprovado', 'quarentena', 'reprovado')),
    hash_sha256           VARCHAR(64) NOT NULL,           -- integridade do registro (calculado na máquina)
    data_sincronizacao    TIMESTAMP NOT NULL DEFAULT now(),
    criado_em             TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_toras_talhao   ON toras_inspecionadas(talhao_id);
CREATE INDEX IF NOT EXISTS idx_toras_maquina  ON toras_inspecionadas(maquina_id);
CREATE INDEX IF NOT EXISTS idx_toras_data     ON toras_inspecionadas(data_inspecao);
CREATE INDEX IF NOT EXISTS idx_toras_status   ON toras_inspecionadas(status_classificacao);

-- 8 indicadores por tora (ver README, seção 5). metodo_medicao diz de onde
-- o número veio — o dashboard lê o clone de 'lookup_clone_<ID>' e a vista
-- de 'imagem_secao_*' / 'imagem_lateral_*'.
CREATE TABLE IF NOT EXISTS indicadores_qualidade (
    id                SERIAL PRIMARY KEY,
    tora_id           INTEGER NOT NULL REFERENCES toras_inspecionadas(id) ON DELETE CASCADE,
    tipo_indicador    VARCHAR(30) NOT NULL
                      CHECK (tipo_indicador IN ('densidade', 'massa_seca', 'altura', 'diametro', 'tortuosidade', 'porcentagem_casca', 'volume_util', 'apodrecimento_pragas')),
    valor             NUMERIC(10,4) NOT NULL,
    unidade           VARCHAR(20),             -- "kg/m3", "cm", "indice", "%", "m3", "kg"
    metodo_medicao    VARCHAR(50) NOT NULL,    -- ex: "lookup_clone_SP3108", "imagem_secao_marcador_aruco", "opencv_eixo_flecha"
    criado_em         TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_indicadores_tora ON indicadores_qualidade(tora_id);
CREATE INDEX IF NOT EXISTS idx_indicadores_tipo ON indicadores_qualidade(tipo_indicador);

-- Defeitos confirmados pelos filtros (classes do modelo: Live_Knot,
-- Dead_Knot, Knot_missing, knot_with_crack, Crack, Marrow, Quartzity,
-- resin — e "Crack" também pela visão clássica na face de corte).
CREATE TABLE IF NOT EXISTS defeitos_detectados (
    id              SERIAL PRIMARY KEY,
    tora_id         INTEGER NOT NULL REFERENCES toras_inspecionadas(id) ON DELETE CASCADE,
    tipo_defeito    VARCHAR(50) NOT NULL,
    pos_x           NUMERIC(8,2) NOT NULL,     -- bounding box em pixels do frame (x, y, largura, altura)
    pos_y           NUMERIC(8,2) NOT NULL,
    largura         NUMERIC(8,2) NOT NULL,
    altura          NUMERIC(8,2) NOT NULL,
    confianca       NUMERIC(5,4) NOT NULL,
    criado_em       TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_defeitos_tora ON defeitos_detectados(tora_id);


-- ============================================================
-- PARTE 2 — SEED DE TESTE (bate com maquina_id / talhao_id do config.json)
-- ============================================================

INSERT INTO maquinas (modelo, numero_serie)
VALUES ('Harvester (maquete de testes)', 'RASPI-DEMO-01')
ON CONFLICT (numero_serie) DO NOTHING;

INSERT INTO talhoes (nome, area_hectares, especie)
SELECT 'Talhão Demo', 12.5, 'Eucalyptus grandis x urophylla'
WHERE NOT EXISTS (SELECT 1 FROM talhoes WHERE nome = 'Talhão Demo');


-- ============================================================
-- PARTE 3 — DENSIDADE POR CLONE (fonte de verdade)
-- ============================================================
-- Feedback da John Deere na mentoria: densidade não é um número fixo num
-- arquivo — a mesma espécie tem dezenas de clones, cada um com a sua.
-- Numa operação de ~6.000 toras/dia (Eldorado), 30 kg/m3 de diferença
-- são ~26 t/dia de massa prevista. Por isso:
--   1. esta tabela é a FONTE DE VERDADE (o laboratório atualiza aqui);
--   2. o sync_daemon.py baixa e regrava data/clones_densidade.json na máquina;
--   3. o main.py recarrega o JSON sozinho quando ele muda — cadastrou
--      aqui, em segundos a máquina usa o valor novo, sem reiniciar.
--
-- tipo_dado: 'laboratorio' (laudo real), 'literatura' (valor publicado para
-- o clone) ou 'referencia_generica' (média do híbrido, aguardando laudo).
-- densidade_base NULL é permitido de propósito: o main.py avisa e usa
-- 500 kg/m3 em vez de mascarar dado faltante.

CREATE TABLE IF NOT EXISTS clones_densidade (
    id              SERIAL PRIMARY KEY,
    clone_id        VARCHAR(50) UNIQUE NOT NULL,   -- ex: 'I144', 'GG100', 'AEC 0144'
    especie         VARCHAR(100),                  -- ex: 'Urograndis (E. grandis x E. urophylla)'
    densidade_base  NUMERIC(6,1),                  -- kg/m3
    densidade_min   NUMERIC(6,1),                  -- faixa, quando a fonte dá faixa em vez de valor único
    densidade_max   NUMERIC(6,1),
    tipo_dado       VARCHAR(30) NOT NULL DEFAULT 'referencia_generica',
    fonte           TEXT,
    atualizado_em   TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_clones_densidade_clone ON clones_densidade(clone_id);

-- Clones citados pela JD na mentoria (11) + códigos da planilha de inventário (6)
INSERT INTO clones_densidade (clone_id, especie, densidade_base, densidade_min, densidade_max, tipo_dado, fonte) VALUES
    ('I144', 'Urograndis (híbrido espontâneo de E. urophylla)', 485.0, 460.0, 510.0, 'literatura',
     'Faixa 460-510 kg/m3 publicada por fontes comerciais de viveiro (mfrural, Avam Flora). NOTA: I144 e AEC 0144 parecem ser o mesmo clone -- confirmar com o contato da JD.'),

    ('AEC 0144', 'Urograndis (híbrido espontâneo de E. urophylla)', 485.0, 460.0, 510.0, 'literatura',
     'Mesmo material que I144 (fontes descrevem "Clone I144 (AEC-0144)"). Clone de domínio público segundo Embrapa CT-316.'),

    ('GG100', 'Urograndis (E. urophylla)', 490.0, NULL, NULL, 'referencia_generica',
     'Sem densidade específica publicada encontrada. Clone de domínio público (Embrapa CT-316). OBS relevante para nosso indicador de tortuosidade: a Embrapa registra que o GG100 apresenta tortuosidade de fuste. Aguardando dado de laboratório.'),

    ('VM01 (VM1)', 'Urograndis', 490.0, NULL, NULL, 'referencia_generica',
     'Sem densidade específica publicada encontrada. Aguardando dado de laboratório.'),

    ('I042', 'Urograndis', 490.0, NULL, NULL, 'referencia_generica',
     'Sem densidade específica publicada encontrada. Aguardando dado de laboratório.'),

    ('58', 'Urograndis', 490.0, NULL, NULL, 'referencia_generica',
     'Sem densidade específica publicada encontrada. Aguardando dado de laboratório.'),

    ('386', 'Urograndis', 490.0, NULL, NULL, 'referencia_generica',
     'Sem densidade específica publicada encontrada. Aguardando dado de laboratório.'),

    ('H13', 'Urograndis', 490.0, NULL, NULL, 'referencia_generica',
     'Sem densidade específica publicada encontrada. Aguardando dado de laboratório.'),

    ('H15', 'Urograndis', 490.0, NULL, NULL, 'referencia_generica',
     'Sem densidade específica publicada encontrada. Aguardando dado de laboratório.'),

    ('H17', 'Urograndis', 490.0, NULL, NULL, 'referencia_generica',
     'Sem densidade específica publicada encontrada. Aguardando dado de laboratório.'),

    ('H19', 'Urograndis', 490.0, NULL, NULL, 'referencia_generica',
     'Sem densidade específica publicada encontrada. Aguardando dado de laboratório.')
ON CONFLICT (clone_id) DO NOTHING;

INSERT INTO clones_densidade (clone_id, especie, densidade_base, tipo_dado, fonte) VALUES
    ('SP3108', 'Eucalyptus sp. (código interno)', 490.0, 'referencia_generica', 'Código interno da planilha de inventário. Sem correspondência pública. Referência de híbrido comercial E. grandis x urophylla.'),
    ('SP2974', 'Eucalyptus sp. (código interno)', 490.0, 'referencia_generica', 'Código interno da planilha de inventário. Sem correspondência pública.'),
    ('SP3153', 'Eucalyptus sp. (código interno)', 490.0, 'referencia_generica', 'Código interno da planilha de inventário. Sem correspondência pública.'),
    ('SP2887', 'Eucalyptus sp. (código interno)', 490.0, 'referencia_generica', 'Código interno da planilha de inventário. Sem correspondência pública.'),
    ('CO41H_TEST', 'Eucalyptus sp. (clone de teste)', 490.0, 'referencia_generica', 'Clone de teste.'),
    ('SP1048_TEST', 'Eucalyptus sp. (clone de teste)', 490.0, 'referencia_generica', 'Clone de teste.')
ON CONFLICT (clone_id) DO NOTHING;


-- ============================================================
-- PARTE 4 — TEMPO REAL (Postgres -> dashboard)
-- ============================================================
-- A cada tora inserida pelo sync, NOTIFY no canal 'omniroot_toras'. O
-- servidor do dashboard fica em LISTEN e repassa aos navegadores por SSE —
-- a "última inspeção" aparece na tela segundos depois da máquina
-- sincronizar. Sem o trigger o dashboard cai num polling de 5 s.

CREATE OR REPLACE FUNCTION notificar_nova_tora() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify(
        'omniroot_toras',
        json_build_object(
            'id',            NEW.id,
            'uuid_local',    NEW.uuid_local,
            'status',        NEW.status_classificacao,
            'maquina_id',    NEW.maquina_id,
            'talhao_id',     NEW.talhao_id,
            'data_inspecao', NEW.data_inspecao
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_notificar_nova_tora ON toras_inspecionadas;
CREATE TRIGGER trg_notificar_nova_tora
    AFTER INSERT ON toras_inspecionadas
    FOR EACH ROW EXECUTE FUNCTION notificar_nova_tora();


-- ============================================================
-- PARTE 5 — LIMPEZA DE VERSÕES ANTERIORES
-- ============================================================
-- usuarios / logs_auditoria: previstas para um RBAC que não foi
-- implementado (o dashboard autentica por credencial única no .env).
-- Ficar no schema sem código que as use só gera pergunta. Removidas
-- APENAS se estiverem vazias — se alguém tiver populado, ficam.
-- pgcrypto: entrou "para gen_random_uuid()", mas o UUID é gerado na
-- máquina (Python). Nada a usa.

DO $$
DECLARE
    n BIGINT;
BEGIN
    -- EXECUTE (SQL dinâmico) de propósito: num banco zerado a tabela não
    -- existe, e um SELECT direto no IF quebraria no planejamento.
    IF to_regclass('public.logs_auditoria') IS NOT NULL THEN
        EXECUTE 'SELECT COUNT(*) FROM logs_auditoria' INTO n;
        IF n = 0 THEN
            EXECUTE 'DROP TABLE logs_auditoria';
        END IF;
    END IF;
    IF to_regclass('public.usuarios') IS NOT NULL
       AND to_regclass('public.logs_auditoria') IS NULL THEN
        EXECUTE 'SELECT COUNT(*) FROM usuarios' INTO n;
        IF n = 0 THEN
            EXECUTE 'DROP TABLE usuarios';
        END IF;
    END IF;
END $$;

DROP EXTENSION IF EXISTS pgcrypto;
