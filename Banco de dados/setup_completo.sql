-- ============================================================
-- OMNI-ROOT — SETUP COMPLETO DO BANCO POSTGRESQL
-- Challenge 2026 — FIAP x John Deere / Suzano / Eldorado
-- ============================================================
-- Arquivo único, para rodar UMA vez e deixar o banco pronto.
-- Junta, nesta ordem (a ordem importa: o seed depende das tabelas
-- existirem, então não troque):
--
--   PARTE 1 — SCHEMA      : cria as 7 tabelas principais
--   PARTE 2 — SEED TESTE  : máquina + talhão de teste (bate com o config.json)
--   PARTE 3 — DENSIDADE   : tabela clones_densidade + os clones da John Deere
--
-- É IDEMPOTENTE: todos os CREATE usam IF NOT EXISTS e todos os
-- INSERT são protegidos contra duplicata. Rodar duas vezes não
-- quebra nada nem duplica dado.
--
-- ------------------------------------------------------------
-- COMO RODAR (Docker no Windows)
-- ------------------------------------------------------------
-- Opção A — montar como script de inicialização no docker-compose.yml
--   (roda sozinho na primeira subida do container):
--
--     volumes:
--       - ./Banco de dados/setup_completo.sql:/docker-entrypoint-initdb.d/01_setup.sql:ro
--
-- Opção B — rodar manualmente num banco que já está de pé:
--
--   PowerShell:
--     Get-Content "Banco de dados\setup_completo.sql" | docker compose exec -T postgres psql -U postgres -d desafio_madeira
--
--   Git Bash / WSL / Linux / Mac:
--     docker compose exec -T postgres psql -U postgres -d desafio_madeira < "Banco de dados/setup_completo.sql"
--
-- ------------------------------------------------------------
-- COMO CONFERIR SE DEU CERTO (rode depois)
-- ------------------------------------------------------------
--   SELECT COUNT(*) FROM maquinas;           -- esperado: 1
--   SELECT COUNT(*) FROM talhoes;            -- esperado: 1
--   SELECT COUNT(*) FROM clones_densidade;   -- esperado: 17
-- ============================================================


-- ############################################################
-- PARTE 1 — SCHEMA (tabelas principais)
-- ############################################################

CREATE EXTENSION IF NOT EXISTS pgcrypto; -- necessário para gen_random_uuid()

-- ------------------------------------------------------------
-- 1. MAQUINAS — colhedoras/máquinas florestais com câmera acoplada
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS maquinas (
    id_maquina      SERIAL PRIMARY KEY,
    modelo          VARCHAR(100) NOT NULL,
    numero_serie    VARCHAR(100) UNIQUE NOT NULL,
    criado_em       TIMESTAMP NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- 2. TALHOES — áreas florestais monitoradas
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS talhoes (
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
CREATE TABLE IF NOT EXISTS toras_inspecionadas (
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

CREATE INDEX IF NOT EXISTS idx_toras_talhao   ON toras_inspecionadas(talhao_id);
CREATE INDEX IF NOT EXISTS idx_toras_maquina  ON toras_inspecionadas(maquina_id);
CREATE INDEX IF NOT EXISTS idx_toras_data     ON toras_inspecionadas(data_inspecao);
CREATE INDEX IF NOT EXISTS idx_toras_status   ON toras_inspecionadas(status_classificacao);

-- ------------------------------------------------------------
-- 4. INDICADORES_QUALIDADE — os 4 indicadores medidos por tora
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS indicadores_qualidade (
    id                SERIAL PRIMARY KEY,
    tora_id           INTEGER NOT NULL REFERENCES toras_inspecionadas(id) ON DELETE CASCADE,
    tipo_indicador    VARCHAR(30) NOT NULL
                      CHECK (tipo_indicador IN ('densidade', 'massa_seca', 'altura', 'diametro', 'tortuosidade', 'porcentagem_casca', 'volume_util', 'apodrecimento_pragas')),
    valor             NUMERIC(10,4) NOT NULL,
    unidade           VARCHAR(20),             -- ex: "kg/m3", "m", "indice", "%"
    metodo_medicao    VARCHAR(50) NOT NULL,    -- ex: "fusao_sensores", "imagem_4k_ultrassom", "opencv_contorno", "yolo"
    criado_em         TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_indicadores_tora ON indicadores_qualidade(tora_id);
CREATE INDEX IF NOT EXISTS idx_indicadores_tipo ON indicadores_qualidade(tipo_indicador);

-- ------------------------------------------------------------
-- 5. DEFEITOS_DETECTADOS — bounding boxes das detecções do YOLO
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS defeitos_detectados (
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

CREATE INDEX IF NOT EXISTS idx_defeitos_tora ON defeitos_detectados(tora_id);

-- ------------------------------------------------------------
-- 6. USUARIOS — controle de acesso do dashboard (RBAC)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS usuarios (
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
CREATE TABLE IF NOT EXISTS logs_auditoria (
    id              SERIAL PRIMARY KEY,
    usuario_id      INTEGER REFERENCES usuarios(id),
    acao            VARCHAR(100) NOT NULL,     -- ex: "visualizou_dashboard", "editou_status"
    tabela_afetada  VARCHAR(50),
    registro_id     INTEGER,
    criado_em       TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_auditoria_usuario ON logs_auditoria(usuario_id);
CREATE INDEX IF NOT EXISTS idx_auditoria_data    ON logs_auditoria(criado_em);

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

-- ############################################################
-- PARTE 2 — SEED DE TESTE (máquina e talhão)
-- ############################################################
-- Sem isto, o sync_daemon.py roda mas não acha máquina/talhão
-- correspondente (o JOIN por numero_serie/nome não bate com nada)
-- e falha com "Nenhuma máquina encontrada".
--
-- Os valores batem com os padrões do config.json. Se mudarem o
-- maquina_id/talhao_id lá, cadastrem a máquina correspondente aqui.

INSERT INTO maquinas (modelo, numero_serie)
VALUES ('Harvester (maquete de testes)', 'RASPI-DEMO-01')
ON CONFLICT (numero_serie) DO NOTHING;

-- talhoes.nome NÃO tem constraint UNIQUE no schema, então "ON CONFLICT"
-- não protegeria nada aqui (não haveria conflito pra detectar). Usamos
-- WHERE NOT EXISTS em vez disso, pra ser idempotente de verdade --
-- rodar este arquivo duas vezes não duplica a linha.
INSERT INTO talhoes (nome, area_hectares, especie)
SELECT 'Talhão Demo', 12.5, 'Eucalyptus grandis x urophylla'
WHERE NOT EXISTS (SELECT 1 FROM talhoes WHERE nome = 'Talhão Demo');

-- ############################################################
-- PARTE 3 — TABELA DE DENSIDADE POR CLONE
-- ############################################################
-- Motivação (feedback do contato da John Deere na mentoria):
-- densidade não deve ser valor fixo em arquivo, e sim vir de um
-- cadastro da empresa -- porque a mesma ESPÉCIE (urograndis) tem
-- dezenas de CLONES diferentes, cada um com sua densidade.
--
-- Ele apontou que uma operação como a da Eldorado colhe ~6.000
-- toras/dia. Com o volume médio que nosso sistema calcula
-- (~0,147 m3/tora), variar de 480 para 510 kg/m3 dá ~26,5
-- TONELADAS de diferença POR DIA (~6.600 t/ano). Ele está certo:
-- a variação por clone importa em escala industrial.
--
-- ARQUITETURA:
--   1. Esta tabela é a FONTE DE VERDADE (o laboratório da empresa
--      atualiza aqui, sem mexer em código).
--   2. O sync_daemon.py baixa esta tabela quando há internet e
--      atualiza o cache local (data/clones_densidade.json).
--   3. O main.py lê o cache local -- então funciona offline com o
--      último dado sincronizado.

CREATE TABLE IF NOT EXISTS clones_densidade (
    id              SERIAL PRIMARY KEY,
    clone_id        VARCHAR(50) UNIQUE NOT NULL,   -- ex: 'I144', 'GG100', 'AEC 0144'
    especie         VARCHAR(100),                  -- ex: 'Urograndis (E. grandis x E. urophylla)'

    -- Densidade básica em kg/m3. NULL = ainda não cadastrado --
    -- intencionalmente permitido: é melhor o sistema avisar que
    -- não tem o dado do que preencher com um número inventado.
    densidade_base  NUMERIC(6,1),
    densidade_min   NUMERIC(6,1),                  -- faixa, quando a fonte dá faixa em vez de valor único
    densidade_max   NUMERIC(6,1),

    -- De onde veio o número. Isso é o que permite ser honesto na
    -- apresentação e no dashboard sobre a qualidade de cada dado:
    --   'laboratorio'          = laudo real da empresa (o ideal)
    --   'literatura'           = valor publicado para ESTE clone
    --   'referencia_generica'  = média do híbrido, não do clone específico
    tipo_dado       VARCHAR(30) NOT NULL DEFAULT 'referencia_generica',
    fonte           TEXT,

    atualizado_em   TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_clones_densidade_clone ON clones_densidade(clone_id);

-- ------------------------------------------------------------
-- SEED: os 11 clones informados pelo contato da John Deere
-- ------------------------------------------------------------
-- ATENÇÃO a duas coisas, ambas para não inventarmos dado:
--
-- 1. I144 e AEC 0144 aparecem como itens separados na lista, mas
--    fontes comerciais descrevem "Clone I144 (AEC-0144)" como o
--    MESMO material. Cadastramos os dois apontando para o mesmo
--    valor e sinalizamos isso em 'fonte' -- vale confirmar com ele.
--
-- 2. Só I144/AEC 0144 tem faixa de densidade publicada que
--    conseguimos verificar (460-510 kg/m3). Para os outros 9
--    clones NÃO encontramos valor específico publicado -- então
--    ficam com a referência genérica do híbrido (490) e
--    tipo_dado='referencia_generica', explicitamente marcados
--    como "aguardando dado de laboratório". Nada foi inventado.
-- ------------------------------------------------------------

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

-- ------------------------------------------------------------
-- Clones do inventário antigo (planilha de exemplo da JD).
-- Mantidos para o sistema continuar funcionando com os dados de
-- teste que já temos, mas marcados pelo que são.
-- ------------------------------------------------------------
INSERT INTO clones_densidade (clone_id, especie, densidade_base, tipo_dado, fonte) VALUES
    ('SP3108', 'Eucalyptus sp. (código interno)', 490.0, 'referencia_generica', 'Código interno da planilha de inventário. Sem correspondência pública. Referência de híbrido comercial E. grandis x urophylla.'),
    ('SP2974', 'Eucalyptus sp. (código interno)', 490.0, 'referencia_generica', 'Código interno da planilha de inventário. Sem correspondência pública.'),
    ('SP3153', 'Eucalyptus sp. (código interno)', 490.0, 'referencia_generica', 'Código interno da planilha de inventário. Sem correspondência pública.'),
    ('SP2887', 'Eucalyptus sp. (código interno)', 490.0, 'referencia_generica', 'Código interno da planilha de inventário. Sem correspondência pública.'),
    ('CO41H_TEST', 'Eucalyptus sp. (clone de teste)', 490.0, 'referencia_generica', 'Clone de teste.'),
    ('SP1048_TEST', 'Eucalyptus sp. (clone de teste)', 490.0, 'referencia_generica', 'Clone de teste.')
ON CONFLICT (clone_id) DO NOTHING;

-- ------------------------------------------------------------
-- Referência da faixa-alvo da indústria (Embrapa CT-316, citando
-- Fonseca et al. 2010), útil para o dashboard/apresentação:
--   celulose : 480-520 kg/m3
--   papéis   : acima de 520 kg/m3
-- ------------------------------------------------------------

-- ============================================================
-- REFERÊNCIA ÚTIL PARA O DASHBOARD / APRESENTAÇÃO
-- ============================================================
-- Faixa-alvo de densidade da indústria (Embrapa CT-316, citando
-- Fonseca et al. 2010):
--   celulose : 480-520 kg/m3
--   papéis   : acima de 520 kg/m3
--
-- O campo 'tipo_dado' de cada clone diz a procedência do número:
--   'laboratorio'         -> laudo real da empresa (o ideal)
--   'literatura'          -> valor publicado para ESTE clone
--   'referencia_generica' -> média do híbrido, aguardando dado do clone
-- ============================================================

-- ############################################################
-- PARTE 4 — AVISO EM TEMPO REAL PARA O DASHBOARD (LISTEN/NOTIFY)
-- ############################################################
-- Cada tora inserida pelo sync_daemon.py publica um NOTIFY no canal
-- 'omniroot_toras' (no COMMIT, quando indicadores e defeitos já estão
-- gravados). O dashboard fica em LISTEN e atualiza os painéis sem F5.
-- Mesmo conteúdo de migration_notify_toras.sql.

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
