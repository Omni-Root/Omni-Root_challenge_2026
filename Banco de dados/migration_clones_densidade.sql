-- ============================================================
-- MIGRATION: tabela de referência de densidade por clone
-- ============================================================
-- Motivação (feedback do contato da John Deere na mentoria):
-- densidade não deve ser um valor fixo num arquivo, e sim vir de
-- um cadastro da empresa -- porque a mesma ESPÉCIE (urograndis)
-- tem dezenas de CLONES diferentes, cada um com sua densidade.
--
-- Ele apontou que uma operação como a da Eldorado colhe ~6.000
-- toras/dia. Fazendo a conta com o volume médio que nosso próprio
-- sistema calcula (~0,147 m3/tora), uma variação de 480 para
-- 510 kg/m3 representa ~26,5 TONELADAS de diferença POR DIA
-- (~6.600 t/ano). Ou seja: ele está certo, a variação por clone
-- importa em escala industrial.
--
-- COMO ISSO FUNCIONA NA ARQUITETURA:
--   1. Esta tabela no Postgres central é a FONTE DE VERDADE.
--      O laboratório da empresa atualiza aqui, sem mexer em código.
--   2. O sync_daemon.py, quando há internet, BAIXA esta tabela e
--      atualiza o cache local (data/clones_densidade.json).
--   3. O main.py continua lendo o JSON local -- então funciona
--      offline normalmente, usando o último dado sincronizado.
--
-- Rode este arquivo uma vez no banco já existente:
--   docker compose exec -T postgres psql -U postgres -d desafio_madeira < "Banco de dados/migration_clones_densidade.sql"
-- ============================================================

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
