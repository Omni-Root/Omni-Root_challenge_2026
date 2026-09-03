-- seed_teste.sql — dado mínimo pra testar o sync_daemon.py localmente
--
-- Sem isso, a sincronização roda sem erro mas não encontra nenhuma
-- máquina/talhão correspondente (o JOIN por numero_serie/nome não bate
-- com nada), e a tora nunca é inserida em toras_inspecionadas -- foi
-- exatamente o erro "Nenhuma máquina encontrada" que apareceu antes.
--
-- Os valores abaixo batem com os padrões do config.json (maquina_id e
-- talhao_id). Se mudarem esses valores no config.json, atualizem aqui
-- também (ou insiram uma linha extra pra cada máquina/talhão real).

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
