-- ============================================================
-- MIGRATION: aviso em tempo real de tora nova (LISTEN/NOTIFY)
-- ============================================================
-- Toda vez que o sync_daemon.py insere uma tora em toras_inspecionadas,
-- o Postgres publica um aviso no canal 'omniroot_toras'. O dashboard
-- (repositório omni-root-dashboard, server/live.ts) fica em LISTEN nesse
-- canal e empurra a atualização para os navegadores por Server-Sent
-- Events — sem F5, sem polling pesado.
--
-- O NOTIFY só é entregue no COMMIT da transação. Como o sync_daemon
-- insere tora + indicadores + defeitos numa transação só, quando o
-- dashboard é avisado a tora já está completa no banco.
--
-- Idempotente: pode rodar mais de uma vez. Num banco já existente:
--   docker compose exec -T postgres psql -U postgres -d desafio_madeira < "Banco de dados/migration_notify_toras.sql"
-- (o setup_completo.sql já inclui isto para bancos criados do zero)
-- ============================================================

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
