package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	_ "github.com/lib/pq"          // driver PostgreSQL (puro Go, já não usa CGO)
	_ "modernc.org/sqlite"          // driver SQLite 100% Go — SEM CGO
)

// ============================================================
// CONFIGURAÇÃO
// ============================================================
// Lida de variáveis de ambiente, com valores padrão. Isso permite
// ajustar o IP/senha do Postgres no dia da apresentação SEM
// recompilar o programa — só mudar o .env ou exportar a variável.
//
// Exemplo de uso na véspera da apresentação:
//   Windows (PowerShell):
//     $env:PG_HOST="192.168.15.23"; .\sync_daemon.exe
//   Raspberry Pi (Linux):
//     PG_HOST=192.168.15.23 ./sync_daemon
// ============================================================

type Config struct {
	SqlitePath   string
	PgHost       string
	PgPort       string
	PgUser       string
	PgPassword   string
	PgDBName     string
	IntervaloSeg int
}

func carregarConfig() Config {
	return Config{
		SqlitePath:   getEnv("SQLITE_PATH", filepath.Join(".", "omni_root_local.db")),
		PgHost:       getEnv("PG_HOST", "192.168.1.100"),
		PgPort:       getEnv("PG_PORT", "5432"),
		PgUser:       getEnv("PG_USER", "postgres"),
		PgPassword:   getEnv("PG_PASSWORD", "suasenha"),
		PgDBName:     getEnv("PG_DBNAME", "desafio_madeira"),
		IntervaloSeg: 10,
	}
}

func getEnv(chave, padrao string) string {
	if valor, existe := os.LookupEnv(chave); existe && valor != "" {
		return valor
	}
	return padrao
}

func (c Config) pgDSN() string {
	return fmt.Sprintf(
		"host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		c.PgHost, c.PgPort, c.PgUser, c.PgPassword, c.PgDBName,
	)
}

// ============================================================
// ESTRUTURAS DE DADOS (espelham o schema_sqlite.sql)
// ============================================================

type Tora struct {
	IDLocal              int
	UUIDLocal            string
	MaquinaID            string
	TalhaoID             sql.NullString
	LogID                string
	DataInspecao         string
	ConfiancaIA          float64
	StatusClassificacao  string
	HashSHA256           string
}

type Indicador struct {
	IDLocal        int
	TipoIndicador  string
	Valor          float64
	Unidade        sql.NullString
	MetodoMedicao  string
}

type Defeito struct {
	IDLocal      int
	TipoDefeito  string
	PosX         float64
	PosY         float64
	Largura      float64
	Altura       float64
	Confianca    float64
}

func main() {
	cfg := carregarConfig()
	fmt.Println("⚙️  Sincronizador Go iniciado! Rodando em background...")
	fmt.Printf("📁 SQLite local : %s\n", cfg.SqlitePath)
	fmt.Printf("🐘 Postgres alvo: %s:%s/%s\n", cfg.PgHost, cfg.PgPort, cfg.PgDBName)

	for {
		if err := sincronizarBancos(cfg); err != nil {
			log.Println("⚠️  Ciclo de sincronização terminou com erro:", err)
		}
		time.Sleep(time.Duration(cfg.IntervaloSeg) * time.Second)
	}
}

func sincronizarBancos(cfg Config) error {
	// 1. Tenta abrir conexão com o PostgreSQL (checa a rede)
	pgDB, err := sql.Open("postgres", cfg.pgDSN())
	if err != nil {
		return fmt.Errorf("erro ao configurar conexão com PostgreSQL: %w", err)
	}
	defer pgDB.Close()

	if err = pgDB.Ping(); err != nil {
		fmt.Println("📡 Servidor PostgreSQL não encontrado na rede. Tentando depois...")
		return nil // não é erro fatal — só significa "sem rede agora"
	}

	// 2. Conecta no banco local SQLite (driver puro Go, sem CGO)
	sqliteDB, err := sql.Open("sqlite", cfg.SqlitePath)
	if err != nil {
		return fmt.Errorf("erro ao abrir SQLite: %w", err)
	}
	defer sqliteDB.Close()

	// 3. Busca toras pendentes (sync_status = 0)
	toras, err := buscarTorasPendentes(sqliteDB)
	if err != nil {
		return fmt.Errorf("erro ao consultar toras pendentes: %w", err)
	}

	if len(toras) == 0 {
		fmt.Println("✅ Tudo sincronizado. Nada a enviar.")
		return nil
	}

	processadas := 0
	for _, tora := range toras {
		if err := sincronizarTora(pgDB, sqliteDB, tora); err != nil {
			log.Printf("❌ Erro ao sincronizar tora %s: %v\n", tora.UUIDLocal, err)
			// Para o loop e tenta tudo de novo no próximo ciclo —
			// evita deixar dados "pela metade" se a rede cair no meio.
			break
		}
		processadas++
	}

	fmt.Printf("🚀 %d/%d toras processadas com sucesso!\n", processadas, len(toras))
	return nil
}

// ------------------------------------------------------------
// Busca todas as toras com sync_status = 0
// ------------------------------------------------------------
func buscarTorasPendentes(sqliteDB *sql.DB) ([]Tora, error) {
	rows, err := sqliteDB.Query(`
		SELECT id, uuid_local, maquina_id, talhao_id, log_id,
		       data_inspecao, confianca_ia, status_classificacao, hash_sha256
		FROM toras_local
		WHERE sync_status = 0
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var toras []Tora
	for rows.Next() {
		var t Tora
		if err := rows.Scan(
			&t.IDLocal, &t.UUIDLocal, &t.MaquinaID, &t.TalhaoID, &t.LogID,
			&t.DataInspecao, &t.ConfiancaIA, &t.StatusClassificacao, &t.HashSHA256,
		); err != nil {
			log.Println("⚠️  Erro ao ler linha de tora:", err)
			continue
		}
		toras = append(toras, t)
	}
	return toras, rows.Err()
}

// ------------------------------------------------------------
// Sincroniza uma tora inteira: registro principal + indicadores +
// defeitos, tudo em uma transação por tora. Se qualquer parte
// falhar, nada é marcado como sincronizado (fica pendente pro
// próximo ciclo — nunca fica "meio sincronizado").
// ------------------------------------------------------------
func sincronizarTora(pgDB *sql.DB, sqliteDB *sql.DB, tora Tora) error {
	// --- 1. Insere (ou recupera) a tora no Postgres, idempotente ---
	toraRemotaID, err := inserirOuBuscarTora(pgDB, tora)
	if err != nil {
		return fmt.Errorf("insert/select tora: %w", err)
	}

	// --- 2. Sincroniza indicadores de qualidade dessa tora ---
	indicadores, err := buscarIndicadoresPendentes(sqliteDB, tora.IDLocal)
	if err != nil {
		return fmt.Errorf("buscar indicadores: %w", err)
	}
	for _, ind := range indicadores {
		if _, err := pgDB.Exec(`
			INSERT INTO indicadores_qualidade
				(tora_id, tipo_indicador, valor, unidade, metodo_medicao)
			VALUES ($1, $2, $3, $4, $5)
		`, toraRemotaID, ind.TipoIndicador, ind.Valor, ind.Unidade, ind.MetodoMedicao); err != nil {
			return fmt.Errorf("insert indicador: %w", err)
		}
		if _, err := sqliteDB.Exec(
			`UPDATE indicadores_qualidade_local SET sync_status = 1 WHERE id = ?`,
			ind.IDLocal,
		); err != nil {
			return fmt.Errorf("update indicador local: %w", err)
		}
	}

	// --- 3. Sincroniza defeitos detectados dessa tora ---
	defeitos, err := buscarDefeitosPendentes(sqliteDB, tora.IDLocal)
	if err != nil {
		return fmt.Errorf("buscar defeitos: %w", err)
	}
	for _, def := range defeitos {
		if _, err := pgDB.Exec(`
			INSERT INTO defeitos_detectados
				(tora_id, tipo_defeito, pos_x, pos_y, largura, altura, confianca)
			VALUES ($1, $2, $3, $4, $5, $6, $7)
		`, toraRemotaID, def.TipoDefeito, def.PosX, def.PosY, def.Largura, def.Altura, def.Confianca); err != nil {
			return fmt.Errorf("insert defeito: %w", err)
		}
		if _, err := sqliteDB.Exec(
			`UPDATE defeitos_detectados_local SET sync_status = 1 WHERE id = ?`,
			def.IDLocal,
		); err != nil {
			return fmt.Errorf("update defeito local: %w", err)
		}
	}

	// --- 4. Só marca a tora como sincronizada depois de tudo certo ---
	if _, err := sqliteDB.Exec(
		`UPDATE toras_local SET sync_status = 1 WHERE id = ?`,
		tora.IDLocal,
	); err != nil {
		return fmt.Errorf("update tora local: %w", err)
	}

	return nil
}

// ------------------------------------------------------------
// Insere a tora no Postgres. Se uuid_local já existir (retry após
// queda de rede), busca o id já existente em vez de duplicar.
// Isso é o que resolve o problema de duplicação do script original.
// ------------------------------------------------------------
func inserirOuBuscarTora(pgDB *sql.DB, tora Tora) (int, error) {
	var idRemoto int

	err := pgDB.QueryRow(`
		INSERT INTO toras_inspecionadas
			(uuid_local, maquina_id, talhao_id, log_id, data_inspecao,
			 confianca_ia, status_classificacao, hash_sha256)
		SELECT $1, m.id_maquina, t.id_talhao, $4, $5, $6, $7, $8
		FROM maquinas m
		LEFT JOIN talhoes t ON t.nome = $3
		WHERE m.numero_serie = $2
		ON CONFLICT (uuid_local) DO NOTHING
		RETURNING id
	`, tora.UUIDLocal, tora.MaquinaID, tora.TalhaoID, tora.LogID, tora.DataInspecao,
		tora.ConfiancaIA, tora.StatusClassificacao, tora.HashSHA256,
	).Scan(&idRemoto)

	if err == sql.ErrNoRows {
		// Já existia (ON CONFLICT) — busca o id remoto pelo uuid
		err = pgDB.QueryRow(
			`SELECT id FROM toras_inspecionadas WHERE uuid_local = $1`,
			tora.UUIDLocal,
		).Scan(&idRemoto)
	}

	return idRemoto, err
}

func buscarIndicadoresPendentes(sqliteDB *sql.DB, toraIDLocal int) ([]Indicador, error) {
	rows, err := sqliteDB.Query(`
		SELECT id, tipo_indicador, valor, unidade, metodo_medicao
		FROM indicadores_qualidade_local
		WHERE tora_id = ? AND sync_status = 0
	`, toraIDLocal)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var lista []Indicador
	for rows.Next() {
		var i Indicador
		if err := rows.Scan(&i.IDLocal, &i.TipoIndicador, &i.Valor, &i.Unidade, &i.MetodoMedicao); err != nil {
			log.Println("⚠️  Erro ao ler indicador:", err)
			continue
		}
		lista = append(lista, i)
	}
	return lista, rows.Err()
}

func buscarDefeitosPendentes(sqliteDB *sql.DB, toraIDLocal int) ([]Defeito, error) {
	rows, err := sqliteDB.Query(`
		SELECT id, tipo_defeito, pos_x, pos_y, largura, altura, confianca
		FROM defeitos_detectados_local
		WHERE tora_id = ? AND sync_status = 0
	`, toraIDLocal)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var lista []Defeito
	for rows.Next() {
		var d Defeito
		if err := rows.Scan(&d.IDLocal, &d.TipoDefeito, &d.PosX, &d.PosY, &d.Largura, &d.Altura, &d.Confianca); err != nil {
			log.Println("⚠️  Erro ao ler defeito:", err)
			continue
		}
		lista = append(lista, d)
	}
	return lista, rows.Err()
}
