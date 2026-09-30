"""
testar_posicao.py — posição GNSS por tora (sem receptor, sem câmera)

Verifica:
  1. checksum NMEA: aceita o certo, recusa o errado/ausente
  2. GGA e RMC viram graus decimais corretos (S/W negativos), com HDOP/satélites
  3. "sem fix" do receptor apaga a posição atual; posição velha expira
  4. com GGA presente, RMC não sobrescreve (GGA tem HDOP/satélites)
  5. log gravado (.nmea) é reproduzido e marcado como fonte 'gnss_log';
     arquivo sem sentença válida vira "sem receptor", sem derrubar nada
  6. SQLite: banco ANTIGO (sem colunas de posição) é migrado sem perder dados;
     tora gravada com posição guarda os campos; a atualização incremental
     mantém a posição e o hash confere; sem posição, o hash é o de antes

USO (raiz do repositório):
    python tests/testar_posicao.py
"""

import sqlite3
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from omniroot.posicao import FONTE_LOG, FONTE_WINDOWS, LeitorPosicao, checksum_ok, ler_sentenca  # noqa: E402
from omniroot.banco_local import migrar_banco_local  # noqa: E402
from omniroot.classificacao import gerar_hash_sha256  # noqa: E402
from omniroot.config import Config  # noqa: E402
from omniroot.registro import _hash_inspecao, atualizar_inspecao, salvar_inspecao  # noqa: E402


def nmea(corpo: str) -> str:
    """Monta a sentença com o checksum certo (o XOR que o receptor calcularia)."""
    cs = 0
    for c in corpo:
        cs ^= ord(c)
    return f"${corpo}*{cs:02X}"


# FIAP Paulista, aproximadamente: 23°33,846' S  46°39,144' W
LAT, LON = -(23 + 33.846 / 60), -(46 + 39.144 / 60)
GGA_OK = nmea("GPGGA,123519.00,2333.8460,S,04639.1440,W,1,09,0.9,760.0,M,-5.0,M,,")
GGA_SEM_FIX = nmea("GPGGA,123520.00,,,,,0,00,99.9,,M,,M,,")
RMC_OK = nmea("GNRMC,123519.00,A,2333.8460,S,04639.1440,W,0.0,0.0,290926,,,A")
RMC_SEM_FIX = nmea("GNRMC,123519.00,V,,,,,,,290926,,,N")
RMC_OUTRO_LUGAR = nmea("GNRMC,123521.00,A,2200.0000,S,04700.0000,W,0.0,0.0,290926,,,A")

INDICADORES = {
    "diametro": {"valor": 20.0, "unidade": "cm", "metodo": "imagem_lateral_marcador_aruco"},
    "porcentagem_casca": {"valor": 12.5, "unidade": "%", "metodo": "opencv_otsu_casca_residual"},
}

# toras_local como era ANTES das colunas de posição (para testar a migração)
TORAS_LOCAL_ANTIGA = """
CREATE TABLE toras_local (
    id INTEGER PRIMARY KEY AUTOINCREMENT, uuid_local TEXT UNIQUE NOT NULL, maquina_id TEXT NOT NULL,
    talhao_id TEXT, log_id TEXT NOT NULL, data_inspecao TEXT NOT NULL, confianca_ia REAL NOT NULL,
    status_classificacao TEXT NOT NULL, hash_sha256 TEXT NOT NULL,
    sync_status INTEGER NOT NULL DEFAULT 0, criado_em TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE indicadores_qualidade_local (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tora_id INTEGER NOT NULL, tipo_indicador TEXT NOT NULL,
    valor REAL NOT NULL, unidade TEXT, metodo_medicao TEXT NOT NULL,
    sync_status INTEGER NOT NULL DEFAULT 0, criado_em TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE defeitos_detectados_local (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tora_id INTEGER NOT NULL, tipo_defeito TEXT NOT NULL,
    pos_x REAL NOT NULL, pos_y REAL NOT NULL, largura REAL NOT NULL, altura REAL NOT NULL,
    confianca REAL NOT NULL, sync_status INTEGER NOT NULL DEFAULT 0,
    criado_em TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def main() -> int:
    falhas = 0

    def checar(cond: bool, msg: str) -> None:
        nonlocal falhas
        print(("  ✅ " if cond else "  ❌ ") + msg)
        if not cond:
            falhas += 1

    print("1) checksum NMEA")
    checar(checksum_ok(GGA_OK), "sentença íntegra aceita")
    corrompida = GGA_OK.replace("2333.8460", "2333.8461")
    checar(not checksum_ok(corrompida), "um dígito trocado na linha -> recusada")
    checar(ler_sentenca(corrompida) is None, "e não vira coordenada")
    checar(ler_sentenca(GGA_OK.split("*")[0]) is None, "sem checksum -> recusada")
    checar(ler_sentenca(nmea("GPGSV,3,1,11,03,03,111,00")) is None, "sentença que não é GGA/RMC -> ignorada")

    print("2) GGA e RMC -> graus decimais")
    g = ler_sentenca(GGA_OK)
    checar(g is not None and g["valida"], "GGA com fix é válida")
    checar(g is not None and abs(g["lat"] - LAT) < 1e-9 and abs(g["lon"] - LON) < 1e-9,
           f"lat/lon = {LAT:.6f}, {LON:.6f} (S e W negativos)")
    checar(g is not None and g["hdop"] == 0.9 and g["satelites"] == 9, "HDOP 0.9 e 9 satélites do próprio receptor")
    r = ler_sentenca(RMC_OK)
    checar(r is not None and r["valida"] and abs(r["lat"] - LAT) < 1e-9, "RMC (talker GN) com status A é válida")
    checar(not ler_sentenca(GGA_SEM_FIX)["valida"], "GGA com qualidade 0 = sem fix")
    checar(not ler_sentenca(RMC_SEM_FIX)["valida"], "RMC com status V = sem fix")

    print("3) sem fix e posição velha")
    leitor = LeitorPosicao("COM_INEXISTENTE_TESTE")  # não iniciado: só o estado é testado
    checar(leitor.atual(5.0) is None, "antes de qualquer leitura: sem posição")
    leitor.processar_linha(GGA_OK)
    p = leitor.atual(5.0)
    checar(p is not None and p["fonte"] == "gnss_serial" and p["idade_s"] <= 0.1, "fix recebido: posição atual, fonte gnss_serial")
    leitor.processar_linha(GGA_SEM_FIX)
    checar(leitor.atual(5.0) is None, "receptor perdeu o fix: posição some na hora")
    leitor.processar_linha(GGA_OK)
    leitor._recebida_em -= 10.0  # simula 10 s sem leitura nova
    checar(leitor.atual(5.0) is None, "leitura com 10 s e validade de 5 s: expirada")
    checar(leitor.atual(15.0) is not None, "...mas dentro de uma validade de 15 s continua valendo")

    print("4) GGA manda; RMC não sobrescreve")
    leitor = LeitorPosicao("COM_INEXISTENTE_TESTE")
    leitor.processar_linha(RMC_OK)
    checar(leitor.atual(5.0) is not None and leitor.atual(5.0)["hdop"] is None, "só RMC: usa RMC (sem HDOP)")
    leitor.processar_linha(GGA_OK)
    leitor.processar_linha(RMC_OUTRO_LUGAR)
    p = leitor.atual(5.0)
    checar(p is not None and abs(p["lat"] - LAT) < 1e-6 and p["hdop"] == 0.9, "depois de ver GGA, RMC é ignorada")

    print("5) log gravado")
    with tempfile.TemporaryDirectory() as tmp:
        trilha = Path(tmp) / "trilha.nmea"
        trilha.write_text("\n".join([GGA_OK, RMC_OK, nmea("GPGGA,123520.00,2333.8500,S,04639.1500,W,1,08,1.1,760.0,M,-5.0,M,,")]) + "\n", encoding="ascii")
        leitor = LeitorPosicao(str(trilha)).iniciar()
        fim = time.monotonic() + 3.0
        while leitor.atual(5.0) is None and time.monotonic() < fim:
            time.sleep(0.05)
        p = leitor.atual(5.0)
        checar(p is not None and p["fonte"] == FONTE_LOG, "trilha reproduzida, fonte declarada 'gnss_log'")
        checar("log gravado" in leitor.descricao(5.0), f"HUD declara o log: {leitor.descricao(5.0)!r}")
        leitor.parar()

        lixo = Path(tmp) / "lixo.nmea"
        lixo.write_text("isto nao e NMEA\n$GPGGA,sem,checksum\n", encoding="ascii")
        leitor = LeitorPosicao(str(lixo)).iniciar()
        time.sleep(0.3)
        checar(leitor.atual(5.0) is None and "sem receptor" in leitor.descricao(5.0),
               "arquivo sem sentença válida: 'sem receptor', thread segue viva")
        leitor.parar()

    print("5b) Localização do Windows (linhas do script, sem chamar o Windows)")
    leitor = LeitorPosicao("windows")
    checar(leitor.fonte == FONTE_WINDOWS and not leitor.e_arquivo, "'windows' liga a fonte windows_localizacao")
    leitor.processar_windows("SEM_FIX;Initializing")
    checar(leitor.atual(5.0) is None and "aguardando Localizacao" in leitor.descricao(5.0), "sem posição ainda: aguardando")
    leitor.processar_windows(f"POS;{LAT!r};{LON!r};83")
    p = leitor.atual(5.0)
    checar(p is not None and abs(p["lat"] - LAT) < 1e-6 and p["precisao_m"] == 83.0 and p["hdop"] is None,
           "POS vira posição com a precisão do Windows (83 m), sem HDOP inventado")
    checar("+-83 m" in leitor.descricao(5.0) and "Windows" in leitor.descricao(5.0), f"HUD: {leitor.descricao(5.0)!r}")
    leitor.processar_windows("POS;-23,5;-46,6;NaN")
    checar(leitor.atual(5.0)["precisao_m"] == 83.0, "número com vírgula (idioma) é ignorado, posição anterior mantida")
    leitor.processar_windows(f"POS;{LAT!r};{LON!r};NaN")
    checar(leitor.atual(5.0)["precisao_m"] is None, "precisão NaN (Windows não informou) vira vazio")
    leitor.processar_windows("NEGADO")
    checar(leitor.atual(5.0) is None and "indisponivel" in leitor.descricao(5.0), "permissão negada: sem posição, HUD avisa")

    print("6) SQLite: migração, gravação e atualização")
    cfg = Config()
    posicao = {"lat": round(LAT, 7), "lon": round(LON, 7), "hdop": 0.9, "satelites": 9, "fonte": "gnss_serial", "idade_s": 0.4, "precisao_m": None}
    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "antigo.db"
        con = sqlite3.connect(caminho)
        con.executescript(TORAS_LOCAL_ANTIGA)
        con.execute(
            "INSERT INTO toras_local (uuid_local, maquina_id, talhao_id, log_id, data_inspecao, confianca_ia, status_classificacao, hash_sha256) "
            "VALUES ('antiga', 'M1', 'T1', 'LOG-1', '2026-09-01T10:00:00', 0.9, 'aprovado', 'h')"
        )
        con.commit()
        migrar_banco_local(con)
        colunas = {c[1] for c in con.execute("PRAGMA table_info(toras_local)")}
        checar({"pos_lat", "pos_lon", "pos_hdop", "pos_satelites", "pos_fonte", "pos_idade_s", "pos_precisao_m"} <= colunas,
               "banco antigo ganhou as 7 colunas de posição")
        antiga = con.execute("SELECT log_id, pos_lat FROM toras_local WHERE uuid_local = 'antiga'").fetchone()
        checar(antiga == ("LOG-1", None), "tora antiga preservada, sem posição")
        migrar_banco_local(con)
        checar(True, "rodar a migração de novo não quebra (idempotente)")

        u = salvar_inspecao(con, cfg, INDICADORES, [], "aprovado", 0.95, posicao=posicao)
        linha = con.execute(
            "SELECT pos_lat, pos_lon, pos_hdop, pos_satelites, pos_fonte, pos_idade_s, log_id, hash_sha256 FROM toras_local WHERE uuid_local = ?", (u,)
        ).fetchone()
        checar(linha[:6] == (posicao["lat"], posicao["lon"], 0.9, 9, "gnss_serial", 0.4), "tora nova gravada com os campos de posição")
        uw = salvar_inspecao(con, cfg, INDICADORES, [], "aprovado", 0.95,
                             posicao={**posicao, "hdop": None, "satelites": None, "fonte": FONTE_WINDOWS, "precisao_m": 83.0})
        lw = con.execute("SELECT pos_fonte, pos_precisao_m, pos_hdop FROM toras_local WHERE uuid_local = ?", (uw,)).fetchone()
        checar(lw == (FONTE_WINDOWS, 83.0, None), "tora com Localização do Windows guarda a precisão em metros")
        checar(linha[7] == _hash_inspecao(u, linha[6], INDICADORES, [], "aprovado", posicao), "hash cobre a posição")

        novos = {**INDICADORES, "porcentagem_casca": {"valor": 14.0, "unidade": "%", "metodo": "opencv_otsu_casca_residual"}}
        checar(atualizar_inspecao(con, u, novos, [], "quarentena", 0.7), "atualização incremental encontrou a tora")
        linha2 = con.execute("SELECT pos_lat, pos_lon, pos_fonte, hash_sha256 FROM toras_local WHERE uuid_local = ?", (u,)).fetchone()
        checar(linha2[:3] == (posicao["lat"], posicao["lon"], "gnss_serial"), "atualização NÃO muda a posição (é a do corte)")
        checar(linha2[3] == _hash_inspecao(u, linha[6], novos, [], "quarentena", posicao), "hash da atualização confere, com a posição")

        u2 = salvar_inspecao(con, cfg, INDICADORES, [], "aprovado", 0.95)
        l3 = con.execute("SELECT pos_lat, pos_fonte, log_id, hash_sha256 FROM toras_local WHERE uuid_local = ?", (u2,)).fetchone()
        checar(l3[0] is None and l3[1] is None, "sem GNSS: tora gravada normalmente, posição vazia")
        hash_antigo = gerar_hash_sha256({"uuid_local": u2, "log_id": l3[2], "indicadores": INDICADORES, "defeitos": [], "status": "aprovado"})
        checar(l3[3] == hash_antigo, "sem posição, o hash é exatamente o do formato anterior")
        con.close()

    print()
    print("TUDO OK" if falhas == 0 else f"{falhas} FALHA(S)")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
