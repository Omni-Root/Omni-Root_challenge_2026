"""
testar_equivalencia.py — "o sistema continua fazendo exatamente a mesma coisa?"

Teste de regressão (golden master). Passa cenas FIXAS e sintéticas por todo
o pipeline da máquina — segmentação, portão "é madeira?", indicadores,
classificação, evento de tora, luz, posição, gravação no SQLite e o HUD
desenhado — e compara o resultado, campo a campo, com uma referência
gravada em tests/dados/equivalencia.json.

Serve para refatorar com segurança: mover código de lugar não pode mudar
nenhum número, nenhum texto do HUD, nenhum pixel desenhado. Se mudar, o
teste aponta exatamente onde.

Sem câmera e sem o modelo YOLO: os defeitos "do modelo" entram fixos
(defeitos_yolo=...), então o resultado é determinístico.

USO (raiz do repositório):
    python tests/testar_equivalencia.py            # compara com a referência
    python tests/testar_equivalencia.py --gerar    # grava uma referência NOVA
                                                   # (só quando a mudança de
                                                   # comportamento for intencional)
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from main import (  # noqa: E402
    Config,
    FiltroPersistencia,
    RastreadorTora,
    _hash_inspecao,
    analisar_frame,
    atualizar_inspecao,
    calcular_confianca_saude,
    calcular_densidade_estimada,
    carregar_configuracao_json,
    classificar_qualidade,
    consolidar_evento,
    desenhar_analise,
    migrar_banco_local,
    resumir_analise,
    salvar_inspecao,
    severidade_do_defeito,
)
from omniroot.luz import RealceLuz, descrever_luz  # noqa: E402
from omniroot.posicao import LeitorPosicao, ler_sentenca  # noqa: E402

REFERENCIA = RAIZ / "tests" / "dados" / "equivalencia.json"


# ------------------------------------------------------------
# Cenas sintéticas (determinísticas)
# ------------------------------------------------------------

def _textura(h: int, w: int, seed: int, sigma: float = 14.0) -> np.ndarray:
    ruido = np.random.default_rng(seed).normal(0, sigma, (h, w)).astype(np.float32)
    return cv2.GaussianBlur(ruido, (0, 0), 3)


def _pinta(img: np.ndarray, mascara: np.ndarray, cor_bgr, seed: int) -> None:
    t = _textura(*mascara.shape, seed)
    img[mascara > 0] = np.clip(np.array(cor_bgr, np.float32) + t[mascara > 0][:, None], 0, 255)


def cenas() -> dict[str, np.ndarray]:
    h, w = 480, 640
    madeira = (118, 168, 204)
    out: dict[str, np.ndarray] = {}

    # Tora deitada (vista lateral) sobre mesa preta
    img = np.full((h, w, 3), 16.0, np.float32)
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (320, 240), (230, 70), 4, 0, 360, 255, -1)
    _pinta(img, m, madeira, 1)
    out["lateral_mesa_preta"] = img.astype(np.uint8)

    # Mesma tora com manchas de casca (escuras)
    casca = out["lateral_mesa_preta"].copy()
    for cx, cy, r in ((230, 225, 28), (360, 250, 35), (440, 235, 22)):
        mc = np.zeros((h, w), np.uint8)
        cv2.circle(mc, (cx, cy), r, 255, -1)
        mc &= m
        casca[mc > 0] = (38, 52, 70)
    out["lateral_com_casca"] = casca

    # Seção (rodela) sobre mesa preta, com anéis
    img = np.full((h, w, 3), 16.0, np.float32)
    m2 = np.zeros((h, w), np.uint8)
    cv2.circle(m2, (320, 240), 150, 255, -1)
    _pinta(img, m2, (120, 172, 210), 2)
    for r in (40, 80, 120):
        cv2.circle(img, (320, 240), r, (95, 140, 178), 2)
    out["secao_mesa_preta"] = img.astype(np.uint8)

    # Tora torta (eixo curvo) sobre mesa preta
    img = np.full((h, w, 3), 16.0, np.float32)
    m3 = np.zeros((h, w), np.uint8)
    pts = np.array([[90, 300], [200, 230], [320, 205], [440, 230], [550, 300]], np.int32)
    cv2.polylines(m3, [pts], False, 255, 90)
    _pinta(img, m3, madeira, 3)
    out["tora_torta"] = img.astype(np.uint8)

    # Tora sobre papel branco (bancada da FIAP)
    img = np.full((h, w, 3), (228, 232, 235), np.float32)
    _pinta(img, m, madeira, 4)
    out["lateral_fundo_claro"] = img.astype(np.uint8)

    # "Mão" (matiz de pele) sobre mesa preta: o portão tem de recusar
    img = np.full((h, w, 3), 16.0, np.float32)
    m4 = np.zeros((h, w), np.uint8)
    cv2.ellipse(m4, (320, 240), (110, 150), 0, 0, 360, 255, -1)
    _pinta(img, m4, (120, 150, 225), 5)
    out["mao"] = img.astype(np.uint8)

    # Garra vazia
    out["vazio"] = np.full((h, w, 3), 16, np.uint8)
    return out


# Como o modelo entrega (extrair_defeitos_yolo): caixa em px, confiança e a
# área relativa ao quadro (usada pelo filtro de área mínima). Um dentro da
# tora e grave, um leve, um FORA da madeira (tem de ser descartado) e um fino.
DEFEITOS_FIXOS = [
    {"tipo_defeito": t, "pos_x": x, "pos_y": y, "largura": w, "altura": h, "confianca": c,
     "area_relativa": round(w * h / (640 * 480), 6)}
    for t, x, y, w, h, c in (
        ("Dead_Knot", 250.0, 215.0, 40.0, 36.0, 0.82),
        ("Live_Knot", 400.0, 225.0, 34.0, 34.0, 0.75),
        ("resin", 20.0, 20.0, 25.0, 25.0, 0.91),
        ("Crack", 300.0, 234.0, 60.0, 14.0, 0.71),
    )
]


# ------------------------------------------------------------
# Serialização estável
# ------------------------------------------------------------

def _limpo(v):
    """Converte para algo comparável em JSON: floats arredondados, numpy -> python."""
    if isinstance(v, dict):
        return {str(k): _limpo(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_limpo(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return round(float(v), 5)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _md5(img: np.ndarray | None) -> str | None:
    return None if img is None else hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def _analise(a: dict) -> dict:
    contorno = a.get("contorno")
    resumo_contorno = None
    if contorno is not None and len(contorno) >= 3:
        c = np.asarray(contorno, np.int32).reshape(-1, 1, 2)
        resumo_contorno = {"bbox": list(cv2.boundingRect(c)), "pontos": int(len(c)), "area": float(cv2.contourArea(c))}
    return _limpo({
        "status": a["status"],
        "segmentacao": a.get("segmentacao"),
        "eh_tora": a.get("eh_tora"),
        "motivo_sem_tora": a.get("motivo_sem_tora"),
        "confianca_saude": a["confianca_saude"],
        "defeitos": a["defeitos"],
        "descartados": a["descartados"],
        "indicadores": a["indicadores"],
        "roi": a["roi"],
        "vista": a["vista"],
        "marcador": a.get("marcador") is not None,
        "contorno": resumo_contorno,
        "hud": a["hud"],
    })


def _evento(e: dict) -> dict:
    return _limpo({k: v for k, v in e.items() if k not in ("duracao_s", "uuid_local")})


# ------------------------------------------------------------
# Execução
# ------------------------------------------------------------

def executar() -> dict:
    cfg = Config()
    r: dict = {}
    cs = cenas()

    # 1. Um quadro de cada cena, com e sem defeitos fixos do "modelo", e o HUD desenhado
    r["quadros"] = {}
    for nome, img in cs.items():
        for com_defeitos in (False, True):
            a = analisar_frame(img, None, cfg, FiltroPersistencia(1),
                               defeitos_yolo=[dict(d) for d in DEFEITOS_FIXOS] if com_defeitos else [],
                               yolo_novo=True, fundo=None)
            chave = f"{nome}{'+defeitos' if com_defeitos else ''}"
            r["quadros"][chave] = _analise(a)
            r["quadros"][chave]["hud_md5"] = _md5(desenhar_analise(img, a, None))
            if a.get("eh_tora", True) and a["status"] != "sem_tora":
                r["quadros"][chave]["resumo"] = resumir_analise("00000000-teste", cfg, a).replace(" ", " ")

    # 2. Segmentação por fundo de referência (tecla b)
    a = analisar_frame(cs["lateral_fundo_claro"], None, cfg, FiltroPersistencia(1), defeitos_yolo=[], yolo_novo=True,
                       fundo=np.full_like(cs["lateral_fundo_claro"], (228, 232, 235)))
    r["com_fundo_de_referencia"] = _analise(a)

    # 3. Filtro de persistência ao longo de quadros
    persist = FiltroPersistencia(max(1, cfg.filtro_persistencia))
    seq = []
    for i in range(4):
        a = analisar_frame(cs["lateral_mesa_preta"], None, cfg, persist,
                           defeitos_yolo=[dict(d) for d in DEFEITOS_FIXOS[: 1 + i % 3]], yolo_novo=True)
        seq.append([d["tipo_defeito"] for d in a["defeitos"]])
    r["persistencia"] = seq

    # 4. Evento de tora: tora parada, troca de tora, garra vazia
    rast = RastreadorTora(cfg)
    fechados = []
    roteiro = ["lateral_mesa_preta"] * 8 + ["vazio"] * 7 + ["secao_mesa_preta"] * 6 + ["tora_torta"] * 6 + ["vazio"] * 7
    for i, nome in enumerate(roteiro):
        a = analisar_frame(cs[nome], None, cfg, FiltroPersistencia(1),
                           defeitos_yolo=[dict(d) for d in DEFEITOS_FIXOS] if i % 2 == 0 else [], yolo_novo=True)
        a["luz"] = "baixa" if nome == "tora_torta" else "boa"
        e = rast.atualizar(a)
        if e is not None:
            fechados.append(_evento(e))
    r["eventos"] = fechados
    r["rastreador_descricao"] = rast.descricao().split(" | ")[0]

    # 5. Classificação e regras de negócio
    exemplos = [[], DEFEITOS_FIXOS[:1], DEFEITOS_FIXOS[1:2], DEFEITOS_FIXOS]
    r["classificacao"] = [
        {"saude": calcular_confianca_saude(d), "status": classificar_qualidade(calcular_confianca_saude(d), d, cfg)}
        for d in exemplos
    ]
    r["severidade"] = {t: severidade_do_defeito(t) for t in ("Dead_Knot", "Crack", "Live_Knot", "desconhecido")}
    r["densidade"] = {c: calcular_densidade_estimada(c) for c in ("SP3108", "I144", "GG100", "NAO_EXISTE")}
    r["config_repo"] = asdict(carregar_configuracao_json(str(RAIZ / "config.json")))

    # 6. Luz: medidor + realce sobre uma sequência escura
    realce = RealceLuz(limiar_boa=cfg.luz_limiar_boa, limiar_critica=cfg.luz_limiar_critica,
                       ruido_baixa=cfg.luz_ruido_baixa, ruido_critico=cfg.luz_ruido_critico,
                       ganho_max=cfg.luz_ganho_max, empilhar_max=cfg.luz_empilhar_max)
    luz = []
    for i in range(10):
        escuro = np.clip(cs["lateral_mesa_preta"].astype(np.float32) * 0.2
                         + np.random.default_rng(700 + i).normal(0, 5, (480, 640, 3)), 0, 255).astype(np.uint8)
        saida = realce.processar(escuro)
        luz.append({"estado": realce.estado, "hud": descrever_luz(realce.estado), "md5": _md5(saida)})
    r["luz"] = _limpo(luz)

    # 7. Posição (NMEA)
    leitor = LeitorPosicao("windows")
    leitor.processar_windows("POS;-23.5641;-46.6524;83")
    r["posicao"] = _limpo({
        "gga": ler_sentenca("$GPGGA,123519.00,2333.8460,S,04639.1440,W,1,09,0.9,760.0,M,-5.0,M,,*7B"),
        "windows": {k: v for k, v in (leitor.atual(5.0) or {}).items() if k != "idade_s"},
    })

    # 8. SQLite: gravar, atualizar, ler de volta
    with tempfile.TemporaryDirectory() as tmp:
        con = sqlite3.connect(Path(tmp) / "eq.db")
        con.executescript((RAIZ / "Banco de dados" / "schema_sqlite.sql").read_text(encoding="utf-8"))
        migrar_banco_local(con)
        ev = fechados[0]
        pos = {"lat": -23.5641, "lon": -46.6524, "hdop": 0.9, "satelites": 9, "fonte": "gnss_serial", "idade_s": 0.3, "precisao_m": None}
        u = salvar_inspecao(con, cfg, ev["indicadores"], ev["defeitos"], ev["status"], ev["confianca_saude"],
                            uuid_local="11111111-2222-3333-4444-555555555555", posicao=pos)
        atualizar_inspecao(con, u, ev["indicadores"], ev["defeitos"][:1], "quarentena", 0.7)
        con.row_factory = sqlite3.Row
        t = dict(con.execute("SELECT * FROM toras_local WHERE uuid_local = ?", (u,)).fetchone())
        hash_confere = t["hash_sha256"] == _hash_inspecao(u, t["log_id"], ev["indicadores"], ev["defeitos"][:1], "quarentena", pos)
        for campo in ("id", "log_id", "data_inspecao", "hash_sha256", "criado_em"):
            t.pop(campo)
        filhos_i = [dict(x) for x in con.execute(
            "SELECT tipo_indicador, valor, unidade, metodo_medicao, sync_status FROM indicadores_qualidade_local ORDER BY id")]
        filhos_d = [dict(x) for x in con.execute(
            "SELECT tipo_defeito, pos_x, pos_y, largura, altura, confianca, sync_status FROM defeitos_detectados_local ORDER BY id")]
        colunas = [c[1] for c in con.execute("PRAGMA table_info(toras_local)")]
        con.close()
    r["banco"] = _limpo({"tora": t, "hash_confere": hash_confere, "indicadores": filhos_i, "defeitos": filhos_d, "colunas": colunas})
    return r


def comparar(ref, atual, caminho="") -> list[str]:
    difs = []
    if isinstance(ref, dict) and isinstance(atual, dict):
        for k in sorted(set(ref) | set(atual)):
            if k not in ref:
                difs.append(f"{caminho}.{k}: NOVO (não existia na referência)")
            elif k not in atual:
                difs.append(f"{caminho}.{k}: SUMIU")
            else:
                difs += comparar(ref[k], atual[k], f"{caminho}.{k}")
    elif isinstance(ref, list) and isinstance(atual, list):
        if len(ref) != len(atual):
            difs.append(f"{caminho}: tamanho {len(ref)} -> {len(atual)}")
        for i, (x, y) in enumerate(zip(ref, atual)):
            difs += comparar(x, y, f"{caminho}[{i}]")
    elif ref != atual:
        difs.append(f"{caminho}: {ref!r} -> {atual!r}")
    return difs


def main() -> int:
    p = argparse.ArgumentParser(description="Teste de equivalência (golden master) da máquina de campo.")
    p.add_argument("--gerar", action="store_true", help="grava uma referência nova (mudança de comportamento intencional)")
    args = p.parse_args()

    atual = json.loads(json.dumps(executar(), ensure_ascii=False, default=str))
    if args.gerar:
        REFERENCIA.parent.mkdir(parents=True, exist_ok=True)
        REFERENCIA.write_text(json.dumps(atual, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        n_q = len(atual["quadros"])
        print(f"Referência gravada em {REFERENCIA.relative_to(RAIZ)} ({n_q} quadros, {len(atual['eventos'])} eventos).")
        return 0

    if not REFERENCIA.exists():
        print(f"Sem referência em {REFERENCIA.relative_to(RAIZ)}. Rode com --gerar numa versão confiável.")
        return 1
    ref = json.loads(REFERENCIA.read_text(encoding="utf-8"))
    difs = comparar(ref, atual)
    if difs:
        print(f"❌ {len(difs)} DIFERENÇA(S) em relação à referência:")
        for d in difs[:40]:
            print("   " + d)
        if len(difs) > 40:
            print(f"   ... e mais {len(difs) - 40}")
        return 1
    print(f"✅ Equivalente à referência: {len(atual['quadros'])} quadros, {len(atual['eventos'])} eventos, "
          f"luz, posição, classificação e banco idênticos (HUD pixel a pixel).")
    print("TUDO OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
