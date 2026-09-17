"""
testar_evento_tora.py — RastreadorTora com análises sintéticas (sem modelo, sem câmera)

Verifica a semântica "uma tora = um registro":
  1. tora parada 40 quadros + garra vazia         -> 1 evento, mediana/união corretas
  2. garra vazia o tempo todo                      -> 0 eventos
  3. duas toras sem quadro vazio entre elas (pulo) -> 2 eventos
  4. tora que aparece 2 quadros e some (ruído)     -> 0 eventos
  5. defeito visto em 1 quadro só é descartado; visto em 3 fica

USO (raiz do repositório):
    python tests/testar_evento_tora.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import Config, RastreadorTora, consolidar_defeitos  # noqa: E402


def contorno_rect(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32).reshape(-1, 1, 2)


def analise(vista="lateral", contorno=None, diam=20.0, compr=300.0, tort=3.0, casca=10.0, defeitos=None):
    """Monta um dict com a forma de analisar_frame(), só com o que o rastreador usa."""
    if contorno is None and vista != "desconhecida":
        contorno = contorno_rect(100, 100, 400, 120)
    lateral = vista == "lateral"
    return {
        "status": "aprovado",
        "confianca_saude": 1.0,
        "defeitos": list(defeitos or []),
        "descartados": [],
        "contorno": contorno if vista != "desconhecida" else None,
        "vista": vista,
        "indicadores": {
            "densidade": {"valor": 490.0, "unidade": "kg/m3", "metodo": "lookup_clone_SP3108"},
            "altura": {"valor": compr if lateral else 600.0, "unidade": "cm",
                       "metodo": "imagem_lateral_marcador_aruco" if lateral else "comprimento_tracamento_config"},
            "diametro": {"valor": diam, "unidade": "cm", "metodo": f"imagem_{vista}_marcador_aruco"},
            "tortuosidade": {"valor": tort if lateral else 0.0, "unidade": "indice",
                             "metodo": "opencv_eixo_flecha" if lateral else "nao_aplicavel_secao"},
            "porcentagem_casca": {"valor": casca, "unidade": "%", "metodo": "opencv_otsu_casca_residual"},
            "volume_util": {"valor": 0.1, "unidade": "m3", "metodo": "cilindro_medido"},
            "massa_seca": {"valor": 49.0, "unidade": "kg", "metodo": "volume_x_densidade_clone_SP3108"},
            "apodrecimento_pragas": {"valor": 100.0, "unidade": "%", "metodo": "yolo_severidade"},
        },
        "hud": [],
    }


def defeito(tipo, x, y, conf):
    return {"tipo_defeito": tipo, "pos_x": x, "pos_y": y, "largura": 40.0, "altura": 40.0,
            "area_relativa": 0.002, "confianca": conf}


def rodar(rastreador, sequencia):
    eventos = []
    for a in sequencia:
        e = rastreador.atualizar(a)
        if e is not None:
            eventos.append(e)
    return eventos


def main() -> int:
    cfg = Config()
    falhas = 0

    def check(cond, msg):
        nonlocal falhas
        print(("  ✅ " if cond else "  ❌ ") + msg)
        if not cond:
            falhas += 1

    # 1. tora parada, com um quadro ruim no meio (mediana ignora)
    print("1) tora parada 40 quadros + garra vazia")
    seq = [analise(diam=20.0 + (i % 3) * 0.1, tort=3.0) for i in range(40)]
    seq[17] = analise(diam=95.0, tort=40.0)  # quadro ruim
    seq += [analise(vista="desconhecida") for _ in range(cfg.evento_frames_fechar)]
    ev = rodar(RastreadorTora(cfg), seq)
    check(len(ev) == 1, f"1 evento (obtido {len(ev)})")
    if ev:
        d = ev[0]["indicadores"]["diametro"]["valor"]
        t = ev[0]["indicadores"]["tortuosidade"]["valor"]
        check(19.9 <= d <= 20.3, f"diâmetro = mediana ≈ 20 (obtido {d})")
        check(t < 5, f"tortuosidade não contaminada pelo quadro ruim (obtido {t})")
        check(ev[0]["quadros"] == 40, f"40 quadros consolidados (obtido {ev[0]['quadros']})")
        check(ev[0]["vista"] == "lateral", "vista predominante lateral")

    # 2. garra vazia
    print("2) garra vazia 100 quadros")
    ev = rodar(RastreadorTora(cfg), [analise(vista="desconhecida") for _ in range(100)])
    check(len(ev) == 0, f"0 eventos (obtido {len(ev)})")

    # 3. duas toras sem gap: contorno pula de lugar
    print("3) duas toras sem quadro vazio entre elas")
    r = RastreadorTora(cfg)
    seq = [analise(contorno=contorno_rect(50, 50, 300, 100), diam=18.0) for _ in range(30)]
    seq += [analise(contorno=contorno_rect(600, 500, 300, 100), diam=30.0) for _ in range(30)]
    seq += [analise(vista="desconhecida") for _ in range(cfg.evento_frames_fechar)]
    ev = rodar(r, seq)
    check(len(ev) == 2, f"2 eventos (obtido {len(ev)})")
    if len(ev) == 2:
        check(abs(ev[0]["indicadores"]["diametro"]["valor"] - 18.0) < 0.01, "1ª tora diâmetro 18")
        check(abs(ev[1]["indicadores"]["diametro"]["valor"] - 30.0) < 0.01, "2ª tora diâmetro 30")
        check(ev[0]["quadros"] + ev[1]["quadros"] == 60, f"nenhum quadro perdido ({ev[0]['quadros']}+{ev[1]['quadros']})")

    # 4. ruído: aparece 2 quadros e some
    print("4) tora que aparece 2 quadros e some")
    r = RastreadorTora(cfg)
    seq = [analise() for _ in range(2)] + [analise(vista="desconhecida") for _ in range(20)]
    ev = rodar(r, seq)
    check(len(ev) == 0 and r.numero == 0, "nada gravado")

    # 5. consolidação de defeitos
    print("5) união de defeitos com mínimo de quadros")
    quadros = [
        [defeito("Crack", 100, 100, 0.6), defeito("Live_Knot", 300, 100, 0.9)],
        [defeito("Crack", 105, 102, 0.8)],
        [defeito("Crack", 98, 99, 0.7), defeito("Dead_Knot", 500, 300, 0.95)],
        [],
    ]
    cons = consolidar_defeitos(quadros, min_quadros=2)
    tipos = sorted(d["tipo_defeito"] for d in cons)
    check(tipos == ["Crack"], f"só Crack (3 quadros) sobrevive; Live_Knot/Dead_Knot (1 quadro) caem (obtido {tipos})")
    if cons:
        check(cons[0]["confianca"] == 0.8, f"fica a detecção de maior confiança (obtido {cons[0]['confianca']})")

    # 5b. evento com defeito grave em 3 quadros -> reprovado
    print("5b) status do evento recalculado sobre os defeitos consolidados")
    seq = [analise(defeitos=[defeito("Dead_Knot", 200, 150, 0.9)] if i % 4 == 0 else []) for i in range(20)]
    seq += [analise(vista="desconhecida") for _ in range(cfg.evento_frames_fechar)]
    ev = rodar(RastreadorTora(cfg), seq)
    check(len(ev) == 1 and ev[0]["status"] == "reprovado", f"nó morto visto em 5 quadros reprova (obtido {ev[0]['status'] if ev else None})")

    print()
    print("TUDO OK" if falhas == 0 else f"{falhas} FALHA(S)")
    return 0 if falhas == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
