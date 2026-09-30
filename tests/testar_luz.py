"""
testar_luz.py — pouca luz (omniroot/luz.py) com cenas sintéticas, sem câmera

A mesma "tora" (elipse cor de madeira sobre mesa preta) em três condições:
  - luz boa;
  - luz BAIXA: cena escurecida para 30% + ruído de sensor (câmera no ganho alto);
  - luz CRÍTICA: cena a 7% + ruído (mede, com baixa confiança).

Verifica:
  1. o medidor classifica cada condição (boa / baixa / critica);
  2. em luz boa o quadro passa intacto;
  3. em luz baixa o realce (empilhamento + ganho) reduz o ruído e PRESERVA a
     cor da madeira (matiz), e o pipeline REAL do main.py (segmentação +
     portão "é madeira?") volta a reconhecer a tora;
  4. movimento zera o empilhamento (não borra tora entrando);
  5. o evento consolidado em luz baixa marca "_luz_baixa" nos métodos.

É cena sintética: prova a lógica, não substitui o teste de bancada
(apagar a luz com a webcam e calibrar os limiares luz_* no config.json).

USO (raiz do repositório):
    python tests/testar_luz.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omniroot.luz import RealceLuz, medir_luz, nivel_do_evento  # noqa: E402
from omniroot.config import Config  # noqa: E402
from omniroot.evento import consolidar_evento  # noqa: E402
from omniroot.segmentacao import segmentar_tora_origem, validar_tora  # noqa: E402


def cena(fator: float, sigma_ruido: float, seed: int = 0, dx: int = 0) -> np.ndarray:
    """Tora cor de madeira com textura sobre mesa preta; `fator` = quanta luz; ruído de sensor gaussiano."""
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), 16.0, dtype=np.float32)
    mascara = np.zeros((480, 640), np.uint8)
    cv2.ellipse(mascara, (320 + dx, 240), (230, 70), 4, 0, 360, 255, -1)
    madeira = np.array([118, 168, 204], np.float32)  # BGR
    textura = cv2.GaussianBlur(np.random.default_rng(99).normal(0, 14, (480, 640)).astype(np.float32), (0, 0), 3)
    img[mascara > 0] = madeira + textura[mascara > 0][:, None]
    img *= fator
    img += rng.normal(0, sigma_ruido, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def matiz_madeira(frame: np.ndarray) -> float:
    mascara = np.zeros(frame.shape[:2], np.uint8)
    cv2.ellipse(mascara, (320, 240), (200, 55), 4, 0, 360, 255, -1)
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (9, 9), 0), cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[..., 0][mascara > 0]))


def portao(frame: np.ndarray, cfg: Config) -> tuple[bool, str]:
    mask, contorno, origem = segmentar_tora_origem(frame)
    ok, motivo, _ = validar_tora(frame, mask, contorno, origem, cfg)
    return ok, motivo


def main() -> int:
    falhas = 0

    def checar(cond: bool, msg: str) -> None:
        nonlocal falhas
        print(("  ✅ " if cond else "  ❌ ") + msg)
        if not cond:
            falhas += 1

    cfg = Config()

    def realce_novo() -> RealceLuz:
        return RealceLuz(
            limiar_boa=cfg.luz_limiar_boa, limiar_critica=cfg.luz_limiar_critica,
            ruido_baixa=cfg.luz_ruido_baixa, ruido_critico=cfg.luz_ruido_critico,
            ganho_max=cfg.luz_ganho_max, empilhar_max=cfg.luz_empilhar_max,
        )

    print("1) medidor")
    for nome, fator, sigma, esperado in [("boa", 1.0, 2.0, "boa"), ("baixa", 0.30, 5.0, "baixa"), ("critica", 0.07, 5.0, "critica")]:
        r = realce_novo()
        for i in range(10):
            r.processar(cena(fator, sigma, seed=i))
        b, n = medir_luz(cena(fator, sigma, seed=50))
        checar(r.nivel == esperado, f"luz {nome}: medido '{r.nivel}' (brilho {b:.0f}, ruído {n:.1f})")

    print("2) luz boa passa intacta")
    r = realce_novo()
    q = cena(1.0, 2.0, seed=1)
    checar(np.array_equal(r.processar(q), q), "quadro de saída idêntico ao de entrada, ganho 1.0")
    ok_boa, motivo_boa = portao(q, cfg)
    checar(ok_boa, f"referência: em luz boa o portão reconhece a tora ({motivo_boa or 'ok'})")

    print("3) luz baixa: realce")
    r = realce_novo()
    saida = None
    for i in range(12):  # tora parada: vários quadros com ruído independente
        saida = r.processar(cena(0.30, 5.0, seed=100 + i))
    crua = cena(0.30, 5.0, seed=200)
    _, ruido_cru = medir_luz(crua)
    brilho_saida, ruido_saida = medir_luz(saida)
    ruido_rel_cru = ruido_cru / max(medir_luz(crua)[0], 1)
    ruido_rel_saida = ruido_saida / max(brilho_saida, 1)
    checar(r.estado["empilhados"] == cfg.luz_empilhar_max, f"{r.estado['empilhados']} quadros empilhados")
    checar(r.ganho > 1.5, f"ganho aplicado {r.ganho:.2f}x (brilho {medir_luz(crua)[0]:.0f} -> {brilho_saida:.0f})")
    checar(ruido_rel_saida < ruido_rel_cru * 0.6,
           f"ruído relativo ao brilho caiu: {ruido_rel_cru:.3f} -> {ruido_rel_saida:.3f}")
    m_boa, m_saida = matiz_madeira(cena(1.0, 0.0)), matiz_madeira(saida)
    checar(abs(m_boa - m_saida) <= 3, f"cor preservada: matiz da madeira {m_boa:.0f} (luz boa) vs {m_saida:.0f} (realçada)")
    ok_crua, motivo_crua = portao(crua, cfg)
    ok_real, motivo_real = portao(saida, cfg)
    print(f"     portão sem realce: {'tora' if ok_crua else 'SEM TORA'} ({motivo_crua or 'ok'})")
    checar(ok_real, f"portão COM realce reconhece a tora ({motivo_real or 'ok'})")

    print("3b) onde o pipeline cru desiste e o realce ainda mede")
    r = realce_novo()
    for i in range(12):
        s = r.processar(cena(0.15, 6.0, seed=500 + i))
    crua = cena(0.15, 6.0, seed=600)
    ok_crua, motivo_crua = portao(crua, cfg)
    ok_real, motivo_real = portao(s, cfg)
    checar(not ok_crua, f"15% de luz SEM realce: o portão perde a tora ({motivo_crua or 'ok'}) — é o problema")
    checar(ok_real and r.nivel == "baixa", f"15% de luz COM realce: tora reconhecida ({motivo_real or 'ok'}), nível '{r.nivel}'")

    print("4) movimento zera o empilhamento")
    r = realce_novo()
    for i in range(6):
        r.processar(cena(0.30, 5.0, seed=300 + i))
    antes = r.estado["empilhados"]
    r.processar(cena(0.30, 5.0, seed=400, dx=90))  # a tora "pulou" 90 px
    checar(antes >= 5 and r.estado["empilhados"] == 1, f"empilhados {antes} -> {r.estado['empilhados']} após a tora mexer")

    print("5) proveniência no evento")
    base = {
        "status": "aprovado", "confianca_saude": 1.0, "defeitos": [], "vista": "lateral",
        "indicadores": {
            "densidade": {"valor": 490.0, "unidade": "kg/m3", "metodo": "lookup_clone_SP3108"},
            "altura": {"valor": 300.0, "unidade": "cm", "metodo": "imagem_lateral_marcador_aruco"},
            "diametro": {"valor": 20.0, "unidade": "cm", "metodo": "imagem_lateral_marcador_aruco"},
            "tortuosidade": {"valor": 3.0, "unidade": "indice", "metodo": "opencv_eixo_flecha"},
            "porcentagem_casca": {"valor": 10.0, "unidade": "%", "metodo": "opencv_otsu_casca_residual"},
        },
    }
    ev_baixa = consolidar_evento([{**base, "luz": "baixa"}] * 4 + [{**base, "luz": "boa"}], cfg)
    ev_boa = consolidar_evento([{**base, "luz": "boa"}] * 5, cfg)
    ind = ev_baixa["indicadores"]
    checar(ev_baixa["luz"] == "baixa" and ind["porcentagem_casca"]["metodo"].endswith("_luz_baixa")
           and ind["diametro"]["metodo"].endswith("_luz_baixa") and ind["tortuosidade"]["metodo"].startswith("opencv"),
           f"evento em luz baixa marcado ({ind['porcentagem_casca']['metodo']})")
    checar(not ind["densidade"]["metodo"].endswith("_luz_baixa"), "densidade (não vem da imagem) não é marcada")
    checar(ev_boa["luz"] == "boa" and not ev_boa["indicadores"]["porcentagem_casca"]["metodo"].endswith("_luz_baixa"),
           "evento em luz boa não é marcado")
    ev_critica = consolidar_evento([{**base, "luz": "critica"}] * 3 + [{**base, "luz": "baixa"}] * 2, cfg)
    checar(ev_critica["luz"] == "critica"
           and ev_critica["indicadores"]["porcentagem_casca"]["metodo"] == "opencv_otsu_casca_residual_luz_critica",
           "evento em luz crítica é MEDIDO e marcado _luz_critica (não recusado)")

    print("6) nível da tora a partir dos quadros")
    checar(nivel_do_evento(["boa"] * 3 + ["critica"] * 2) == "boa", "3 boa + 2 crítica -> boa")
    checar(nivel_do_evento(["boa"] * 2 + ["critica"] * 2 + ["baixa"]) == "baixa",
           "2 boa + 2 crítica + 1 baixa -> baixa (maioria em luz ruim, mas não maioria crítica)")
    checar(nivel_do_evento(["critica"] * 3 + ["boa"] * 2) == "critica", "maioria crítica -> critica")
    checar(nivel_do_evento([]) == "boa", "sem quadros -> boa")

    print()
    print("TUDO OK" if falhas == 0 else f"{falhas} FALHA(S)")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
