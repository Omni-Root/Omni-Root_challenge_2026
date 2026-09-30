"""
evento.py — Evento de tora: uma tora = um registro (rastreador de quadros e consolidação por mediana/união).

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

import time

import cv2
import numpy as np

from omniroot.classificacao import calcular_confianca_saude, classificar_qualidade
from omniroot.config import Config
from omniroot.luz import nivel_do_evento
from omniroot.medidas import calcular_massa_seca_kg, calcular_volume_m3


# ============================================================
# EVENTO DE TORA — uma tora, um registro
# ============================================================
# Antes, o loop gravava uma inspeção a cada 2 s, tivesse tora no frame ou
# não: uma tora parada 10 s na garra virava 5 registros e garra vazia
# também gerava linha. O dashboard contava "toras", mas eram janelas de
# 2 s. Aqui a unidade passa a ser a TORA:
#
#   vazio ──(tora estável por N análises)──► aberto ──(sem tora por M)──► fecha
#                                              │
#                                              └─(contorno "pulou": outra tora)─► fecha e reabre
#
# Ao fechar, consolida todos os quadros do evento:
#   - indicadores geométricos: MEDIANA (robusta a quadro ruim);
#   - vista predominante decide diâmetro; comprimento/tortuosidade só
#     dos quadros laterais, se houve algum;
#   - defeitos: UNIÃO por tipo+posição, exigindo presença em >= K quadros;
#   - saúde e status recalculados sobre os defeitos consolidados (mesmas
#     funções do quadro único — a regra de negócio não muda);
#   - volume e massa recalculados das medianas.
# ============================================================

def _bbox_contorno(contorno) -> tuple[int, int, int, int] | None:
    if contorno is None or len(contorno) < 3:
        return None
    x, y, w, h = cv2.boundingRect(np.asarray(contorno, dtype=np.int32).reshape(-1, 1, 2))
    return int(x), int(y), int(w), int(h)


def _iou_bbox(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = float(ix * iy)
    uniao = float(aw * ah + bw * bh) - inter
    return inter / uniao if uniao > 0 else 0.0


def consolidar_defeitos(quadros: list[list[dict]], min_quadros: int, tolerancia_px: float = 80.0) -> list:
    """
    União dos defeitos de todos os quadros de um evento: agrupa por tipo e
    posição (centro a <= tolerancia_px), fica com a detecção de maior
    confiança de cada grupo e descarta grupos vistos em menos de
    `min_quadros` quadros (um defeito real reaparece; ruído não).
    """
    grupos: list[dict] = []  # {"rep": defeito, "quadros": set(idx)}
    for i, defeitos in enumerate(quadros):
        for d in defeitos:
            cx = d["pos_x"] + d["largura"] / 2.0
            cy = d["pos_y"] + d["altura"] / 2.0
            alvo = None
            for g in grupos:
                r = g["rep"]
                if r["tipo_defeito"] != d["tipo_defeito"]:
                    continue
                rx = r["pos_x"] + r["largura"] / 2.0
                ry = r["pos_y"] + r["altura"] / 2.0
                if ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 <= tolerancia_px:
                    alvo = g
                    break
            if alvo is None:
                grupos.append({"rep": dict(d), "quadros": {i}})
            else:
                alvo["quadros"].add(i)
                if d["confianca"] > alvo["rep"]["confianca"]:
                    alvo["rep"] = dict(d)
    minimo = min(min_quadros, max(1, len(quadros)))
    return [g["rep"] for g in grupos if len(g["quadros"]) >= minimo]


def consolidar_evento(analises: list[dict], cfg: Config) -> dict:
    """
    Reduz as análises de um evento (uma tora) a UM resultado com a mesma
    forma de `analisar_frame` (status, confianca_saude, defeitos,
    indicadores, vista) — é o que vai para salvar_inspecao.
    """
    vistas = [a["vista"] for a in analises]
    vista = max(("secao", "lateral", "desconhecida"), key=vistas.count)
    da_vista = [a for a in analises if a["vista"] == vista] or analises
    laterais = [a for a in analises if a["vista"] == "lateral"]

    def mediana(lista: list[dict], chave: str) -> float:
        return float(np.median([a["indicadores"][chave]["valor"] for a in lista]))

    def metodo_mais_comum(lista: list[dict], chave: str) -> str:
        ms = [a["indicadores"][chave]["metodo"] for a in lista]
        return max(set(ms), key=ms.count)

    diametro_cm = round(mediana(da_vista, "diametro"), 2)
    metodo_diam = metodo_mais_comum(da_vista, "diametro")
    if laterais:
        comprimento_cm = round(mediana(laterais, "altura"), 2)
        metodo_compr = metodo_mais_comum(laterais, "altura")
        tortuosidade = round(mediana(laterais, "tortuosidade"), 2)
        metodo_tort = "opencv_eixo_flecha"
        comprimento_medido = True
    else:
        comprimento_cm = float(cfg.comprimento_corte_cm)
        metodo_compr = "comprimento_tracamento_config"
        tortuosidade = 0.0
        metodo_tort = "nao_aplicavel_secao"
        comprimento_medido = False
    porcentagem_casca = round(mediana(analises, "porcentagem_casca"), 1)
    densidade = float(analises[-1]["indicadores"]["densidade"]["valor"])

    defeitos = consolidar_defeitos([a["defeitos"] for a in analises], cfg.evento_defeito_min_quadros)
    confianca_saude = calcular_confianca_saude(defeitos)
    status = classificar_qualidade(confianca_saude, defeitos, cfg)
    volume_util_m3 = calcular_volume_m3(diametro_cm, comprimento_cm, confianca_saude)
    massa_seca_kg = calcular_massa_seca_kg(volume_util_m3, densidade)

    indicadores = {
        "densidade": dict(analises[-1]["indicadores"]["densidade"]),
        "altura": {"valor": comprimento_cm, "unidade": "cm", "metodo": metodo_compr},
        "diametro": {"valor": diametro_cm, "unidade": "cm", "metodo": metodo_diam},
        "tortuosidade": {"valor": tortuosidade, "unidade": "indice", "metodo": metodo_tort},
        "porcentagem_casca": {"valor": porcentagem_casca, "unidade": "%", "metodo": "opencv_otsu_casca_residual"},
        "volume_util": {"valor": volume_util_m3, "unidade": "m3", "metodo": "cilindro_" + ("medido" if comprimento_medido else "diam_medido_compr_config")},
        "massa_seca": {"valor": massa_seca_kg, "unidade": "kg", "metodo": f"volume_x_densidade_clone_{cfg.clone_id}"},
        "apodrecimento_pragas": {"valor": round(confianca_saude * 100, 2), "unidade": "%", "metodo": "yolo_severidade"},
    }

    # Proveniência da luz (ver omniroot/luz.py): tora medida em luz baixa ou
    # crítica (realçada) carrega "_luz_baixa" / "_luz_critica" no método dos
    # indicadores MEDIDOS NA IMAGEM — quem lê o dado sabe em que condição ele
    # saiu, e o dashboard não dispara alerta só com tora de luz crítica. Não
    # toca no que não vem da imagem (densidade, comprimento de traçamento,
    # tortuosidade não aplicável).
    luz = nivel_do_evento([a.get("luz", "boa") for a in analises])
    if luz != "boa":
        for chave in ("diametro", "altura", "tortuosidade", "porcentagem_casca"):
            metodo = indicadores[chave]["metodo"]
            if metodo not in ("comprimento_tracamento_config", "nao_aplicavel_secao"):
                indicadores[chave]["metodo"] = f"{metodo}_luz_{luz}"

    return {
        "status": status,
        "confianca_saude": confianca_saude,
        "defeitos": defeitos,
        "descartados": [],
        "indicadores": indicadores,
        "vista": vista,
        "quadros": len(analises),
        "quadros_laterais": len(laterais),
        "luz": luz,
    }


class RastreadorTora:
    """
    Máquina de estados que transforma a sequência de análises (uma por
    quadro) em EVENTOS de tora. `atualizar(a)` devolve o evento consolidado
    quando um fecha, senão None. Sem modelo, sem banco: só decide "é a
    mesma tora?" e agrega — testável com análises sintéticas.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.numero = 0                 # toras fechadas até agora
        self.analises: list[dict] = []  # quadros do evento aberto (ou candidatos a abrir)
        self.aberto = False
        self.inicio = 0.0
        self.ausentes = 0
        self.trocas = 0
        self.bbox_anterior: tuple | None = None
        self.descartados_ruido = 0
        self.uuid_atual: str | None = None  # registro já gravado do evento aberto (incremental)

    def parcial(self) -> dict | None:
        """Consolidação do evento ABERTO até agora (None se não há evento com quadros suficientes)."""
        if not self.aberto or len(self.analises) < self.cfg.evento_frames_minimo:
            return None
        evento = consolidar_evento(self.analises, self.cfg)
        evento["numero"] = self.numero + 1
        evento["duracao_s"] = round(time.monotonic() - self.inicio, 2)
        return evento

    @staticmethod
    def _presente(a: dict) -> bool:
        return a.get("eh_tora", True) and a.get("contorno") is not None and a.get("vista") != "desconhecida"

    def _fechar(self) -> dict | None:
        analises, self.analises = self.analises, []
        uuid_evento, self.uuid_atual = self.uuid_atual, None
        self.aberto = False
        self.ausentes = 0
        self.trocas = 0
        self.bbox_anterior = None
        if len(analises) < self.cfg.evento_frames_minimo:
            self.descartados_ruido += 1
            return None
        self.numero += 1
        evento = consolidar_evento(analises, self.cfg)
        evento["numero"] = self.numero
        evento["duracao_s"] = round(time.monotonic() - self.inicio, 2)
        evento["uuid_local"] = uuid_evento  # None se nunca foi gravado (não deveria acontecer)
        return evento

    def atualizar(self, a: dict) -> dict | None:
        cfg = self.cfg
        presente = self._presente(a)
        fechado = None

        if not self.aberto:
            if presente:
                self.analises.append(a)
                if len(self.analises) == 1:
                    self.inicio = time.monotonic()
                if len(self.analises) >= cfg.evento_frames_abrir:
                    self.aberto = True
                    self.bbox_anterior = _bbox_contorno(a["contorno"])
            else:
                self.analises = []  # candidato não estabilizou
            return None

        if not presente:
            self.ausentes += 1
            if self.ausentes >= cfg.evento_frames_fechar:
                fechado = self._fechar()
            return fechado

        self.ausentes = 0
        bbox = _bbox_contorno(a["contorno"])
        if self.bbox_anterior is not None and bbox is not None and _iou_bbox(bbox, self.bbox_anterior) < cfg.evento_iou_troca:
            self.trocas += 1
        else:
            self.trocas = 0
        if self.trocas >= cfg.evento_frames_troca:
            # Contorno pulou por vários quadros: é outra tora. Os quadros da
            # "troca" pertencem à nova; tira-os da antiga antes de fechar.
            novos = self.analises[-(cfg.evento_frames_troca - 1):] if cfg.evento_frames_troca > 1 else []
            self.analises = self.analises[: len(self.analises) - len(novos)]
            fechado = self._fechar()
            self.analises = novos + [a]
            self.inicio = time.monotonic()
            self.aberto = len(self.analises) >= cfg.evento_frames_abrir
            self.bbox_anterior = bbox
            return fechado

        self.analises.append(a)
        # A referência só acompanha quadros que BATERAM com a tora atual; num
        # quadro suspeito ela fica parada, senão a "troca" nunca acumula.
        if self.trocas == 0:
            self.bbox_anterior = bbox
        return None

    def descricao(self) -> str:
        """Linha de HUD: estado atual do rastreador."""
        if self.aberto:
            gravada = "gravada, atualizando" if self.uuid_atual else "abrindo"
            return f"Tora #{self.numero + 1} em analise ({gravada}) | {len(self.analises)} quadros | {time.monotonic() - self.inicio:.1f}s"
        if self.analises:
            return f"Tora entrando... ({len(self.analises)}/{self.cfg.evento_frames_abrir})"
        return f"Aguardando tora | {self.numero} gravadas"
