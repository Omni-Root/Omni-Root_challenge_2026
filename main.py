"""
main.py — Script principal do Raspberry Pi (Tarefa 2)
Projeto: Qualidade da Madeira — Challenge FIAP x John Deere/Suzano

O que este script faz, em ordem:
  1. Captura um frame da câmera acoplada à máquina
  2. Roda o modelo YOLO (NCNN) para detectar defeitos (praga, apodrecimento)
  3. Converte pixel -> cm real usando uma distância câmera-tora FIXA e
     calibrada (não sensor físico — a câmera é montada numa posição fixa
     em relação à tora, então a distância não muda entre leituras)
  4. Calcula os 4 indicadores de qualidade (densidade, altura, tortuosidade,
     apodrecimento_pragas) — densidade vem de um lookup por clone/material
     genético (data/clones_densidade.json), NÃO de sensor físico
  5. Classifica a tora (aprovado / quarentena / reprovado) usando o limiar
     de confiança de 85%
  6. Grava tudo no SQLite local (schema_sqlite.sql), pronto para o
     sync.go sincronizar com o PostgreSQL quando houver rede

Modo de uso:
  python3 main.py --simulado --gui

  (a distinção entre "Raspberry real" e "simulado" não existe mais para o
  sensor — o único hardware é a câmera. --simulado agora só controla se a
  janela com as bounding boxes é exibida na demonstração.)

Dependências (requirements.txt):
  ultralytics
  opencv-python-headless
  numpy
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

@dataclass
class Config:
    # --- Identidade da máquina (deve bater com o numero_serie no Postgres) ---
    maquina_id: str = "RASPI-DEMO-01"
    talhao_id: str = "Talhão Demo"
    clone_id: str = "SP3108"             # Clone genético do eucalipto (ex: SP3108, SP2974, CO41H_TEST)
    idade_talhao_anos: float = 5.3       # Idade do plantio na data da colheita

    # --- Banco local ---
    sqlite_path: str = "./omni_root_local.db"

    # --- Modelo de IA ---
    modelo_ncnn_path: str = "./models/wood_ncnn_model"
    conf_threshold: float = 0.40
    iou_threshold: float = 0.55
    imgsz: int = 1024

    # --- Câmera ---
    camera_index: int = 0
    intervalo_captura_seg: float = 2.0  # tempo entre análises

    # --- Regra de negócio (limiar de aprovação) ---
    limiar_aprovacao: float = 0.85       # 85% de confiança mínima
    limiar_quarentena: float = 0.60      # abaixo disso já é reprovado direto

    # --- Distância câmera-tora, FIXA e calibrada (sem sensor) ---
    # A câmera é montada numa posição fixa em relação à tora (no braço do
    # harvester ou na maquete), então a distância não muda entre leituras.
    # CALIBREM ESSE VALOR com a montagem real antes da apresentação: meçam
    # a distância física câmera-tora uma vez, com fita métrica, e coloquem
    # o número aqui. É o único "hardware" que isso substitui — nenhum
    # sensor físico é necessário.
    distancia_camera_tora_cm: float = 45.0

    # --- Câmera: parâmetros para cálculo de dimensão real (GSD) ---
    # Precisam ser calibrados com a câmera real usada na apresentação!
    distancia_focal_mm: float = 4.0      # foco da lente
    largura_sensor_mm: float = 6.3       # largura física do sensor da câmera
    largura_imagem_px: int = 640         # resolução usada na inferência


# ============================================================
# PROCESSADOR DO INVENTÁRIO FLORESTAL (John Deere / Suzano)
# ============================================================
def carregar_inventario_florestal(
    caminho_densidade: str = "./data/clones_densidade.json",
    caminho_dendrometria: str = "./data/inventario_johndeere.json",
) -> dict:
    """
    Monta o dicionário de referência por clone usado pela IA, a partir de
    DUAS fontes com papéis bem diferentes — importante não misturar:

      - clones_densidade.json: densidade básica (kg/m3) por clone/material
        genético. É esse o dado que, na prática do setor (confirmado por
        e-mail com o contato da John Deere), varia por genética e não por
        medição em campo.
        ATENÇÃO: os valores aí HOJE são placeholders de demonstração, não
        dados publicados/verificados. Antes da apresentação final, troquem
        por valores reais — pedidos à Suzano/John Deere, ou de literatura
        técnica (IPEF, Embrapa Florestas etc.), e citem a fonte no relatório.

      - inventario_johndeere.json: amostras de DAP por árvore, agrupadas por
        clone — isso sim vem da planilha real de inventário que a JD
        passou. NÃO tem densidade. Serve só como contexto dendrométrico
        (DAP médio, idade do talhão) para exibir no dashboard/relatório e,
        opcionalmente, conferir o diâmetro medido pela câmera contra a
        média do talhão. NÃO é usado para calcular densidade — DAP não é
        um bom preditor de densidade básica, e tentar derivar um a partir
        do outro foi o erro da versão anterior deste arquivo.
    """
    estatisticas_clones: dict = {}

    p_dens = Path(caminho_densidade)
    if p_dens.exists():
        with open(p_dens, "r", encoding="utf-8") as f:
            dados_densidade = json.load(f)
        for clone_key, info in dados_densidade.items():
            if not isinstance(info, dict):
                continue
            # densidade_base pode vir None: o clone está cadastrado no
            # Postgres mas ainda aguarda laudo de laboratório. Nesse caso
            # deixamos None de propósito -- calcular_densidade_estimada
            # avisa no console em vez de mascarar com um número inventado.
            dens_bruta = info.get("densidade_base")
            estatisticas_clones[str(clone_key).upper()] = {
                "densidade_base": float(dens_bruta) if dens_bruta is not None else None,
                "taxa_maturacao": info.get("taxa_maturacao"),  # guardado, NÃO aplicado (ver calcular_densidade_estimada)
                "especie": info.get("especie", info.get("descricao", "Eucalyptus sp.")),
                "dap_medio_inventario": None,
                "idade_anos": None,
            }

    p_dendro = Path(caminho_dendrometria)
    if p_dendro.exists():
        with open(p_dendro, "r", encoding="utf-8") as f:
            dados_dendro = json.load(f)
        for clone_key, info in dados_dendro.get("clones", dados_dendro).items():
            if not isinstance(info, dict):
                continue
            key_upper = str(clone_key).upper()
            amostras = info.get("dap_amostras_cm", [])
            dap_medio = float(np.mean(amostras)) if amostras else None

            entrada = estatisticas_clones.setdefault(key_upper, {
                "densidade_base": None,  # sem referência de densidade cadastrada para este clone
                "taxa_maturacao": None,
                "especie": info.get("especie", "Eucalyptus sp."),
                "dap_medio_inventario": None,
                "idade_anos": None,
            })
            entrada["dap_medio_inventario"] = round(dap_medio, 2) if dap_medio is not None else None
            entrada["idade_anos"] = info.get("idade_anos")

    return estatisticas_clones


def carregar_configuracao_json(caminho_json: str = "./config.json") -> Config:
    """Carrega as configurações operacionais da máquina a partir de um arquivo JSON estático."""
    cfg = Config()
    p = Path(caminho_json)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                dados = json.load(f)
                if isinstance(dados, dict):
                    for k, v in dados.items():
                        if hasattr(cfg, k):
                            setattr(cfg, k, v)
        except Exception as e:
            print(f"⚠️ Erro ao carregar {caminho_json}: {e}")
    return cfg


INVENTARIO_FLORESTAL = carregar_inventario_florestal()
CONFIG = carregar_configuracao_json()


# ============================================================
# CÁLCULO DOS INDICADORES DE QUALIDADE
# ============================================================

def calcular_dimensao_real_cm(tamanho_px: float, distancia_cm: float, cfg: Config) -> float:
    """
    Converte um tamanho em pixels para centímetros reais, usando a
    trigonometria de GSD (Ground Sample Distance) — a mesma lógica
    usada em fotografia aérea, adaptada para distância câmera-tora.

    GSD (cm/pixel) = (distância_cm * largura_sensor_mm) / (foco_mm * largura_imagem_px)
    """
    gsd_cm_por_px = (distancia_cm * cfg.largura_sensor_mm) / (
        cfg.distancia_focal_mm * cfg.largura_imagem_px
    )
    return round(tamanho_px * gsd_cm_por_px, 2)


def calcular_volume_m3(diametro_cm: float, comprimento_cm: float, confianca_saude: float) -> float:
    """
    Estima o volume da tora a partir de medidas REAIS (diâmetro e
    comprimento, ambos vindos do bounding box do contorno + GSD) —
    não mais de uma proporção arbitrária entre os dois.

    Fórmula do cilindro (aproximação de Huber, sem afilamento):
        volume_bruto = pi * raio^2 * comprimento

    Para volume comercial mais preciso (considerando o afilamento
    natural da tora), a indústria usa a fórmula de Smalian, que pede
    o diâmetro nas DUAS pontas:
        volume = (pi/8) * (diametro_fino^2 + diametro_grosso^2) * comprimento
    Isso exigiria medir o diâmetro nas duas extremidades da tora (hoje
    só medimos uma vez, no centro do frame) — fica como próximo passo
    se quiserem refinar depois da apresentação.

    volume_util desconta, de forma proporcional, o volume equivalente
    à severidade dos defeitos detectados (confianca_saude = 1.0 = tora
    limpa, sem desconto).
    """
    raio_m = (diametro_cm / 100.0) / 2.0
    comprimento_m = comprimento_cm / 100.0
    volume_bruto_m3 = float(np.pi * (raio_m ** 2) * comprimento_m)
    return round(float(volume_bruto_m3 * confianca_saude), 3)


def calcular_massa_seca_kg(volume_util_m3: float, densidade_kg_m3: float) -> float:
    """
    Massa seca estimada = volume útil medido pela câmera (m3) x densidade
    de referência do clone (kg/m3).

    Cruza um dado MEDIDO (volume, vindo da visão computacional) com um dado
    ESTIMADO (densidade, vindo do lookup por clone — ver
    calcular_densidade_estimada). O resultado herda a incerteza da
    densidade: é uma estimativa de massa, não uma pesagem real. Isso é
    exatamente o que a apresentação deve deixar claro ao citar este número.
    """
    return round(float(volume_util_m3) * float(densidade_kg_m3), 1)


def calcular_densidade_estimada(clone_id: str, inventario_stats: dict | None = None) -> float:
    """
    Retorna a densidade básica de referência (kg/m3) para o clone/material
    genético informado — um lookup simples, não um modelo derivado.

    De onde vem o valor (ver data/clones_densidade.json para as fontes
    completas): buscamos os códigos de clone do inventário (SP3108, SP2974
    etc.) em literatura científica e não há correspondência pública — são
    códigos internos proprietários, o que bate com o que o contato da John
    Deere já tinha explicado por e-mail. A própria planilha de inventário
    também não permite identificar a espécie botânica real por trás de cada
    código (a coluna "Espécie" está preenchida com o próprio código do
    clone). Por isso, TODOS os clones cadastrados hoje usam o mesmo valor:
    a densidade básica média de híbridos comerciais de Eucalyptus grandis x
    E. urophylla — o material genético mais comum em plantios industriais
    de celulose no Brasil — medida em 3 fontes técnicas/acadêmicas reais
    (tese USP, boletim técnico e artigo Revista Árvore/SciELO, citados no
    JSON). Isso é uma aproximação de literatura no nível de espécie/híbrido,
    não o dado de laboratório do clone específico (que é proprietário).

    Por quê lookup e não fórmula: densidade básica da madeira é, na prática
    do setor, majoritariamente definida pelo material genético, não algo
    que se calcule a partir de DAP/altura/idade medidos em campo — uma
    versão anterior deste arquivo tentava "derivar" densidade com uma
    fórmula sem base científica (e com um bug que contava o efeito da
    idade duas vezes); foi removida.

    Se o clone não tiver densidade cadastrada em data/clones_densidade.json,
    cai num valor genérico de eucalipto e avisa no console — isso é
    intencional, para não mascarar dado faltante com um número inventado.
    """
    inv = inventario_stats if inventario_stats is not None else INVENTARIO_FLORESTAL
    key = str(clone_id).upper()
    info = inv.get(key)

    if not info or info.get("densidade_base") is None:
        print(
            f"⚠️  Sem densidade cadastrada para o clone '{clone_id}'. Usando média "
            f"genérica de eucalipto (500.0 kg/m3). Para corrigir, cadastre o valor "
            f"na tabela 'clones_densidade' do PostgreSQL central (fonte de verdade) "
            f"— o sync_daemon.py traz a atualização automaticamente na próxima "
            f"sincronização."
        )
        return 500.0

    return round(float(info["densidade_base"]), 1)


def calcular_porcentagem_casca(frame: np.ndarray | None, contorno: np.ndarray | None) -> float:
    """
    Mede a proporção da área de casca (%) em relação à seção da tora via OpenCV,
    analisando a diferença entre a área total do contorno e a região interna
    (obtida por erosão morfológica).

    Para eucalipto comercial, a faixa típica fica entre 8% e 20%. Valores fora
    dessa faixa não são impossíveis (tora descascada = ~2%, casca muito grossa
    ou contorno mal segmentado = >25%), mas geram um aviso no log.

    Retorna o valor REAL medido, sem clamp — se o número parecer estranho, é
    sinal de que o contorno precisa de ajuste, não de que devemos esconder a
    medida.
    """
    if frame is None or contorno is None or len(contorno) < 5:
        return 0.0  # sem dados suficientes para medir — retorna 0 (desconhecido)

    try:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if isinstance(contorno, list):
            contorno_pts = np.array(contorno, dtype=np.int32)
        else:
            contorno_pts = contorno.astype(np.int32)
        cv2.drawContours(mask, [contorno_pts], -1, 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_miolo = cv2.erode(mask, kernel, iterations=2)
        mask_casca = cv2.subtract(mask, mask_miolo)

        area_total = float(np.count_nonzero(mask))
        area_casca = float(np.count_nonzero(mask_casca))

        if area_total <= 0:
            return 0.0

        pct = (area_casca / area_total) * 100.0
        pct = round(float(pct), 1)

        # Aviso de outlier (não bloqueia, só informa)
        if pct < 5.0 or pct > 30.0:
            print(
                f"⚠️  Porcentagem de casca fora da faixa típica de eucalipto "
                f"(8-20%): {pct}%. Verifique a segmentação do contorno."
            )

        return pct
    except Exception:
        return 0.0


# ------------------------------------------------------------
# PRÉ-PROCESSAMENTO PARA O MODELO
# ------------------------------------------------------------
# O modelo foi TREINADO com as imagens convertidas para escala de
# cinza e equalizadas com CLAHE (ver a célula de preparação do
# dataset no notebook de treino). Se a inferência mandar o frame
# BGR cru da câmera, o modelo recebe uma distribuição de pixels
# diferente da que aprendeu -- isso degrada a precisão sem gerar
# erro nenhum, o tipo de problema que passa despercebido.
#
# Esta função replica EXATAMENTE o pré-processamento do treino:
#     gray = cvtColor(img, BGR2GRAY)
#     gray = CLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
#     img  = cvtColor(gray, GRAY2BGR)
#
# ⚠️ Se mudarem clipLimit/tileGridSize no notebook de treino,
#    mudem AQUI TAMBÉM -- os dois precisam andar juntos sempre.
# ------------------------------------------------------------
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocessar_para_modelo(frame: np.ndarray) -> np.ndarray:
    """Aplica grayscale + CLAHE, igual ao pré-processamento do treino."""
    if frame is None:
        return frame
    try:
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cinza = _CLAHE.apply(cinza)
        return cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
    except Exception:
        # Em caso de falha, devolve o frame original: melhor inferir
        # com qualidade degradada do que derrubar o loop inteiro.
        return frame


def extrair_contorno_tora(frame: np.ndarray) -> np.ndarray | None:
    """
    Segmenta a região da tora no frame usando técnicas de visão computacional
    clássica (Otsu thresholding + operações morfológicas).
    Retorna o contorno principal da tora ou None.
    """
    if frame is None:
        return None

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contornos, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            return None

        area_minima = (frame.shape[0] * frame.shape[1]) * 0.05
        validos = [c for c in contornos if cv2.contourArea(c) >= area_minima]
        return max(validos, key=cv2.contourArea) if validos else max(contornos, key=cv2.contourArea)
    except Exception:
        return None


def calcular_tortuosidade(mascara_ou_contorno) -> float:
    """
    Calcula um índice de tortuosidade (0 = perfeitamente reta,
    valores maiores = mais tortuosa) a partir do contorno da tora.

    Método: ajusta uma reta ao eixo do contorno e mede o desvio
    máximo perpendicular a essa reta, normalizado pelo comprimento.
    """
    if mascara_ou_contorno is None or len(mascara_ou_contorno) < 5:
        return 0.0

    try:
        pontos = mascara_ou_contorno.reshape(-1, 2).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pontos, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        direcao = np.array([vx, vy])
        origem = np.array([x0, y0])

        desvios = []
        for p in pontos:
            vetor = p - origem
            proj = np.dot(vetor, direcao) * direcao
            perpendicular = vetor - proj
            desvios.append(np.linalg.norm(perpendicular))

        extensao = pontos.max(axis=0) - pontos.min(axis=0)
        comprimento = max(float(np.linalg.norm(extensao)), 1.0)
        indice = (max(desvios) / comprimento) * 100.0
        return round(float(min(indice, 100.0)), 2)
    except Exception:
        return 0.0


def extrair_defeitos_yolo(resultado, cfg: Config, frame_shape: tuple | None = None) -> list:
    """Converte as detecções do YOLO em uma lista de defeitos estruturados com área relativa."""
    defeitos = []
    nomes_classes = resultado.names

    img_h, img_w = frame_shape[:2] if frame_shape is not None else (640, 640)
    area_total_img = float(img_h * img_w)

    for box in resultado.boxes:
        classe_id = int(box.cls[0])
        tipo_defeito = nomes_classes.get(classe_id, "wood_defect")
        confianca = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        largura = max(0.0, x2 - x1)
        altura = max(0.0, y2 - y1)
        area_box = largura * altura
        area_relativa = round(area_box / area_total_img, 4)

        defeitos.append({
            "tipo_defeito": tipo_defeito,
            "pos_x": round(x1, 2),
            "pos_y": round(y1, 2),
            "largura": round(largura, 2),
            "altura": round(altura, 2),
            "area_relativa": area_relativa,
            "confianca": round(confianca, 4),
        })

    return defeitos


# ============================================================
# REGRA DE NEGÓCIO — CLASSIFICAÇÃO FINAL DA TORA
# ============================================================

# ------------------------------------------------------------
# SEVERIDADE POR TIPO DE DEFEITO
# ------------------------------------------------------------
# Nem todo defeito pesa igual para a indústria de celulose. Esta
# graduação é o que diferencia a triagem de um simples "detectou
# algo = reprovado".
#
# IMPORTANTE: os nomes aqui são EXATAMENTE as classes do modelo
# treinado (data.yaml, nc=8). A versão anterior desta função
# procurava por "apodrecimento"/"praga"/"rot", que NÃO existem no
# modelo — ou seja, aquela regra nunca disparava.
#
# Justificativa de cada nível:
#   GRAVE
#     Dead_Knot       nó morto: não está integrado à fibra ao redor,
#                     pode soltar e virar buraco no processamento
#     Knot_missing    nó já ausente: o buraco existe
#     knot_with_crack nó + rachadura: combina dois problemas
#     resin           bolsa de resina: resina é extrativo, e o próprio
#                     enunciado do desafio liga teor de extrativos ao
#                     consumo de químicos para separar celulose da
#                     lignina
#   MODERADO
#     Crack           rachadura: gravidade varia com extensão/profundidade
#     Marrow          medula: tecido mole, fibra de baixa qualidade,
#                     mas área pequena
#     Quartzity       inclusão mineral: danifica lâmina, não a fibra
#   LEVE
#     Live_Knot       nó vivo: integrado à madeira ao redor; é o defeito
#                     mais comum e o de menor impacto para celulose
#
# Estes pesos são uma DECISÃO DE PROJETO baseada em características
# conhecidas da madeira, não uma norma técnica publicada. Se a
# Suzano/JD fornecer o critério de triagem oficial deles, é só
# ajustar os conjuntos abaixo — nada mais no código muda.
# ------------------------------------------------------------
DEFEITOS_GRAVES = ("Dead_Knot", "Knot_missing", "knot_with_crack", "resin")
DEFEITOS_MODERADOS = ("Crack", "Marrow", "Quartzity")
DEFEITOS_LEVES = ("Live_Knot",)

PESO_SEVERIDADE = {
    **{d: 1.0 for d in DEFEITOS_GRAVES},
    **{d: 0.5 for d in DEFEITOS_MODERADOS},
    **{d: 0.2 for d in DEFEITOS_LEVES},
}
PESO_PADRAO = 0.5  # classe desconhecida: trata como moderada, não ignora


def severidade_do_defeito(tipo_defeito: str) -> float:
    """Peso de 0.2 (leve) a 1.0 (grave) para um tipo de defeito."""
    return PESO_SEVERIDADE.get(tipo_defeito, PESO_PADRAO)


def calcular_confianca_saude(defeitos: list) -> float:
    """
    Score de saúde da tora (1.0 = limpa), ponderado por severidade.

    Antes, qualquer defeito descontava igual: um nó vivo detectado com
    90% de confiança derrubava a saúde tanto quanto um nó morto. Agora
    o desconto é proporcional à gravidade do defeito para o processo
    de celulose.
    """
    if not defeitos:
        return 1.0

    # O defeito mais crítico define o desconto principal...
    impacto_principal = max(
        d["confianca"] * severidade_do_defeito(d["tipo_defeito"]) for d in defeitos
    )
    # ...e a quantidade de defeitos adiciona um desconto menor,
    # limitado, para que uma tora cheia de defeitos leves ainda seja
    # penalizada — só que menos que uma com um defeito grave.
    impacto_acumulado = min(
        0.25,
        sum(d["confianca"] * severidade_do_defeito(d["tipo_defeito"]) for d in defeitos) * 0.05,
    )

    saude = 1.0 - (impacto_principal * 0.5) - impacto_acumulado
    return round(max(0.0, min(1.0, saude)), 4)


def classificar_qualidade(confianca_saude: float, defeitos: list, cfg: Config) -> str:
    """
    Triagem da tora, ponderada pela gravidade do defeito para a
    indústria de celulose:

      1. Defeito GRAVE com confiança >= 70%              -> reprovado
      2. Defeito extenso (>= 5% da área) com conf >= 65% -> reprovado
      3. Saúde abaixo de limiar_quarentena (60%)         -> reprovado
      4. Sem defeito E saúde >= limiar_aprovacao (85%)   -> aprovado
      5. Só defeitos LEVES e saúde alta                  -> aprovado
      6. Qualquer outro caso                             -> quarentena

    A regra 5 é a diferença prática do modelo ponderado: uma tora com
    apenas nós vivos (o defeito mais comum e de menor impacto) não
    precisa ir para revisão manual — antes ia, o que geraria fila de
    quarentena desnecessária em operação real.
    """
    # 1. Defeito grave com confiança alta
    tem_grave = any(
        d["tipo_defeito"] in DEFEITOS_GRAVES and d["confianca"] >= 0.70
        for d in defeitos
    )
    if tem_grave:
        return "reprovado"

    # 2. Defeito extenso (ignora leves: um nó vivo grande não reprova a tora)
    tem_defeito_extenso = any(
        d.get("area_relativa", 0.0) >= 0.05
        and d["confianca"] >= 0.65
        and d["tipo_defeito"] not in DEFEITOS_LEVES
        for d in defeitos
    )
    if tem_defeito_extenso:
        return "reprovado"

    # 3. Saúde muito baixa
    if confianca_saude < cfg.limiar_quarentena:
        return "reprovado"

    # 4. Tora limpa
    if confianca_saude >= cfg.limiar_aprovacao and len(defeitos) == 0:
        return "aprovado"

    # 5. Apenas defeitos leves, com saúde alta -> aprovado
    so_leves = defeitos and all(d["tipo_defeito"] in DEFEITOS_LEVES for d in defeitos)
    if so_leves and confianca_saude >= cfg.limiar_aprovacao:
        return "aprovado"

    # 6. Zona cinza -> revisão manual
    return "quarentena"



def gerar_hash_sha256(dados: dict) -> str:
    """Gera um hash de integridade do registro (rastreabilidade)."""
    dados_serializados = json.dumps(dados, sort_keys=True, default=str)
    return hashlib.sha256(dados_serializados.encode("utf-8")).hexdigest()


# ============================================================
# CAMADA DE BANCO DE DADOS (SQLite local)
# ============================================================

def conectar_banco(cfg: Config) -> sqlite3.Connection:
    conexao = sqlite3.connect(cfg.sqlite_path)
    conexao.execute("PRAGMA foreign_keys = ON")

    # Verifica se a tabela toras_local já existe, caso contrário carrega o schema
    tabelas = conexao.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='toras_local'").fetchall()
    if not tabelas:
        schema_path = Path(cfg.sqlite_path).parent / "Banco de dados" / "schema_sqlite.sql"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                conexao.executescript(f.read())
    return conexao


def salvar_inspecao(
    conexao: sqlite3.Connection,
    cfg: Config,
    indicadores: dict,
    defeitos: list,
    status_classificacao: str,
    confianca_media: float,
) -> str:
    """Grava a tora + indicadores + defeitos nas 3 tabelas locais."""
    cursor = conexao.cursor()

    uuid_local = str(uuid.uuid4())
    log_id = f"LOG-{datetime.now():%Y%m%d%H%M%S}-{uuid_local[:4]}"
    data_inspecao = datetime.now().isoformat(timespec="seconds")

    hash_dados = gerar_hash_sha256({
        "uuid_local": uuid_local,
        "log_id": log_id,
        "indicadores": indicadores,
        "defeitos": defeitos,
        "status": status_classificacao,
    })

    # --- 1. Insere a tora ---
    cursor.execute(
        """
        INSERT INTO toras_local
            (uuid_local, maquina_id, talhao_id, log_id, data_inspecao,
             confianca_ia, status_classificacao, hash_sha256, sync_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (uuid_local, cfg.maquina_id, cfg.talhao_id, log_id, data_inspecao,
         confianca_media, status_classificacao, hash_dados),
    )
    tora_id_local = cursor.lastrowid

    # --- 2. Insere os indicadores de qualidade ---
    for tipo, dados_indicador in indicadores.items():
        cursor.execute(
            """
            INSERT INTO indicadores_qualidade_local
                (tora_id, tipo_indicador, valor, unidade, metodo_medicao, sync_status)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (tora_id_local, tipo, dados_indicador["valor"],
             dados_indicador["unidade"], dados_indicador["metodo"]),
        )

    # --- 3. Insere os defeitos detectados ---
    for defeito in defeitos:
        cursor.execute(
            """
            INSERT INTO defeitos_detectados_local
                (tora_id, tipo_defeito, pos_x, pos_y, largura, altura, confianca, sync_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (tora_id_local, defeito["tipo_defeito"], defeito["pos_x"], defeito["pos_y"],
             defeito["largura"], defeito["altura"], defeito["confianca"]),
        )

    conexao.commit()
    return uuid_local


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="IA de Qualidade da Madeira — Raspberry Pi")
    parser.add_argument(
        "--simulado", action="store_true",
        help="Abre a janela com bounding boxes (mesmo efeito de --gui, mantido por compatibilidade)"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Abre janela gráfica exibindo as bounding boxes da câmera ao vivo"
    )
    parser.add_argument(
        "--config-file", type=str, default="./config.json",
        help="Caminho do arquivo JSON de configuração da máquina/operação"
    )
    parser.add_argument(
        "--clone", type=str, default=None,
        help="Sobrescreve o Código do Clone/Material Genético (ex: SP3108, SP2974, CO41H_TEST)"
    )
    parser.add_argument(
        "--idade", type=float, default=None,
        help="Sobrescreve a Idade do talhão em anos na colheita (ex: 5.3)"
    )
    args = parser.parse_args()

    cfg = carregar_configuracao_json(args.config_file)
    if args.clone is not None:
        cfg.clone_id = args.clone
    if args.idade is not None:
        cfg.idade_talhao_anos = args.idade

    print("📦 Carregando modelo IA...")
    caminho_modelo = None
    for p in [Path(cfg.modelo_ncnn_path), Path("./models/wood_best.pt"), Path("./models/wood_ncnn_model"), Path("./yolov8n.pt")]:
        if p.exists():
            caminho_modelo = str(p)
            break
    if caminho_modelo is None:
        caminho_modelo = "yolov8n.pt"

    modelo = YOLO(caminho_modelo, task="detect")
    print(f"✅ Modelo carregado com sucesso ({caminho_modelo})!")

    print(f"📷 Abrindo câmera (índice {cfg.camera_index})...")
    if sys.platform == "win32":
        # No Windows, o backend padrão do OpenCV às vezes demora ou falha
        # pra abrir a webcam. DirectShow é mais rápido e confiável lá.
        captura = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    else:
        captura = cv2.VideoCapture(cfg.camera_index)
    if not captura.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera.")

    # Buffer de 1 frame: por padrão o OpenCV enfileira frames e o read()
    # devolve os ANTIGOS quando o processamento não acompanha — é isso que dá
    # a sensação de atraso. Com buffer 1, sempre pegamos o frame mais recente.
    captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ------------------------------------------------------------------
    # Arquitetura em tempo real (a imagem nunca congela):
    #   - Thread PRINCIPAL: só captura e exibe o vídeo (roda a ~FPS da
    #     câmera, sempre fluido).
    #   - Thread de TRABALHO: roda o YOLO no frame mais recente e devolve as
    #     caixas/HUD para a principal desenhar. A inferência pesada nunca
    #     bloqueia o vídeo — no máximo as caixas atualizam um pouco atrás.
    # ------------------------------------------------------------------
    estado = {"frame": None, "boxes": [], "hud": [], "cor": (0, 255, 0)}
    lock = threading.Lock()
    parar = threading.Event()

    def worker_analise():
        # A conexão SQLite é criada AQUI: objetos sqlite3 só podem ser usados
        # na mesma thread em que foram criados.
        conexao = conectar_banco(cfg)
        ultima_gravacao = 0.0
        try:
            while not parar.is_set():
                with lock:
                    frame = None if estado["frame"] is None else estado["frame"].copy()
                if frame is None:
                    time.sleep(0.01)
                    continue
                try:
                    # --- 1. Roda o YOLO no frame mais recente ---
                    # O modelo recebe o frame PRÉ-PROCESSADO (grayscale + CLAHE),
                    # igual ao treino. As medições por OpenCV mais abaixo continuam
                    # usando o frame COLORIDO original -- contorno e porcentagem de
                    # casca funcionam melhor com a informação de cor preservada.
                    frame_modelo = preprocessar_para_modelo(frame)

                    resultados = modelo.predict(
                        source=frame_modelo,
                        conf=cfg.conf_threshold,
                        iou=cfg.iou_threshold,
                        imgsz=cfg.imgsz,
                        device="cpu",
                        verbose=False,
                    )
                    resultado = resultados[0]
                    defeitos = extrair_defeitos_yolo(resultado, cfg, frame.shape)

                    # Saúde da tora, ponderada pela GRAVIDADE de cada defeito
                    # (ver calcular_confianca_saude / PESO_SEVERIDADE).
                    confianca_saude = calcular_confianca_saude(defeitos)

                    # --- 2. Distância câmera-tora: fixa e calibrada, sem sensor ---
                    distancia_cm = cfg.distancia_camera_tora_cm

                    # --- 3. Extrai contorno da tora e calcula indicadores ---
                    contorno_tora = None
                    if resultado.masks is not None and len(resultado.masks.xy) > 0:
                        contorno_tora = max(resultado.masks.xy, key=len)
                    else:
                        contorno_tora = extrair_contorno_tora(frame)

                    if contorno_tora is not None and len(contorno_tora) > 0:
                        _, _, w_box, h_box = cv2.boundingRect(contorno_tora)
                        largura_px = float(w_box)
                        altura_px = float(h_box)
                    else:
                        largura_px = float(frame.shape[1] * 0.2)
                        altura_px = float(frame.shape[0] * 0.7)

                    altura_cm = calcular_dimensao_real_cm(altura_px, distancia_cm, cfg)
                    diametro_cm = calcular_dimensao_real_cm(largura_px, distancia_cm, cfg)
                    densidade = calcular_densidade_estimada(cfg.clone_id)
                    tortuosidade = calcular_tortuosidade(contorno_tora)
                    porcentagem_casca = calcular_porcentagem_casca(frame, contorno_tora)
                    volume_util_m3 = calcular_volume_m3(diametro_cm, altura_cm, confianca_saude)
                    massa_seca_kg = calcular_massa_seca_kg(volume_util_m3, densidade)

                    status = classificar_qualidade(confianca_saude, defeitos, cfg)
                    cor = {"aprovado": (0, 255, 0), "quarentena": (0, 255, 255), "reprovado": (0, 0, 255)}[status]

                    # Caixas + HUD para a thread principal desenhar no vídeo ao vivo
                    boxes = []
                    for box in resultado.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        classe = resultado.names.get(int(box.cls[0]), "?")
                        conf = float(box.conf[0])
                        boxes.append((int(x1), int(y1), int(x2), int(y2), f"{classe} {conf:.2f}"))
                    hud = [
                        f"Status: {status.upper()} | Clone: {cfg.clone_id}",
                        f"Saude: {confianca_saude:.1%} | Casca: {porcentagem_casca:.1f}%",
                        f"Altura: {altura_cm:.1f}cm | Densidade: {densidade:.0f}kg/m3 | Tortuos: {tortuosidade:.1f}",
                    ]
                    with lock:
                        estado["boxes"] = boxes
                        estado["hud"] = hud
                        estado["cor"] = cor

                    # --- 4. Grava no banco no máximo uma vez por intervalo ---
                    # (a inferência para exibição roda mais rápido que isso).
                    agora = time.monotonic()
                    if agora - ultima_gravacao >= cfg.intervalo_captura_seg:
                        ultima_gravacao = agora
                        indicadores = {
                            "densidade": {"valor": densidade, "unidade": "kg/m3", "metodo": f"lit_hibrido_grandis_urophylla_{cfg.clone_id}"},
                            "altura": {"valor": altura_cm, "unidade": "cm", "metodo": "imagem_gsd_ultrassom"},
                            "diametro": {"valor": diametro_cm, "unidade": "cm", "metodo": "imagem_gsd_ultrassom"},
                            "tortuosidade": {"valor": tortuosidade, "unidade": "indice", "metodo": "opencv_contorno"},
                            "porcentagem_casca": {"valor": porcentagem_casca, "unidade": "%", "metodo": "opencv_textura_hsv"},
                            "volume_util": {"valor": volume_util_m3, "unidade": "m3", "metodo": "geometria_medida"},
                            "massa_seca": {"valor": massa_seca_kg, "unidade": "kg", "metodo": f"volume_x_densidade_est_{cfg.clone_id}"},
                            "apodrecimento_pragas": {"valor": round(confianca_saude * 100, 2), "unidade": "%", "metodo": "yolo"},
                        }
                        uuid_gerado = salvar_inspecao(conexao, cfg, indicadores, defeitos, status, confianca_saude)

                        # Defeito que mais pesou na decisão — útil pra explicar o
                        # resultado ao operador (e ao avaliador, na demonstração)
                        if defeitos:
                            pior = max(defeitos, key=lambda d: d["confianca"] * severidade_do_defeito(d["tipo_defeito"]))
                            resumo_defeito = f"pior={pior['tipo_defeito']}({pior['confianca']:.2f})"
                        else:
                            resumo_defeito = "sem defeito"

                        emoji_status = {"aprovado": "✅", "quarentena": "⚠️", "reprovado": "❌"}[status]
                        print(
                            f"{emoji_status} [{uuid_gerado[:8]}] Clone={cfg.clone_id} status={status} "
                            f"saúde={confianca_saude:.2%} altura={altura_cm}cm diametro={diametro_cm}cm "
                            f"densidade={densidade}kg/m3 tortuosidade={tortuosidade} casca={porcentagem_casca}% "
                            f"volume={volume_util_m3}m3 massa={massa_seca_kg}kg "
                            f"defeitos={len(defeitos)} {resumo_defeito}"
                        )
                except Exception as e:
                    print(f"⚠️  Erro na análise (seguindo em frente): {e}")
                    time.sleep(0.05)
        finally:
            conexao.close()

    thread = threading.Thread(target=worker_analise, daemon=True)
    thread.start()

    mostrar = args.simulado or args.gui
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    print(
        "🚀 Rodando em tempo real. "
        + ("Tecle 'q' na janela ou " if mostrar else "")
        + "Ctrl+C para parar.\n"
    )

    try:
        while not parar.is_set():
            sucesso, frame = captura.read()
            if not sucesso:
                print("⚠️  Falha ao capturar frame. Tentando de novo...")
                time.sleep(0.1)
                continue

            # Publica o frame mais recente para o worker e pega o último resultado.
            with lock:
                estado["frame"] = frame
                boxes = list(estado["boxes"])
                hud = list(estado["hud"])
                cor = estado["cor"]

            if mostrar:
                vis = frame.copy()
                for (x1, y1, x2, y2, label) in boxes:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), cor, 2)
                    cv2.putText(vis, label, (x1, max(15, y1 - 6)), fonte, 0.5, cor, 1)
                for i, linha in enumerate(hud):
                    escala = 0.9 if i == 0 else (0.7 if i == 1 else 0.55)
                    cor_txt = cor if i == 0 else (255, 255, 255)
                    espessura = 2 if i <= 1 else 1
                    cv2.putText(vis, linha, (20, 40 + i * 33), fonte, escala, cor_txt, espessura)
                cv2.imshow("Omni-Root | John Deere Wood Inspection", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Sem janela: só mantém o worker alimentado sem ocupar 100% da CPU.
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n🛑 Encerrando por solicitação do usuário...")
    finally:
        parar.set()
        thread.join(timeout=2.0)
        captura.release()
        cv2.destroyAllWindows()
        print("✅ Recursos liberados. Até a próxima inspeção!")


if __name__ == "__main__":
    main()