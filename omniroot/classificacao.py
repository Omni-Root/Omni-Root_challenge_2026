"""
classificacao.py — Regra de negócio: severidade dos defeitos, saúde da tora, classificação (aprovado/quarentena/reprovado) e hash.

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

import hashlib
import json

from omniroot.config import Config


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
