"""
inventario.py — Densidade por clone: cache local sincronizado do Postgres (recarga automática) e dados dendrométricos.

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

import json
import threading
import time
from pathlib import Path

import numpy as np


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
        O arquivo é um CACHE gravado pelo sync_daemon.py a partir da tabela
        clones_densidade do Postgres (fonte de verdade); cada clone traz
        'tipo_dado' e 'fonte'. Não editar à mão. A recarga em tempo de
        execução é feita por inventario_atual().

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


# ------------------------------------------------------------
# RECARGA AUTOMÁTICA DA TABELA DE DENSIDADE
# ------------------------------------------------------------
# O sync_daemon.py regrava data/clones_densidade.json toda vez que baixa a
# tabela do Postgres central. Antes, o main.py lia o arquivo UMA vez no
# import: cadastrar a densidade de um clone no banco só valia depois de
# reiniciar a máquina -- justamente a demonstração que a JD pediu. Agora
# `inventario_atual()` confere o mtime/tamanho dos JSONs (um stat, barato)
# no máximo 1x por segundo e recarrega quando mudaram. O sync grava por
# arquivo temporário + replace (atômico), então nunca lemos JSON pela metade;
# se mesmo assim a leitura falhar, fica o cache anterior.
# ------------------------------------------------------------
_ARQUIVOS_INVENTARIO = ("./data/clones_densidade.json", "./data/inventario_johndeere.json")
_INVENTARIO = {"dados": {}, "assinatura": None, "checado_em": -1e9}
_INVENTARIO_LOCK = threading.Lock()
_AVISOS_CLONE_SEM_DENSIDADE: set = set()


def _assinatura_inventario() -> tuple:
    saida = []
    for caminho in _ARQUIVOS_INVENTARIO:
        p = Path(caminho)
        try:
            st = p.stat()
            saida.append((st.st_mtime_ns, st.st_size))
        except OSError:
            saida.append(None)
    return tuple(saida)


def inventario_atual(intervalo_checagem_s: float = 1.0) -> dict:
    """Dicionário por clone, recarregado sozinho quando os JSONs mudam em disco."""
    global INVENTARIO_FLORESTAL
    with _INVENTARIO_LOCK:
        agora = time.monotonic()
        if agora - _INVENTARIO["checado_em"] < intervalo_checagem_s:
            return _INVENTARIO["dados"]
        _INVENTARIO["checado_em"] = agora

        assinatura = _assinatura_inventario()
        if assinatura == _INVENTARIO["assinatura"]:
            return _INVENTARIO["dados"]
        try:
            novo = carregar_inventario_florestal(*_ARQUIVOS_INVENTARIO)
        except Exception as e:
            print(f"⚠️  Não consegui reler a tabela de densidade ({e}); mantendo a anterior.")
            return _INVENTARIO["dados"]

        antigo = _INVENTARIO["dados"]
        if _INVENTARIO["assinatura"] is not None:
            # Não é a primeira carga: diz o que mudou, para a demo ficar visível
            mudancas = []
            for clone, info in novo.items():
                d_novo = info.get("densidade_base")
                d_antigo = (antigo.get(clone) or {}).get("densidade_base")
                if clone not in antigo:
                    mudancas.append(f"{clone}: novo ({d_novo} kg/m3)")
                elif d_novo != d_antigo:
                    mudancas.append(f"{clone}: {d_antigo} -> {d_novo} kg/m3")
            for clone in antigo.keys() - novo.keys():
                mudancas.append(f"{clone}: removido")
            # O sync regrava o arquivo a cada ciclo (timestamp novo) mesmo sem
            # mudança: só fala quando algum VALOR mudou, senão vira spam.
            if mudancas:
                resumo = "; ".join(mudancas[:6]) + (" ..." if len(mudancas) > 6 else "")
                print(f"📥 Tabela de densidade recarregada do disco ({len(novo)} clones). Mudanças: {resumo}")
                _AVISOS_CLONE_SEM_DENSIDADE.clear()  # se o clone foi cadastrado, o aviso some; se não, avisa de novo

        _INVENTARIO["dados"] = novo
        _INVENTARIO["assinatura"] = assinatura
        INVENTARIO_FLORESTAL = novo
        return novo


INVENTARIO_FLORESTAL = inventario_atual()


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
    inv = inventario_stats if inventario_stats is not None else inventario_atual()
    key = str(clone_id).upper()
    info = inv.get(key)

    if not info or info.get("densidade_base") is None:
        # Avisa UMA vez por clone (a análise roda ~10x/s); volta a avisar se
        # a tabela for recarregada e o clone continuar sem densidade.
        if key not in _AVISOS_CLONE_SEM_DENSIDADE:
            _AVISOS_CLONE_SEM_DENSIDADE.add(key)
            print(
                f"⚠️  Sem densidade cadastrada para o clone '{clone_id}'. Usando média "
                f"genérica de eucalipto (500.0 kg/m3). Para corrigir, cadastre o valor "
                f"na tabela 'clones_densidade' do PostgreSQL central (fonte de verdade) "
                f"— o sync_daemon.py traz a atualização automaticamente na próxima "
                f"sincronização, e o main.py recarrega sozinho, sem reiniciar."
            )
        return 500.0

    return round(float(info["densidade_base"]), 1)
