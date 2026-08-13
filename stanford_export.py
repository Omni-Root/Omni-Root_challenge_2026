"""
stanford_export.py — Exportador StanForD 2010 (arquivo .hpr)
Projeto: Qualidade da Madeira — Challenge FIAP x John Deere/Suzano

O QUE ESTE SCRIPT FAZ
----------------------
Lê o banco SQLite local (omni_root_local.db, gerado pelo main.py) e gera um
arquivo .hpr (Harvested Production Report) por talhão, seguindo a estrutura
oficial do StanForD 2010 descrita na documentação pública da Skogforsk:

    HarvestedProduction
      Header            (metadados da mensagem)
      Machine           (identidade da máquina)
        ObjectDef       (talhão)
          Stem          (uma por tora inspecionada)
            Log
              LogMeasurement
            UserDefinedData     <- aqui entram os 4 indicadores de IA
                                   (densidade, altura, tortuosidade,
                                   apodrecimento_pragas) e os defeitos
                                   detectados pelo YOLO, usando o
                                   mecanismo OFICIAL do StanForD 2010 para
                                   dados customizados (não existe campo
                                   nativo para "densidade" ou
                                   "tortuosidade" no padrão).

IMPORTANTE — LEIA ANTES DE APRESENTAR
--------------------------------------
Este script gera um XML *estruturalmente compatível* com StanForD 2010,
baseado na documentação pública ("Introduction to StanForD 2010",
Skogforsk). Ele NÃO foi validado contra os XSDs oficiais, que são mantidos
pela Skogforsk e precisam ser baixados/solicitados em skogforsk.se para uma
validação formal. Para o Challenge, isso já é suficiente para demonstrar
que o sistema "fala o idioma" do padrão usado pelas máquinas John Deere
reais — mas não declare no pitch que é "certificado StanForD", porque não
passou por validação oficial. Se quiser ir além, valide o XML gerado contra
o XSD oficial (xmllint --schema stanford2010_hpr.xsd arquivo.hpr) antes de
qualquer uso em produção.

USO
---
    python3 stanford_export.py
    python3 stanford_export.py --db ./omni_root_local.db --out ./export_stanford
    python3 stanford_export.py --talhao "Talhão Demo"   # exporta só um talhão
    python3 stanford_export.py --somente-pendentes      # só o que sync_status=0
"""

import argparse
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

# ============================================================
# CONFIGURAÇÃO
# ============================================================

NAMESPACE = "urn:skogforsk:StanForD2010"

# Espécie/grupo de espécie default, usado quando o talhão não informa.
# No schema_sqlite.sql o talhao_id local é só um texto (nome/código), sem
# campo de espécie — no Postgres (tabela talhoes) já existe a coluna
# "especie". Se vocês quiserem, dá pra puxar isso do Postgres antes de
# exportar; por enquanto fica configurável aqui.
ESPECIE_PADRAO = "Eucalipto"

# Mapeamento status_classificacao (seu domínio) -> StemGrade (aproximação
# StanForD). O StanForD não tem um enum pronto para "aprovado/quarentena/
# reprovado" de IA de defeitos — por isso o valor completo e a confiança
# também vão para UserDefinedData, e aqui usamos só uma nota textual.
GRADE_POR_STATUS = {
    "aprovado": "OK",
    "quarentena": "REVISAO_MANUAL",
    "reprovado": "REJEITADO",
}


# ============================================================
# LEITURA DO SQLITE
# ============================================================

def conectar(db_path: str) -> sqlite3.Connection:
    conexao = sqlite3.connect(db_path)
    conexao.row_factory = sqlite3.Row
    return conexao


def buscar_talhoes(conexao: sqlite3.Connection, talhao_filtro: str | None) -> list[str]:
    query = "SELECT DISTINCT talhao_id FROM toras_local WHERE talhao_id IS NOT NULL"
    params: tuple = ()
    if talhao_filtro:
        query += " AND talhao_id = ?"
        params = (talhao_filtro,)
    linhas = conexao.execute(query, params).fetchall()
    return [linha["talhao_id"] for linha in linhas]


def buscar_toras_do_talhao(
    conexao: sqlite3.Connection, talhao_id: str, somente_pendentes: bool
) -> list[sqlite3.Row]:
    query = """
        SELECT id, uuid_local, maquina_id, talhao_id, log_id, data_inspecao,
               confianca_ia, status_classificacao, hash_sha256, sync_status
        FROM toras_local
        WHERE talhao_id = ?
    """
    params: tuple = (talhao_id,)
    if somente_pendentes:
        query += " AND sync_status = 0"
    query += " ORDER BY id ASC"
    return conexao.execute(query, params).fetchall()


def buscar_indicadores_da_tora(conexao: sqlite3.Connection, tora_id: int) -> list[sqlite3.Row]:
    return conexao.execute(
        """
        SELECT tipo_indicador, valor, unidade, metodo_medicao
        FROM indicadores_qualidade_local
        WHERE tora_id = ?
        ORDER BY id ASC
        """,
        (tora_id,),
    ).fetchall()


def buscar_defeitos_da_tora(conexao: sqlite3.Connection, tora_id: int) -> list[sqlite3.Row]:
    return conexao.execute(
        """
        SELECT tipo_defeito, pos_x, pos_y, largura, altura, confianca
        FROM defeitos_detectados_local
        WHERE tora_id = ?
        ORDER BY id ASC
        """,
        (tora_id,),
    ).fetchall()


# ============================================================
# CONSTRUÇÃO DO XML (estrutura StanForD 2010)
# ============================================================

def _sub(pai: Element, tag: str, texto=None) -> Element:
    """Atalho para criar um SubElement e opcionalmente já setar o texto."""
    elemento = SubElement(pai, tag)
    if texto is not None:
        elemento.text = str(texto)
    return elemento


def montar_header(raiz: Element, machine_user_id: str) -> None:
    header = SubElement(
        raiz,
        "Header",
        {
            "messageType": "hpr",
            "nation": "BR",
            "areaUnit": "ha",
            "diameterUnit": "mm",
            "lengthUnit": "cm",
            "volumeUnit": "m3sob",
            "weightUnit": "kg",
            "version": "draft_1.0",  # ver nota no docstring: não validado contra XSD oficial
        },
    )
    _sub(header, "MessageId", str(uuid.uuid4()))
    _sub(header, "Sender", "OmniRoot-Challenge-FIAP-JohnDeere-Suzano")
    _sub(header, "SaveDate", datetime.now(timezone.utc).isoformat(timespec="seconds"))


def montar_machine(raiz: Element, machine_user_id: str) -> Element:
    machine = SubElement(raiz, "Machine")
    # MachineKey deveria ser um GUID fixo, gerado uma única vez por máquina
    # (idealmente gravado em config, não recalculado a cada export). Aqui
    # derivamos de forma determinística do numero_serie só para manter
    # consistência entre exports sem exigir configuração extra.
    machine_key = str(uuid.uuid5(uuid.NAMESPACE_DNS, machine_user_id))
    _sub(machine, "MachineKey", machine_key)
    _sub(machine, "MachineUserId", machine_user_id)
    _sub(machine, "MachineInfo", "Raspberry Pi - YOLOv8m/NCNN - Demo Challenge")
    return machine


def montar_object_def(machine: Element, talhao_id: str, obj_key: int) -> Element:
    obj = SubElement(machine, "ObjectDef")
    _sub(obj, "ObjKey", obj_key)
    _sub(obj, "ObjUserId", talhao_id)
    _sub(obj, "ObjName", talhao_id)
    _sub(obj, "ForestOwner", "Suzano")
    return obj


def montar_stem(
    object_def: Element,
    tora: sqlite3.Row,
    stem_key: int,
    stem_number: int,
    indicadores: list[sqlite3.Row],
    defeitos: list[sqlite3.Row],
) -> None:
    stem = SubElement(object_def, "Stem")
    _sub(stem, "StmKey", stem_key)
    _sub(stem, "StmNumber", stem_number)
    _sub(stem, "SpcGrpUserId", ESPECIE_PADRAO)
    _sub(stem, "HarvDate", tora["data_inspecao"])

    # --- Log (a tora em si). Sem bucking real, é 1 log por stem. ---
    log = SubElement(stem, "Log")
    _sub(log, "LogKey", 1)
    _sub(log, "ProductKey", 1)  # produto único "genérico" — sem ProductDefinition detalhado

    altura = next((i for i in indicadores if i["tipo_indicador"] == "altura"), None)
    if altura is not None:
        medicao = SubElement(log, "LogMeasurement")
        _sub(medicao, "LogLength", altura["valor"])
        _sub(medicao, "LogMeasureUnit", altura["unidade"])

    grade = GRADE_POR_STATUS.get(tora["status_classificacao"], "INDEFINIDO")
    _sub(stem, "StemGrade", grade)

    # --- UserDefinedData: aqui entram os campos que o StanForD não tem
    #     nativamente. Segue o mecanismo oficial DataTableGroup/DataTable/
    #     Row/ColumnData (outputDataLocation = ProductionObject). ---
    udd = SubElement(stem, "UserDefinedData")
    grupo = SubElement(
        udd, "DataTableGroup",
        {"tableGroupId": "QualidadeMadeiraIA", "tableGroupName": "Indicadores de Qualidade (IA)"},
    )

    tabela_indicadores = SubElement(grupo, "DataTable", {"tableId": "indicadores_qualidade"})
    linha_indicadores = SubElement(tabela_indicadores, "Row", {"rowId": "1"})
    _sub(linha_indicadores, "ColumnData", tora["uuid_local"]).set("columnId", "uuid_local")
    _sub(linha_indicadores, "ColumnData", tora["log_id"]).set("columnId", "log_id_original")
    _sub(linha_indicadores, "ColumnData", tora["confianca_ia"]).set("columnId", "confianca_ia")
    _sub(linha_indicadores, "ColumnData", tora["status_classificacao"]).set(
        "columnId", "status_classificacao"
    )
    _sub(linha_indicadores, "ColumnData", tora["hash_sha256"]).set("columnId", "hash_sha256")
    for indicador in indicadores:
        campo = SubElement(linha_indicadores, "ColumnData")
        campo.set("columnId", indicador["tipo_indicador"])
        campo.set("unit", indicador["unidade"] or "")
        campo.set("metodoMedicao", indicador["metodo_medicao"])
        campo.text = str(indicador["valor"])

    if defeitos:
        tabela_defeitos = SubElement(grupo, "DataTable", {"tableId": "defeitos_detectados"})
        for indice, defeito in enumerate(defeitos, start=1):
            linha = SubElement(tabela_defeitos, "Row", {"rowId": str(indice)})
            _sub(linha, "ColumnData", defeito["tipo_defeito"]).set("columnId", "tipo_defeito")
            _sub(linha, "ColumnData", defeito["confianca"]).set("columnId", "confianca")
            _sub(linha, "ColumnData", defeito["pos_x"]).set("columnId", "pos_x")
            _sub(linha, "ColumnData", defeito["pos_y"]).set("columnId", "pos_y")
            _sub(linha, "ColumnData", defeito["largura"]).set("columnId", "largura")
            _sub(linha, "ColumnData", defeito["altura"]).set("columnId", "altura")


def montar_hpr(
    conexao: sqlite3.Connection, talhao_id: str, somente_pendentes: bool
) -> Element | None:
    toras = buscar_toras_do_talhao(conexao, talhao_id, somente_pendentes)
    if not toras:
        return None

    maquina_id = toras[0]["maquina_id"]

    raiz = Element("HarvestedProduction", {"xmlns": NAMESPACE})
    montar_header(raiz, maquina_id)
    machine = montar_machine(raiz, maquina_id)
    object_def = montar_object_def(machine, talhao_id, obj_key=1)

    for stem_number, tora in enumerate(toras, start=1):
        indicadores = buscar_indicadores_da_tora(conexao, tora["id"])
        defeitos = buscar_defeitos_da_tora(conexao, tora["id"])
        montar_stem(object_def, tora, stem_key=tora["id"], stem_number=stem_number,
                    indicadores=indicadores, defeitos=defeitos)

    return raiz


# ============================================================
# ESCRITA EM DISCO
# ============================================================

def salvar_xml(raiz: Element, caminho_saida: Path) -> None:
    xml_bruto = tostring(raiz, encoding="utf-8")
    xml_formatado = minidom.parseString(xml_bruto).toprettyxml(indent="  ", encoding="utf-8")
    caminho_saida.write_bytes(xml_formatado)


def nome_arquivo_seguro(talhao_id: str) -> str:
    seguro = "".join(c if c.isalnum() else "_" for c in talhao_id)
    return f"{seguro}.hpr"


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta o SQLite local para arquivos .hpr (StanForD 2010-inspired)"
    )
    parser.add_argument("--db", default="./omni_root_local.db", help="Caminho do SQLite local")
    parser.add_argument("--out", default="./export_stanford", help="Pasta de saída dos .hpr")
    parser.add_argument("--talhao", default=None, help="Exporta só este talhão (default: todos)")
    parser.add_argument(
        "--somente-pendentes", action="store_true",
        help="Exporta só toras com sync_status = 0 (ainda não sincronizadas com o Postgres)",
    )
    args = parser.parse_args()

    pasta_saida = Path(args.out)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    conexao = conectar(args.db)
    talhoes = [args.talhao] if args.talhao else buscar_talhoes(conexao, None)

    if not talhoes:
        print("⚠️  Nenhum talhão encontrado no banco local.")
        return

    total_arquivos = 0
    for talhao_id in talhoes:
        raiz = montar_hpr(conexao, talhao_id, args.somente_pendentes)
        if raiz is None:
            continue
        caminho = pasta_saida / nome_arquivo_seguro(talhao_id)
        salvar_xml(raiz, caminho)
        total_arquivos += 1
        print(f"[OK] Exportado: {caminho}")

    conexao.close()
    print(f"\n[StanForD] {total_arquivos} arquivo(s) .hpr gerado(s) em '{pasta_saida}/'.")
    print(
        "[Lembrete]: estrutura baseada na documentacao publica do StanForD 2010, "
        "nao validada contra o XSD oficial da Skogforsk. Ver docstring do arquivo."
    )


if __name__ == "__main__":
    main()
