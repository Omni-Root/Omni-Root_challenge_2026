"""
testar_densidade_clones.py

Verifica o caminho COMPLETO da densidade, sem precisar de câmera,
modelo YOLO ou tora nenhuma:

    PostgreSQL (tabela clones_densidade)
        |  (sync_daemon.py baixa)
        v
    data/clones_densidade.json  (cache local)
        |  (main.py lê)
        v
    calcular_densidade_estimada()  <- o que a IA usa de verdade

USO:
    # 1. rode o sync uma vez pra baixar a tabela (Ctrl+C depois da
    #    mensagem "Tabela de densidade atualizada")
    python sync_daemon.py

    # 2. rode este script pra conferir se a IA está lendo os valores
    python tests/testar_densidade_clones.py

Não precisa de webcam, não precisa de modelo treinado, não precisa
inserir tora nenhuma.
"""

import json
import sys
import types
from pathlib import Path

# O main.py importa cv2 e ultralytics no topo. Como este teste só quer
# a lógica de densidade (não a de visão), substituímos esses dois por
# módulos vazios -- assim o script roda mesmo em máquina sem o modelo
# instalado/baixado.
try:
    import cv2  # noqa: F401  (se estiver instalado, usa o real)
except ImportError:
    # MagicMock aceita qualquer chamada (main.py cria um CLAHE no import)
    from unittest.mock import MagicMock
    sys.modules["cv2"] = MagicMock()
_ultra = types.ModuleType("ultralytics")
_ultra.YOLO = lambda *a, **k: None
sys.modules.setdefault("ultralytics", _ultra)

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

CAMINHO_CACHE = RAIZ / "data" / "clones_densidade.json"

# Os 11 clones que o contato da John Deere passou na mentoria
CLONES_JD = ["I144", "GG100", "VM01 (VM1)", "I042", "58", "386",
             "H13", "H15", "H17", "H19", "AEC 0144"]


def main() -> None:
    print("=" * 66)
    print("1. ESTADO DO CACHE LOCAL (data/clones_densidade.json)")
    print("=" * 66)

    if not CAMINHO_CACHE.exists():
        print(f"❌ Arquivo não encontrado: {CAMINHO_CACHE}")
        print("   Rode o main.py ao menos uma vez, ou o sync_daemon.py.")
        return

    with open(CAMINHO_CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    sincronizado_em = cache.get("_sincronizado_em")
    if sincronizado_em:
        print(f"✅ Cache SINCRONIZADO do Postgres em: {sincronizado_em}")
        print(f"   Total de clones no cache: {cache.get('_total_clones', '?')}")
    else:
        print("⚠️  Este cache ainda é o arquivo ORIGINAL do repositório")
        print("   (não tem '_sincronizado_em'), ou seja: o sync_daemon.py")
        print("   ainda não baixou a tabela do Postgres.")
        print("   -> rode 'python sync_daemon.py' e espere a mensagem")
        print("      '📥 Tabela de densidade atualizada'.")

    # Conta por procedência do dado
    por_tipo: dict = {}
    for chave, valor in cache.items():
        if chave.startswith("_") or not isinstance(valor, dict):
            continue
        tipo = valor.get("tipo_dado", "(sem tipo)")
        por_tipo[tipo] = por_tipo.get(tipo, 0) + 1
    if por_tipo:
        print("\n   Procedência dos dados no cache:")
        for tipo, qtd in sorted(por_tipo.items()):
            print(f"     - {tipo}: {qtd} clone(s)")

    print()
    print("=" * 66)
    print("2. O QUE A IA (main.py) LÊ PARA CADA CLONE DA JOHN DEERE")
    print("=" * 66)

    import main as m  # noqa: E402  (import tardio de propósito, ver stubs acima)

    print(f"{'CLONE':<14} {'DENSIDADE':>12}   PROCEDÊNCIA")
    print("-" * 66)

    faltando = []
    for clone in CLONES_JD:
        info = m.INVENTARIO_FLORESTAL.get(clone.upper())

        # O main.py imprime um aviso por clone sem densidade. Aqui isso
        # poluiria a tabela, então silenciamos a saída dele durante a
        # chamada -- a informação já aparece na coluna PROCEDÊNCIA.
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            densidade = m.calcular_densidade_estimada(clone)

        if info is None:
            procedencia = "❌ NÃO ESTÁ NO CACHE"
            faltando.append(clone)
        else:
            bruto = cache.get(clone, {})
            procedencia = bruto.get("tipo_dado", "(sem tipo)")
            if bruto.get("densidade_min") is not None:
                procedencia += f"  (faixa {bruto['densidade_min']:.0f}-{bruto['densidade_max']:.0f})"

        print(f"{clone:<14} {densidade:>9.1f} kg/m3   {procedencia}")

    print()
    print("=" * 66)
    print("3. RESULTADO")
    print("=" * 66)

    if faltando:
        print(f"❌ {len(faltando)} clone(s) não chegaram no cache: {', '.join(faltando)}")
        print("   Confira se a migration rodou no Postgres:")
        print("     SELECT COUNT(*) FROM clones_densidade;   -- esperado: 17")
    elif not sincronizado_em:
        print("⚠️  Os clones estão no arquivo, mas ele ainda não veio do Postgres.")
        print("   O caminho Postgres -> cache ainda não foi validado.")
    else:
        print("✅ Caminho completo validado:")
        print("   Postgres -> sync_daemon.py -> cache JSON -> main.py")
        print("   A IA está lendo os valores de densidade vindos do banco central.")
        print()
        print("   Teste extra: mude um valor no Postgres e rode o sync de novo --")
        print("   o número aqui deve mudar junto, sem tocar em nenhum arquivo.")
        print("     UPDATE clones_densidade SET densidade_base = 505")
        print("      WHERE clone_id = 'GG100';")


if __name__ == "__main__":
    main()
