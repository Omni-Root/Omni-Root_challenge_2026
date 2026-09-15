"""
simular_cenario.py — Simulador end-to-end SEM câmera
Projeto: Qualidade da Madeira — Challenge FIAP x John Deere/Suzano

Roda o MESMO pipeline do main.py (`analisar_frame`) sobre imagens de
disco -- ou sobre frames sintéticos, se não houver imagem nenhuma -- e
grava cada inspeção no SQLite local, exatamente como a máquina faria.

Serve para:
  - testar mudanças no pipeline sem tora física nem webcam;
  - popular o SQLite (e, via sync_daemon.py, o Postgres/dashboard) com
    dados de demonstração produzidos pelo código real, não por seed.

USO (a partir da raiz do repositório):
    python tests/simular_cenario.py
    python tests/simular_cenario.py --num-toras 15 --pasta ./fotos_eucalipto
    python tests/simular_cenario.py --config-file ./config.json --clone GG100
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Permite `python tests/simular_cenario.py` a partir da raiz do repositório.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import (  # noqa: E402
    FiltroPersistencia,
    analisar_frame,
    carregar_configuracao_json,
    carregar_modelo,
    conectar_banco,
    resumir_analise,
    salvar_inspecao,
)

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def frame_sintetico(seed: int) -> np.ndarray:
    """Tora "desenhada": só para o pipeline ter o que processar sem imagem real."""
    rng = random.Random(seed)
    h, w = 720, 1280
    fundo = np.full((h, w, 3), (35, 60, 30), dtype=np.uint8)          # verde-terra
    cor_tora = (40, 90, 150)                                            # marrom (BGR)
    pt1 = (rng.randint(500, 560), 40)
    pt2 = (rng.randint(700, 780), h - 40)
    cv2.line(fundo, pt1, pt2, cor_tora, thickness=rng.randint(180, 260))
    if rng.random() > 0.4:                                              # mancha escura = "nó"
        cx = (pt1[0] + pt2[0]) // 2 + rng.randint(-30, 30)
        cy = h // 2 + rng.randint(-150, 150)
        cv2.circle(fundo, (cx, cy), rng.randint(15, 35), (20, 40, 70), -1)
    return fundo


def obter_frames(pasta: Path | None, num: int) -> list[tuple[str, np.ndarray]]:
    frames: list[tuple[str, np.ndarray]] = []
    if pasta is not None and pasta.exists():
        arquivos = sorted(p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES)
        if arquivos:
            for p in random.sample(arquivos, min(num, len(arquivos))):
                img = cv2.imread(str(p))
                if img is not None:
                    frames.append((p.name, img))
    i = 0
    while len(frames) < num:
        frames.append((f"sintetico_{i:02d}", frame_sintetico(i)))
        i += 1
    return frames[:num]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulador de inspeção de madeira sem câmera")
    parser.add_argument("--num-toras", type=int, default=10)
    parser.add_argument("--pasta", type=str, default="./dataset/wood_yolo/images/val",
                        help="Pasta com imagens de teste (se vazia/inexistente, gera frames sintéticos)")
    parser.add_argument("--config-file", type=str, default="./config.json")
    parser.add_argument("--clone", type=str, default=None)
    parser.add_argument("--sem-banco", action="store_true", help="Só analisa, não grava no SQLite")
    args = parser.parse_args()

    cfg = carregar_configuracao_json(args.config_file)
    if args.clone:
        cfg.clone_id = args.clone

    print("=" * 70)
    print("🌲 SIMULADOR DE INSPEÇÃO — mesmo pipeline do main.py, sem câmera")
    print("=" * 70)
    modelo = carregar_modelo(cfg)
    conexao = None if args.sem_banco else conectar_banco(cfg)
    persistencia = FiltroPersistencia(min_ocorrencias=max(1, cfg.filtro_persistencia))

    frames = obter_frames(Path(args.pasta) if args.pasta else None, args.num_toras)
    print(f"📸 {len(frames)} frames ({sum(1 for n, _ in frames if not n.startswith('sintetico'))} reais)\n")

    resumo = {"aprovado": 0, "quarentena": 0, "reprovado": 0}
    volume_total = 0.0
    massa_total = 0.0
    for idx, (nome, frame) in enumerate(frames, start=1):
        # Persistência exige o defeito em N análises seguidas: com a peça
        # "parada" na frente da câmera, cada frame é analisado N vezes,
        # como aconteceria ao vivo.
        persistencia.reset()
        a = None
        for _ in range(max(1, cfg.filtro_persistencia)):
            a = analisar_frame(frame, modelo, cfg, persistencia)
        assert a is not None

        uuid_gerado = "sem-banco"
        if conexao is not None:
            uuid_gerado = salvar_inspecao(conexao, cfg, a["indicadores"], a["defeitos"], a["status"], a["confianca_saude"])

        resumo[a["status"]] += 1
        volume_total += a["indicadores"]["volume_util"]["valor"]
        massa_total += a["indicadores"]["massa_seca"]["valor"]
        print(f"#{idx:02d} {nome:<28} " + resumir_analise(uuid_gerado, cfg, a))

    if conexao is not None:
        conexao.close()

    n = len(frames)
    print("\n" + "=" * 70)
    print("📊 RESUMO")
    print("=" * 70)
    for status, emoji in (("aprovado", "✅"), ("quarentena", "⚠️"), ("reprovado", "❌")):
        print(f"{emoji} {status:<11}: {resumo[status]:>3} ({resumo[status] / n:.0%})")
    print(f"🪵 Volume útil total : {volume_total:.3f} m³")
    print(f"⚖️  Massa seca total  : {massa_total:.1f} kg (estimada: volume medido x densidade do clone {cfg.clone_id})")
    if conexao is not None:
        print(f"💾 Gravado em        : {cfg.sqlite_path} (rode sync_daemon.py para enviar ao Postgres)")


if __name__ == "__main__":
    main()
