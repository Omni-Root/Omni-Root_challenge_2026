"""
preparar_dataset_eucalipto.py — Fabrica o dataset de fine-tuning no domínio da demo

Não existe dataset público de tora de eucalipto com defeito anotado (procuramos:
o que há é madeira serrada europeia — o que o modelo atual viu — e datasets de
tora inteira para máquina florestal, sem defeito). O caminho realista é montar
um dataset PEQUENO, no domínio exato em que o sistema vai rodar (as peças, a
câmera, a luz e o fundo da maquete), e fine-tunar o checkpoint atual em cima.

Três ingredientes, e o script cuida dos três:

  1. FOTOS DE MADEIRA (poucas centenas bastam): peças/toras que vocês têm,
     fotografadas pela MESMA webcam do main.py (tecla 's' salva em capturas/).
     Variem: seção e lateral, com/sem casca, com/sem rachadura, luz, ângulo,
     distância. O modelo atual PRÉ-ANOTA cada foto (conf baixa); vocês só
     corrigem no Roboflow/CVAT/LabelImg em vez de anotar do zero.

  2. NEGATIVOS (fotos SEM madeira, label vazio): mão, teclado, mesa, garra
     vazia, parede, chão. É a forma documentada pelo Ultralytics de ensinar
     "isso não é defeito" — resolve o "detectou defeito na mão" na raiz,
     não só nos filtros. 5-10% do dataset.

  3. REHEARSAL do dataset original (amostra do Kaggle já preparado em
     dataset/wood_yolo): evita que o modelo esqueça as 8 classes enquanto
     aprende a textura nova (catastrophic forgetting).

USO — dois passos:

  # 1) pré-anotar as fotos próprias (gera labels-rascunho + prévias para revisar)
  python preparar_dataset_eucalipto.py pre-anotar --fotos ./fotos_eucalipto --saida ./anotacoes

     -> revisem anotacoes/labels/*.txt (formato YOLO) numa ferramenta de anotação;
        as prévias em anotacoes/previas/ mostram o que o modelo chutou.

  # 2) montar o dataset final (CLAHE igual ao treino, split, data.yaml)
  python preparar_dataset_eucalipto.py montar --anotadas ./anotacoes --negativos ./fotos_negativas \
      --base ./dataset/wood_yolo --rehearsal 1500 --saida ./dataset/eucalipto_yolo

  # 3) treinar
  python fine_tuning_eucalipto.py --data ./dataset/eucalipto_yolo/data.yaml
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import yaml

from main import carregar_configuracao_json, carregar_modelo, preprocessar_para_modelo

CLASSES = ["Quartzity", "Live_Knot", "Marrow", "resin", "Dead_Knot", "knot_with_crack", "Knot_missing", "Crack"]
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CORES = [(255, 128, 0), (0, 200, 0), (200, 0, 200), (0, 128, 255), (0, 0, 255), (0, 200, 255), (128, 0, 255), (255, 0, 128)]


def listar_imagens(pasta: Path) -> list[Path]:
    if not pasta or not pasta.is_dir():
        return []
    return sorted(p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES)


# ============================================================
# PASSO 1 — pré-anotação com o modelo atual
# ============================================================

def pre_anotar(args) -> None:
    fotos = listar_imagens(Path(args.fotos))
    if not fotos:
        sys.exit(f"Nenhuma imagem em {args.fotos}")
    saida = Path(args.saida)
    (saida / "images").mkdir(parents=True, exist_ok=True)
    (saida / "labels").mkdir(parents=True, exist_ok=True)
    (saida / "previas").mkdir(parents=True, exist_ok=True)

    cfg = carregar_configuracao_json(args.config_file)
    if args.pesos:
        cfg.modelo_path = args.pesos
    modelo = carregar_modelo(cfg)

    total_caixas = 0
    for p in fotos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        # Mesmo pré-processamento da inferência em produção.
        res = modelo.predict(source=preprocessar_para_modelo(img), conf=args.conf, iou=0.5,
                             imgsz=cfg.imgsz, device="cpu", verbose=False)[0]
        linhas = []
        previa = img.copy()
        for box in res.boxes:
            c = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            linhas.append(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            cor = CORES[c % len(CORES)]
            cv2.rectangle(previa, (int(x1), int(y1)), (int(x2), int(y2)), cor, 2)
            cv2.putText(previa, f"{CLASSES[c]} {float(box.conf[0]):.2f}", (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA)
        total_caixas += len(linhas)
        destino = saida / "images" / (p.stem + ".jpg")
        cv2.imwrite(str(destino), img)
        (saida / "labels" / (p.stem + ".txt")).write_text("\n".join(linhas), encoding="utf-8")
        cv2.imwrite(str(saida / "previas" / (p.stem + ".jpg")), previa)

    (saida / "classes.txt").write_text("\n".join(CLASSES), encoding="utf-8")
    print(f"✅ {len(fotos)} fotos pré-anotadas em {saida} ({total_caixas} caixas-rascunho, conf >= {args.conf}).")
    print("   Revisem labels/ (YOLO txt; classes.txt tem a ordem) — apaguem caixas erradas, ajustem, adicionem as que faltam.")
    print("   As prévias em previas/ mostram o chute do modelo. Foto sem defeito = label vazio (é informação também).")


# ============================================================
# PASSO 2 — montar o dataset final
# ============================================================

def _copiar_par(img: Path, label: Path | None, destino_img: Path, destino_lbl: Path, clahe: bool) -> None:
    im = cv2.imread(str(img))
    if im is None:
        return
    if clahe:
        im = preprocessar_para_modelo(im)  # gray + CLAHE, igual ao treino original
    cv2.imwrite(str(destino_img), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    texto = label.read_text(encoding="utf-8") if label and label.exists() else ""
    destino_lbl.write_text(texto.strip(), encoding="utf-8")


def _split(itens: list, fracao_val: float, rng: random.Random) -> tuple[list, list]:
    itens = list(itens)
    rng.shuffle(itens)
    n_val = max(1, int(round(len(itens) * fracao_val))) if len(itens) >= 4 else 0
    return itens[n_val:], itens[:n_val]


def montar(args) -> None:
    rng = random.Random(args.seed)
    saida = Path(args.saida)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (saida / sub).mkdir(parents=True, exist_ok=True)

    grupos: dict[str, list[tuple[Path, Path | None]]] = {}

    # a) fotos próprias, já revisadas
    anot = Path(args.anotadas)
    proprias = listar_imagens(anot / "images")
    grupos["eucalipto"] = [(p, anot / "labels" / (p.stem + ".txt")) for p in proprias]

    # b) negativos: label vazio
    grupos["negativo"] = [(p, None) for p in listar_imagens(Path(args.negativos))] if args.negativos else []

    # c) rehearsal do dataset original (já está em gray+CLAHE, não reaplica)
    grupos["base"] = []
    if args.base and args.rehearsal > 0:
        base = Path(args.base)
        cand = listar_imagens(base / "images" / "train")
        rng.shuffle(cand)
        for p in cand[: args.rehearsal]:
            lbl = base / "labels" / "train" / (p.stem + ".txt")
            grupos["base"].append((p, lbl if lbl.exists() else None))

    if not grupos["eucalipto"]:
        sys.exit(f"Nenhuma foto anotada em {anot / 'images'} — rode o passo pre-anotar primeiro.")

    resumo = {}
    for nome, itens in grupos.items():
        treino, val = _split(itens, args.val, rng)
        for split, lista in (("train", treino), ("val", val)):
            for img, lbl in lista:
                dst_img = saida / "images" / split / f"{nome}_{img.stem}.jpg"
                dst_lbl = saida / "labels" / split / f"{nome}_{img.stem}.txt"
                _copiar_par(img, lbl, dst_img, dst_lbl, clahe=(nome != "base"))
        resumo[nome] = (len(treino), len(val))

    data = {
        "path": str(saida.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    with open(saida / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"✅ Dataset em {saida}")
    for nome, (t, v) in resumo.items():
        print(f"   {nome:<10} treino {t:>5}  val {v:>4}")
    total = sum(t + v for t, v in resumo.values())
    neg = sum(resumo["negativo"])
    print(f"   total {total} imagens — negativos {100.0 * neg / max(1, total):.0f}% (alvo 5-10%)")
    print(f"   data.yaml: {saida / 'data.yaml'}")
    print("Próximo passo: python fine_tuning_eucalipto.py --data", saida / "data.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta o dataset de fine-tuning de eucalipto")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pre-anotar", help="pré-anota fotos próprias com o modelo atual")
    p1.add_argument("--fotos", required=True, help="pasta com as fotos (webcam do main.py, tecla 's')")
    p1.add_argument("--saida", default="./anotacoes")
    p1.add_argument("--conf", type=float, default=0.25, help="conf baixa de propósito: melhor apagar caixa do que desenhar")
    p1.add_argument("--pesos", default=None, help="padrão: modelo do config.json (OpenVINO) — ou ./models/wood_best.pt")
    p1.add_argument("--config-file", default="./config.json")
    p1.set_defaults(func=pre_anotar)

    p2 = sub.add_parser("montar", help="monta o dataset final (fotos revisadas + negativos + rehearsal)")
    p2.add_argument("--anotadas", default="./anotacoes", help="saída do pre-anotar, já revisada")
    p2.add_argument("--negativos", default=None, help="pasta com fotos SEM madeira (mão, teclado, mesa, garra vazia)")
    p2.add_argument("--base", default="./dataset/wood_yolo", help="dataset original preparado (rehearsal)")
    p2.add_argument("--rehearsal", type=int, default=1500, help="quantas imagens do dataset original misturar (0 desliga)")
    p2.add_argument("--val", type=float, default=0.15)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--saida", default="./dataset/eucalipto_yolo")
    p2.set_defaults(func=montar)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
