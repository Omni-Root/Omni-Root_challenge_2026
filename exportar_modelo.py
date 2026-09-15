"""
exportar_modelo.py — Converte models/wood_best.pt para OpenVINO (CPU Intel)

Por que: o harvester não tem GPU. Medido neste notebook, no mesmo estado
térmico, por frame de 640x480:
    .pt no PyTorch @1024 ........ ~2.400 ms   (config antiga)
    .pt no PyTorch @640 ......... ~1.050 ms
    OpenVINO FP32 @640 .......... ~1.000 ms   (440 ms com a CPU fria)
    OpenVINO INT8 @640 .......... ~500 ms     <- padrão
Mesmos pesos, mesma rede — só o runtime e a precisão numérica mudam. O
INT8 é calibrado com imagens reais da operação (pasta capturas/, salvas
com a tecla 's' no main.py); nas capturas de teste as detecções ficaram
equivalentes às do FP32.

Rode UMA vez por máquina (o export é grande, ~170 MB, e fica fora do Git):

    python exportar_modelo.py                    # INT8, calibra com ./capturas
    python exportar_modelo.py --fp32             # sem quantizar (não precisa de imagens)
    python exportar_modelo.py --calib ./fotos    # outra pasta de calibração
    python exportar_modelo.py --formato ncnn     # CPU ARM (Raspberry etc.)

ATENÇÃO: o export tem tamanho de entrada FIXO. O `imgsz` do config.json
precisa ser o mesmo usado aqui — este script lê o config por padrão.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from main import carregar_configuracao_json, preprocessar_para_modelo

CLASSES = ["Quartzity", "Live_Knot", "Marrow", "resin", "Dead_Knot", "knot_with_crack", "Knot_missing", "Crack"]
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def montar_calibracao(pasta_fotos: Path, destino: Path) -> Path | None:
    """
    Monta um dataset YOLO mínimo para o NNCF calibrar o INT8: cada foto entra
    crua, pré-processada (grayscale + CLAHE — é isso que o modelo recebe em
    produção) e espelhada. Sem labels: calibração só olha as ativações.
    """
    fotos = sorted(p for p in pasta_fotos.iterdir() if p.suffix.lower() in EXTENSOES) if pasta_fotos.is_dir() else []
    if not fotos:
        return None
    imgs = destino / "images" / "val"
    imgs.mkdir(parents=True, exist_ok=True)
    (destino / "labels" / "val").mkdir(parents=True, exist_ok=True)
    for p in fotos:
        im = cv2.imread(str(p))
        if im is None:
            continue
        pp = preprocessar_para_modelo(im)
        cv2.imwrite(str(imgs / f"{p.stem}_raw.jpg"), im)
        cv2.imwrite(str(imgs / f"{p.stem}_pp.jpg"), pp)
        cv2.imwrite(str(imgs / f"{p.stem}_flip.jpg"), cv2.flip(pp, 1))
        cv2.imwrite(str(imgs / f"{p.stem}_flipv.jpg"), cv2.flip(pp, 0))
    yaml_path = destino / "data.yaml"
    yaml.safe_dump(
        {"path": str(destino.resolve()), "train": "images/val", "val": "images/val", "nc": len(CLASSES), "names": CLASSES},
        open(yaml_path, "w", encoding="utf-8"),
    )
    print(f"🖼️  Calibração INT8 com {len(fotos)} fotos de {pasta_fotos} (x4 variações). Quanto mais fotos reais, melhor.")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta o modelo treinado para um runtime de CPU")
    parser.add_argument("--pesos", default="./models/wood_best.pt")
    parser.add_argument("--formato", choices=["openvino", "ncnn", "onnx"], default="openvino")
    parser.add_argument("--imgsz", type=int, default=None, help="Padrão: imgsz do config.json")
    parser.add_argument("--fp32", action="store_true", help="Não quantizar (padrão: INT8 quando há fotos de calibração)")
    parser.add_argument("--calib", default="./capturas", help="Pasta com fotos reais para calibrar o INT8")
    parser.add_argument("--config-file", default="./config.json")
    args = parser.parse_args()

    cfg = carregar_configuracao_json(args.config_file)
    imgsz = args.imgsz or cfg.imgsz
    pesos = Path(args.pesos)
    if not pesos.exists():
        raise SystemExit(f"Pesos não encontrados: {pesos} (peça o wood_best.pt ao time e coloque em models/)")

    int8 = args.formato == "openvino" and not args.fp32
    tmp = Path(tempfile.mkdtemp(prefix="calib_"))
    try:
        data = montar_calibracao(Path(args.calib), tmp) if int8 else None
        if int8 and data is None:
            print(f"⚠️  Sem fotos em {args.calib} para calibrar — exportando FP32. (Salve frames com 's' no main.py e rode de novo.)")
            int8 = False
        print(f"📦 Exportando {pesos} -> {args.formato} @ {imgsz}px {'INT8' if int8 else 'FP32'} ...")
        saida = YOLO(str(pesos)).export(
            format=args.formato, imgsz=imgsz, half=False, dynamic=False,
            int8=int8, data=str(data) if data else None,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"✅ Pronto: {saida}")
    print(f"   O main.py usa esse modelo automaticamente (modelo_path no config.json: {cfg.modelo_path}).")
    if imgsz != cfg.imgsz:
        print(f"   ⚠️ imgsz do export ({imgsz}) é DIFERENTE do config.json ({cfg.imgsz}) — ajuste o config!")


if __name__ == "__main__":
    main()
