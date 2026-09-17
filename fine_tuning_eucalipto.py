"""
fine_tuning_eucalipto.py — Especializa o modelo atual no domínio da demo

Parte do checkpoint já treinado (models/wood_best.pt: 8 classes, madeira
serrada) e continua o treino com o dataset montado por
preparar_dataset_eucalipto.py (fotos próprias revisadas + negativos +
rehearsal do dataset original). Não é um treino do zero: é adaptação de
domínio, então

  - `freeze=10` congela o backbone (as primeiras 10 camadas): com poucas
    centenas de fotos, deixar tudo livre só decora o dataset;
  - `lr0` baixo e `warmup` curto: os pesos já são bons, é ajuste fino;
  - `imgsz=640`: o MESMO tamanho da inferência (export OpenVINO/config.json).
    O treino original foi a 1024 e a inferência a 640 — aqui alinhamos;
  - mosaic reduzido e sem mixup/copy_paste: textura de madeira misturada
    vira defeito falso.

Ao final, valida em DOIS conjuntos: o novo (eucalipto) e o original
(dataset/wood_yolo/val), para mostrar que o modelo aprendeu o domínio novo
SEM esquecer o antigo — é isso que se apresenta para a banca.

USO:
    python fine_tuning_eucalipto.py --data ./dataset/eucalipto_yolo/data.yaml
    python fine_tuning_eucalipto.py --data ... --epochs 40 --modelo yolov8s.pt   # comparar um modelo menor
    python fine_tuning_eucalipto.py --data ... --sem-freeze                        # se o dataset for grande (> 2k fotos)

Depois:
    python exportar_modelo.py --pesos ./models/wood_eucalipto_best.pt
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning de eucalipto a partir do checkpoint atual")
    parser.add_argument("--data", required=True, help="data.yaml gerado por preparar_dataset_eucalipto.py")
    parser.add_argument("--pesos", default="./models/wood_best.pt", help="checkpoint de partida")
    parser.add_argument("--modelo", default=None, help="em vez de --pesos, parte de um modelo genérico (ex.: yolov8s.pt) para comparar tamanhos")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = AutoBatch na GPU; em CPU use 4-8")
    parser.add_argument("--lr0", type=float, default=2e-4)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--sem-freeze", action="store_true")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--base-val", default="./dataset/wood_yolo/data.yaml", help="data.yaml do dataset original, para medir esquecimento")
    parser.add_argument("--nome", default="wood_eucalipto")
    parser.add_argument("--saida", default="./models/wood_eucalipto_best.pt")
    parser.add_argument("--runs", default="./runs")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  Sem GPU: vai treinar em CPU (lento). Se puder, rode na máquina com a RTX. Sugestão: --batch 4 --epochs 15.")
        if args.batch == -1:
            args.batch = 4

    origem = args.modelo or args.pesos
    if not args.modelo and not Path(args.pesos).exists():
        raise SystemExit(f"Checkpoint não encontrado: {args.pesos} (peça o wood_best.pt ao time)")
    print(f"📦 Partindo de {origem} | dados {args.data} | {args.epochs} épocas @ {args.imgsz}px | device {device}")

    modelo = YOLO(origem)
    resultado = modelo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        freeze=0 if args.sem_freeze else args.freeze,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.05,
        warmup_epochs=1.0,
        cos_lr=True,
        patience=args.patience,
        # Augmentações contidas: é adaptação, não treino do zero
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.4,
        degrees=10.0, flipud=0.5, fliplr=0.5,
        mosaic=0.5, close_mosaic=8,
        mixup=0.0, copy_paste=0.0,
        project=args.runs,
        name=args.nome,
        exist_ok=True,
        plots=True,
        verbose=True,
        seed=42,
    )

    melhor = Path(resultado.save_dir) / "weights" / "best.pt"
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(melhor, args.saida)
    print(f"\n✅ Melhor checkpoint copiado para {args.saida}")

    # --- Validação dupla: domínio novo x domínio original ---
    fino = YOLO(args.saida)
    print("\n📈 Validação no dataset NOVO (eucalipto + negativos):")
    v_novo = fino.val(data=args.data, imgsz=args.imgsz, device=device, plots=False, verbose=False)
    print(f"   mAP50 {v_novo.box.map50:.3f} | mAP50-95 {v_novo.box.map:.3f} | P {v_novo.box.mp:.3f} | R {v_novo.box.mr:.3f}")

    if Path(args.base_val).exists() and not args.modelo:
        print("📈 Validação no dataset ORIGINAL (mede esquecimento; compare com o modelo antigo):")
        v_base = fino.val(data=args.base_val, imgsz=args.imgsz, device=device, plots=False, verbose=False)
        print(f"   mAP50 {v_base.box.map50:.3f} | mAP50-95 {v_base.box.map:.3f} | P {v_base.box.mp:.3f} | R {v_base.box.mr:.3f}")
        antigo = YOLO(args.pesos)
        v_ant = antigo.val(data=args.base_val, imgsz=args.imgsz, device=device, plots=False, verbose=False)
        print(f"   (modelo antigo no mesmo conjunto: mAP50 {v_ant.box.map50:.3f} | mAP50-95 {v_ant.box.map:.3f})")

    print(f"\nPróximo passo: python exportar_modelo.py --pesos {args.saida}")
    print("e apontem modelo_path no config.json para a pasta *_openvino_model gerada.")


if __name__ == "__main__":
    main()
