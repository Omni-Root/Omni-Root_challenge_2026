"""
testar_modelo_eucalipto.py

Roda o modelo YOLO atual (o mesmo que o main.py carregaria) contra uma
PASTA DE IMAGENS, em vez de abrir a webcam. Feito pra testar rápido se o
modelo generaliza pra fotos reais de eucalipto, sem precisar mexer no
main.py nem ter a câmera/tora física em mãos.

USO:
    python testar_modelo_eucalipto.py --pasta ./fotos_eucalipto

    (opcional, pra comparar nano vs small na mesma pasta)
    python testar_modelo_eucalipto.py --pasta ./fotos_eucalipto --modelo ./models/wood_best_nano.pt
    python testar_modelo_eucalipto.py --pasta ./fotos_eucalipto --modelo ./models/wood_best_small.pt

SAÍDA:
    - Imagens anotadas (com as caixas/labels desenhadas) em ./resultados_teste/
    - Um resumo no console: arquivo, classe detectada, confiança — e destaque
      pras imagens SEM nenhuma detecção (pode ser tão importante quanto falso
      positivo, se uma tora real não for detectada) e pras detecções com
      confiança baixa (podem ser os "quase-cabelo" que vocês querem pegar).

Não precisa de config.json nem banco de dados — é só pra inspeção visual
rápida do modelo, isolado do resto do pipeline.
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolver_caminho_modelo(caminho_informado: str | None) -> str:
    """Mesma lógica de resolução de caminho que o main.py usa, pra garantir
    que esse teste carrega o MESMO modelo que rodaria em produção."""
    if caminho_informado:
        return caminho_informado

    for p in [
        Path("../models/wood_ncnn_model"),
        Path("../models/wood_best.pt"),
        Path("../yolov8n.pt"),
    ]:
        if p.exists():
            return str(p)
    return "yolov8n.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Testa o modelo YOLO numa pasta de imagens de eucalipto.")
    parser.add_argument("--pasta", required=True, help="Pasta com as fotos de teste (jpg/png).")
    parser.add_argument("--modelo", default=None, help="Caminho do modelo .pt/ncnn. Se omitido, usa a mesma resolução do main.py.")
    parser.add_argument("--conf", type=float, default=0.40, help="Confiança mínima (padrão igual ao config.json: 0.40).")
    parser.add_argument("--iou", type=float, default=0.55, help="IoU threshold (padrão igual ao config.json: 0.55).")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho de imagem pro modelo (padrão: 640).")
    parser.add_argument("--saida", default="./resultados_teste", help="Pasta onde salvar as imagens anotadas.")
    args = parser.parse_args()

    pasta_entrada = Path(args.pasta)
    if not pasta_entrada.exists():
        raise SystemExit(f"❌ Pasta não encontrada: {pasta_entrada}")

    pasta_saida = Path(args.saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    caminho_modelo = resolver_caminho_modelo(args.modelo)
    print(f"📦 Carregando modelo: {caminho_modelo}")
    modelo = YOLO(caminho_modelo)

    imagens = sorted(
        f for f in pasta_entrada.iterdir()
        if f.suffix.lower() in EXTENSOES_VALIDAS
    )
    if not imagens:
        raise SystemExit(f"❌ Nenhuma imagem encontrada em {pasta_entrada} (extensões aceitas: {EXTENSOES_VALIDAS})")

    print(f"🖼️  {len(imagens)} imagem(ns) encontrada(s). Rodando inferência...\n")

    sem_deteccao = []
    confianca_baixa = []  # possíveis falsos positivos "meio-termo"

    for img_path in imagens:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"⚠️  Não consegui abrir {img_path.name}, pulando.")
            continue

        resultados = modelo.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device="cpu",
            verbose=False,
        )
        resultado = resultados[0]

        num_deteccoes = len(resultado.boxes) if resultado.boxes is not None else 0

        if num_deteccoes == 0:
            print(f"  {img_path.name:35s} -> NENHUMA detecção")
            sem_deteccao.append(img_path.name)
        else:
            nomes_classes = resultado.names
            detalhes = []
            for box in resultado.boxes:
                classe_id = int(box.cls[0])
                confianca = float(box.conf[0])
                nome_classe = nomes_classes.get(classe_id, str(classe_id))
                detalhes.append(f"{nome_classe}({confianca:.2f})")
                if confianca < 0.60:
                    confianca_baixa.append((img_path.name, nome_classe, confianca))
            print(f"  {img_path.name:35s} -> {', '.join(detalhes)}")

        # Salva a imagem com as caixas desenhadas, pra inspeção visual
        frame_anotado = resultado.plot()
        cv2.imwrite(str(pasta_saida / img_path.name), frame_anotado)

    print("\n" + "=" * 60)
    print(f"Resumo: {len(imagens)} imagens testadas")
    print(f"  Sem nenhuma detecção: {len(sem_deteccao)}")
    if sem_deteccao:
        for nome in sem_deteccao:
            print(f"    - {nome}")
    print(f"  Detecções com confiança < 0.60 (revisar de perto): {len(confianca_baixa)}")
    if confianca_baixa:
        for nome, classe, conf in confianca_baixa:
            print(f"    - {nome}: {classe} ({conf:.2f})")
    print(f"\n✅ Imagens anotadas salvas em: {pasta_saida.resolve()}")
    print("Abram essa pasta e confiram visualmente — número sozinho não conta a história toda.")


if __name__ == "__main__":
    main()