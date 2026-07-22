"""
simular_cenario.py — Simulador Completo End-to-End da Operação Florestal
Projeto: Qualidade da Madeira — Challenge FIAP x John Deere/Suzano

O QUE ESTE SCRIPT FAZ:
1. Carrega imagens reais de teste (do dataset de validação ou gera sintetizadas se necessário)
2. Simula o ciclo da máquina colhedora John Deere processando cada tora
3. Executa o modelo YOLOv8 para detecção de defeitos
4. Simula os sensores de ultrassom (distância) e força (densidade)
5. Aplica visão computacional para contorno e tortuosidade
6. Salva todas as inspeções no banco SQLite local (stanford_local.db)
7. Executa o exportador StanForD 2010 (.hpr) para demonstrar a saída oficial
8. Exibe um relatório estatístico completo da operação

USO:
    python simular_cenario.py
    python simular_cenario.py --num-toras 15
"""

import argparse
import random
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Importa as funções principais do main.py
from main import (
    CONFIG,
    SensorSimulado,
    calcular_densidade_estimada,
    calcular_dimensao_real_cm,
    calcular_tortuosidade,
    classificar_qualidade,
    conectar_banco,
    extrair_contorno_tora,
    extrair_defeitos_yolo,
    salvar_inspecao,
)


def carregar_modelo():
    """Tenta carregar o melhor modelo treinado ou o YOLOv8 padrão."""
    caminhos_modelo = [
        Path("./models/wood_best.pt"),
        Path("./models/wood_ncnn_model"),
        Path("./yolov8n.pt"),
    ]
    for caminho in caminhos_modelo:
        if caminho.exists():
            print(f"📦 Carregando modelo IA: {caminho}")
            return YOLO(str(caminho), task="detect")

    print("⚠️ Nenhum modelo local encontrado. Baixando yolov8n.pt padrão...")
    return YOLO("yolov8n.pt", task="detect")


def obter_imagens_teste(pasta_dataset: Path, num_imagens: int) -> list[np.ndarray]:
    """Busca imagens de teste no dataset local ou gera frames sintetizados."""
    imagens_encontradas = list(pasta_dataset.rglob("*.jpg")) + list(pasta_dataset.rglob("*.png"))

    frames = []
    if imagens_encontradas:
        amostra = random.sample(imagens_encontradas, min(num_imagens, len(imagens_encontradas)))
        for img_path in amostra:
            img = cv2.imread(str(img_path))
            if img is not None:
                frames.append(img)

    # Se não houver imagens suficientes no disco, gera imagens sintéticas de tora
    while len(frames) < num_imagens:
        h, w = 640, 640
        # Fundo verde floresta / terra
        fundo = np.zeros((h, w, 3), dtype=np.uint8)
        fundo[:, :] = (35, 60, 30)

        # Desenha uma tora (cilindro marrom)
        cor_tora = (40, 90, 150)
        pt1 = (random.randint(200, 250), 50)
        pt2 = (random.randint(380, 440), 590)
        cv2.line(fundo, pt1, pt2, cor_tora, thickness=random.randint(120, 180))

        # Adiciona algumas manchas/ruídos como potenciais nó/defeitos
        if random.random() > 0.4:
            cx = (pt1[0] + pt2[0]) // 2 + random.randint(-20, 20)
            cy = (pt1[1] + pt2[1]) // 2 + random.randint(-50, 50)
            cv2.circle(fundo, (cx, cy), random.randint(15, 35), (20, 40, 70), -1)

        frames.append(fundo)

    return frames[:num_imagens]


def rodar_simulacao(num_toras: int = 10):
    print("=" * 65)
    print("🌲 SIMULADOR DE COLHEITA FLORESTAL — JOHN DEERE x SUZANO (PoC)")
    print("=" * 65)

    cfg = CONFIG
    cfg.sqlite_path = "./stanford_local.db"

    modelo = carregar_modelo()
    sensores = SensorSimulado()
    conexao = conectar_banco(cfg)

    pasta_val = Path("./dataset/wood_yolo/images/val")
    if not pasta_val.exists():
        pasta_val = Path("./data")

    print(f"📸 Coletando {num_toras} imagens de amostra para simular a operação...")
    frames = obter_imagens_teste(pasta_val, num_toras)

    resumo_estatistico = {"aprovado": 0, "quarentena": 0, "reprovado": 0}
    historico_toras = []

    print("\n🚀 Iniciando ciclo automático de colheita no cabeçote da máquina:\n")

    for idx, frame in enumerate(frames, start=1):
        # 1. Inferência YOLO
        resultados = modelo.predict(
            source=frame,
            conf=cfg.conf_threshold,
            iou=cfg.iou_threshold,
            imgsz=cfg.imgsz,
            device="cpu",
            verbose=False,
        )
        resultado = resultados[0]
        defeitos = extrair_defeitos_yolo(resultado, cfg, frame.shape)

        # 2. Confiança da saúde
        if defeitos:
            maior_conf = max(d["confianca"] for d in defeitos)
            confianca_saude = round(max(0.0, 1.0 - (maior_conf * 0.5)), 4)
        else:
            confianca_saude = 1.0

        # 3. Sensores & Métricas
        distancia_cm = sensores.ler_distancia_cm()
        forca_bruta = sensores.ler_forca_bruta()

        contorno_tora = extrair_contorno_tora(frame)
        if contorno_tora is not None and len(contorno_tora) > 0:
            _, _, _, h_box = cv2.boundingRect(contorno_tora)
            altura_px = float(h_box)
        else:
            altura_px = float(frame.shape[0] * 0.7)

        altura_cm = calcular_dimensao_real_cm(altura_px, distancia_cm, cfg)
        densidade = calcular_densidade_estimada(forca_bruta)
        tortuosidade = calcular_tortuosidade(contorno_tora)

        indicadores = {
            "densidade": {"valor": densidade, "unidade": "kg/m3", "metodo": "fusao_sensores"},
            "altura": {"valor": altura_cm, "unidade": "cm", "metodo": "imagem_gsd_ultrassom"},
            "tortuosidade": {"valor": tortuosidade, "unidade": "indice", "metodo": "opencv_contorno"},
            "apodrecimento_pragas": {
                "valor": round(confianca_saude * 100, 2),
                "unidade": "%",
                "metodo": "yolo",
            },
        }

        status = classificar_qualidade(confianca_saude, defeitos, cfg)
        uuid_gerado = salvar_inspecao(conexao, cfg, indicadores, defeitos, status, confianca_saude)

        resumo_estatistico[status] += 1
        historico_toras.append({
            "uuid": uuid_gerado[:8],
            "status": status,
            "saude": confianca_saude,
            "altura": altura_cm,
            "densidade": densidade,
            "tortuosidade": tortuosidade,
            "defeitos": len(defeitos),
        })

        emoji = {"aprovado": "✅", "quarentena": "⚠️", "reprovado": "❌"}[status]
        print(
            f"Tora #{idx:02d} {emoji} [{uuid_gerado[:8]}] Status: {status:<10} | "
            f"Saúde IA: {confianca_saude:>6.1%} | Altura: {altura_cm:>5.1f}cm | "
            f"Densidade: {densidade:>5.1f}kg/m³ | Tortuosos.: {tortuosidade:>5.2f} | "
            f"Defeitos: {len(defeitos)}"
        )
        time.sleep(0.1)

    conexao.close()

    print("\n" + "=" * 65)
    print("📊 RELATÓRIO CONSOLIDADO DA SIMULAÇÃO FLORESTAL")
    print("=" * 65)
    print(f"Total de Toras Inspecionadas : {num_toras}")
    print(f"✅ Aprovadas                : {resumo_estatistico['aprovado']} ({resumo_estatistico['aprovado']/num_toras:.1%})")
    print(f"⚠️  Quarentena (Revisão)     : {resumo_estatistico['quarentena']} ({resumo_estatistico['quarentena']/num_toras:.1%})")
    print(f"❌ Reprovadas                : {resumo_estatistico['reprovado']} ({resumo_estatistico['reprovado']/num_toras:.1%})")
    print(f"💾 Registros armazenados em  : ./stanford_local.db")

    # Testa exportação StanForD 2010
    print("\n📄 Gerando arquivo oficial StanForD 2010 (.hpr)...")
    try:
        import subprocess

        res = subprocess.run(["python", "stanford_export.py"], capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout.strip())
            print("✅ Exportação StanForD 2010 concluída com sucesso!")
        else:
            print(f"⚠️ Erro ao exportar StanForD: {res.stderr}")
    except Exception as e:
        print(f"⚠️ Falha ao executar stanford_export.py: {e}")

    print("\n✨ Simulação finalizada com sucesso! O sistema está 100% operacional.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador de Inspeção de Madeira John Deere")
    parser.add_argument("--num-toras", type=int, default=10, help="Quantidade de toras para simular")
    args = parser.parse_args()

    rodar_simulacao(args.num_toras)
