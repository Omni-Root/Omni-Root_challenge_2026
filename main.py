"""
main.py — Script principal do Raspberry Pi (Tarefa 2)
Projeto: Qualidade da Madeira — Challenge FIAP x John Deere/Suzano

O que este script faz, em ordem:
  1. Captura um frame da câmera acoplada à máquina
  2. Roda o modelo YOLO (NCNN) para detectar defeitos (praga, apodrecimento)
  3. Lê os sensores conectados diretamente ao Raspberry (ultrassônico + força)
  4. Calcula os 4 indicadores de qualidade (densidade, altura, tortuosidade,
     apodrecimento_pragas)
  5. Classifica a tora (aprovado / quarentena / reprovado) usando o limiar
     de confiança de 85%
  6. Grava tudo no SQLite local (schema_sqlite.sql), pronto para o
     sync.go sincronizar com o PostgreSQL quando houver rede

Modo de uso:
  Produção no Raspberry:
    python3 main.py

  Teste no seu PC, sem hardware (webcam comum + sensores simulados):
    python3 main.py --simulado

Dependências (requirements.txt):
  ultralytics
  opencv-python-headless
  RPi.GPIO      (só no Raspberry — em --simulado não é necessário)
  spidev        (só no Raspberry — comunicação com o ADC MCP3008)
"""

import argparse
import hashlib
import json
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

@dataclass
class Config:
    # --- Identidade da máquina (deve bater com o numero_serie no Postgres) ---
    maquina_id: str = "RASPI-DEMO-01"
    talhao_id: str = "Talhão Demo"

    # --- Banco local ---
    sqlite_path: str = "./stanford_local.db"

    # --- Modelo de IA ---
    modelo_ncnn_path: str = "./models/wood_ncnn_model"
    conf_threshold: float = 0.40
    iou_threshold: float = 0.55
    imgsz: int = 640

    # --- Câmera ---
    camera_index: int = 0
    intervalo_captura_seg: float = 2.0  # tempo entre análises

    # --- Regra de negócio (limiar de aprovação) ---
    limiar_aprovacao: float = 0.85       # 85% de confiança mínima
    limiar_quarentena: float = 0.60      # abaixo disso já é reprovado direto

    # --- Sensor ultrassônico (HC-SR04) — pinos GPIO (BCM) ---
    pino_trigger: int = 23
    pino_echo: int = 24

    # --- Sensor de força (via ADC MCP3008, canal 0) ---
    adc_canal_forca: int = 0

    # --- Câmera: parâmetros para cálculo de dimensão real (GSD) ---
    # Precisam ser calibrados com a câmera real usada na apresentação!
    distancia_focal_mm: float = 4.0      # foco da lente
    largura_sensor_mm: float = 6.3       # largura física do sensor da câmera
    largura_imagem_px: int = 640         # resolução usada na inferência


CONFIG = Config()


# ============================================================
# CAMADA DE SENSORES
# ============================================================
# Duas implementações: real (GPIO/SPI no Raspberry) e simulada
# (para testar a lógica no notebook, sem hardware).
# ============================================================

class SensorReal:
    """Lê os sensores conectados diretamente ao Raspberry Pi."""

    def __init__(self, cfg: Config):
        import RPi.GPIO as GPIO
        import spidev

        self.GPIO = GPIO
        self.cfg = cfg

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(cfg.pino_trigger, GPIO.OUT)
        GPIO.setup(cfg.pino_echo, GPIO.IN)
        GPIO.output(cfg.pino_trigger, False)

        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 1350000

    def ler_distancia_cm(self) -> float:
        """Mede a distância câmera-tora com o sensor ultrassônico HC-SR04."""
        GPIO = self.GPIO
        cfg = self.cfg

        GPIO.output(cfg.pino_trigger, True)
        time.sleep(0.00001)
        GPIO.output(cfg.pino_trigger, False)

        timeout = time.time() + 0.04  # 40ms de timeout de segurança
        inicio = time.time()
        while GPIO.input(cfg.pino_echo) == 0 and time.time() < timeout:
            inicio = time.time()

        fim = time.time()
        while GPIO.input(cfg.pino_echo) == 1 and time.time() < timeout:
            fim = time.time()

        duracao = fim - inicio
        distancia_cm = (duracao * 34300) / 2  # velocidade do som / 2 (ida e volta)
        return round(distancia_cm, 2)

    def ler_forca_bruta(self) -> int:
        """Lê o valor bruto do ADC (0-1023) referente ao sensor de força."""
        canal = self.cfg.adc_canal_forca
        resposta = self.spi.xfer2([1, (8 + canal) << 4, 0])
        valor = ((resposta[1] & 3) << 8) + resposta[2]
        return valor


class SensorSimulado:
    """Gera leituras plausíveis para testar a lógica sem hardware."""

    def ler_distancia_cm(self) -> float:
        return round(random.uniform(30.0, 80.0), 2)

    def ler_forca_bruta(self) -> int:
        return random.randint(200, 900)


# ============================================================
# CÁLCULO DOS INDICADORES DE QUALIDADE
# ============================================================

def calcular_dimensao_real_cm(tamanho_px: float, distancia_cm: float, cfg: Config) -> float:
    """
    Converte um tamanho em pixels para centímetros reais, usando a
    trigonometria de GSD (Ground Sample Distance) — a mesma lógica
    usada em fotografia aérea, adaptada para distância câmera-tora.

    GSD (cm/pixel) = (distância_cm * largura_sensor_mm) / (foco_mm * largura_imagem_px)
    """
    gsd_cm_por_px = (distancia_cm * cfg.largura_sensor_mm) / (
        cfg.distancia_focal_mm * cfg.largura_imagem_px
    )
    return round(tamanho_px * gsd_cm_por_px, 2)


def calcular_densidade_estimada(valor_forca_bruto: int) -> float:
    """
    Converte a leitura bruta do sensor de força (0-1023) em uma
    estimativa de densidade (kg/m3).

    ATENÇÃO: a fórmula abaixo é uma aproximação linear para fins de
    demonstração. Para uso real, calibrar com amostras de densidade
    conhecida (ex: usando um densímetro de referência) e ajustar o
    slope/intercept da regressão.
    """
    densidade_min, densidade_max = 350.0, 750.0  # faixa típica de eucalipto, kg/m3
    proporcao = valor_forca_bruto / 1023.0
    densidade = densidade_min + proporcao * (densidade_max - densidade_min)
    return round(densidade, 1)


def extrair_contorno_tora(frame: np.ndarray) -> np.ndarray | None:
    """
    Segmenta a região da tora no frame usando técnicas de visão computacional
    clássica (Otsu thresholding + operações morfológicas).
    Retorna o contorno principal da tora ou None.
    """
    if frame is None:
        return None

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contornos, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            return None

        area_minima = (frame.shape[0] * frame.shape[1]) * 0.05
        validos = [c for c in contornos if cv2.contourArea(c) >= area_minima]
        return max(validos, key=cv2.contourArea) if validos else max(contornos, key=cv2.contourArea)
    except Exception:
        return None


def calcular_tortuosidade(mascara_ou_contorno) -> float:
    """
    Calcula um índice de tortuosidade (0 = perfeitamente reta,
    valores maiores = mais tortuosa) a partir do contorno da tora.

    Método: ajusta uma reta ao eixo do contorno e mede o desvio
    máximo perpendicular a essa reta, normalizado pelo comprimento.
    """
    if mascara_ou_contorno is None or len(mascara_ou_contorno) < 5:
        return 0.0

    try:
        pontos = mascara_ou_contorno.reshape(-1, 2).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pontos, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        direcao = np.array([vx, vy])
        origem = np.array([x0, y0])

        desvios = []
        for p in pontos:
            vetor = p - origem
            proj = np.dot(vetor, direcao) * direcao
            perpendicular = vetor - proj
            desvios.append(np.linalg.norm(perpendicular))

        extensao = pontos.max(axis=0) - pontos.min(axis=0)
        comprimento = max(float(np.linalg.norm(extensao)), 1.0)
        indice = (max(desvios) / comprimento) * 100.0
        return round(float(min(indice, 100.0)), 2)
    except Exception:
        return 0.0


def extrair_defeitos_yolo(resultado, cfg: Config, frame_shape: tuple | None = None) -> list:
    """Converte as detecções do YOLO em uma lista de defeitos estruturados com área relativa."""
    defeitos = []
    nomes_classes = resultado.names

    img_h, img_w = frame_shape[:2] if frame_shape is not None else (640, 640)
    area_total_img = float(img_h * img_w)

    for box in resultado.boxes:
        classe_id = int(box.cls[0])
        tipo_defeito = nomes_classes.get(classe_id, "wood_defect")
        confianca = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        largura = max(0.0, x2 - x1)
        altura = max(0.0, y2 - y1)
        area_box = largura * altura
        area_relativa = round(area_box / area_total_img, 4)

        defeitos.append({
            "tipo_defeito": tipo_defeito,
            "pos_x": round(x1, 2),
            "pos_y": round(y1, 2),
            "largura": round(largura, 2),
            "altura": round(altura, 2),
            "area_relativa": area_relativa,
            "confianca": round(confianca, 4),
        })

    return defeitos


# ============================================================
# REGRA DE NEGÓCIO — CLASSIFICAÇÃO FINAL DA TORA
# ============================================================

def classificar_qualidade(confianca_saude: float, defeitos: list, cfg: Config) -> str:
    """
    Aplica a regra de negócio florestal:
      - Sem defeito significativo + saúde alta (>= 85%) -> aprovado
      - Defeitos moderados ou incerteza (60% a 85%)      -> quarentena (revisão manual)
      - Defeito grave (área ocupada > 5% ou podridão)    -> reprovado
    """
    # 1. Podridão grave em modelo multiclasse
    tem_podridao = any(
        d["tipo_defeito"] in ("apodrecimento", "praga", "rot") and d["confianca"] >= 0.70
        for d in defeitos
    )
    if tem_podridao:
        return "reprovado"

    # 2. Defeito extenso em modelo de classe única (wood_defect)
    tem_defeito_extenso = any(
        d.get("area_relativa", 0.0) >= 0.05 and d["confianca"] >= 0.65
        for d in defeitos
    )
    if tem_defeito_extenso:
        return "reprovado"

    # 3. Tora saudável
    if not defeitos and confianca_saude >= cfg.limiar_aprovacao:
        return "aprovado"

    # 4. Quarentena
    if confianca_saude >= cfg.limiar_quarentena or len(defeitos) <= 2:
        return "quarentena"

    return "reprovado"



def gerar_hash_sha256(dados: dict) -> str:
    """Gera um hash de integridade do registro (rastreabilidade)."""
    dados_serializados = json.dumps(dados, sort_keys=True, default=str)
    return hashlib.sha256(dados_serializados.encode("utf-8")).hexdigest()


# ============================================================
# CAMADA DE BANCO DE DADOS (SQLite local)
# ============================================================

def conectar_banco(cfg: Config) -> sqlite3.Connection:
    conexao = sqlite3.connect(cfg.sqlite_path)
    conexao.execute("PRAGMA foreign_keys = ON")

    # Verifica se a tabela toras_local já existe, caso contrário carrega o schema
    tabelas = conexao.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='toras_local'").fetchall()
    if not tabelas:
        schema_path = Path(cfg.sqlite_path).parent / "Banco de dados" / "schema_sqlite.sql"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                conexao.executescript(f.read())
    return conexao


def salvar_inspecao(
    conexao: sqlite3.Connection,
    cfg: Config,
    indicadores: dict,
    defeitos: list,
    status_classificacao: str,
    confianca_media: float,
) -> str:
    """Grava a tora + indicadores + defeitos nas 3 tabelas locais."""
    cursor = conexao.cursor()

    uuid_local = str(uuid.uuid4())
    log_id = f"LOG-{datetime.now():%Y%m%d%H%M%S}-{uuid_local[:4]}"
    data_inspecao = datetime.now().isoformat(timespec="seconds")

    hash_dados = gerar_hash_sha256({
        "uuid_local": uuid_local,
        "log_id": log_id,
        "indicadores": indicadores,
        "defeitos": defeitos,
        "status": status_classificacao,
    })

    # --- 1. Insere a tora ---
    cursor.execute(
        """
        INSERT INTO toras_local
            (uuid_local, maquina_id, talhao_id, log_id, data_inspecao,
             confianca_ia, status_classificacao, hash_sha256, sync_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (uuid_local, cfg.maquina_id, cfg.talhao_id, log_id, data_inspecao,
         confianca_media, status_classificacao, hash_dados),
    )
    tora_id_local = cursor.lastrowid

    # --- 2. Insere os indicadores de qualidade ---
    for tipo, dados_indicador in indicadores.items():
        cursor.execute(
            """
            INSERT INTO indicadores_qualidade_local
                (tora_id, tipo_indicador, valor, unidade, metodo_medicao, sync_status)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (tora_id_local, tipo, dados_indicador["valor"],
             dados_indicador["unidade"], dados_indicador["metodo"]),
        )

    # --- 3. Insere os defeitos detectados ---
    for defeito in defeitos:
        cursor.execute(
            """
            INSERT INTO defeitos_detectados_local
                (tora_id, tipo_defeito, pos_x, pos_y, largura, altura, confianca, sync_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (tora_id_local, defeito["tipo_defeito"], defeito["pos_x"], defeito["pos_y"],
             defeito["largura"], defeito["altura"], defeito["confianca"]),
        )

    conexao.commit()
    return uuid_local


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="IA de Qualidade da Madeira — Raspberry Pi")
    parser.add_argument(
        "--simulado", action="store_true",
        help="Roda com webcam comum + sensores simulados (teste sem hardware)"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Abre janela gráfica exibindo as bounding boxes da câmera ao vivo"
    )
    args = parser.parse_args()

    cfg = CONFIG
    sensores = SensorSimulado() if args.simulado else SensorReal(cfg)

    print("📦 Carregando modelo IA...")
    caminho_modelo = None
    for p in [Path(cfg.modelo_ncnn_path), Path("./models/wood_best.pt"), Path("./models/wood_ncnn_model"), Path("./yolov8n.pt")]:
        if p.exists():
            caminho_modelo = str(p)
            break
    if caminho_modelo is None:
        caminho_modelo = "yolov8n.pt"

    modelo = YOLO(caminho_modelo, task="detect")
    print(f"✅ Modelo carregado com sucesso ({caminho_modelo})!")

    conexao = conectar_banco(cfg)

    print(f"📷 Abrindo câmera (índice {cfg.camera_index})...")
    captura = cv2.VideoCapture(cfg.camera_index)
    if not captura.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera.")

    print("🚀 Iniciando loop de análise. Ctrl+C para parar.\n")

    try:
        while True:
            sucesso, frame = captura.read()
            if not sucesso:
                print("⚠️  Falha ao capturar frame. Tentando de novo...")
                time.sleep(1)
                continue

            # --- 1. Roda o YOLO no frame atual ---
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

            # Confiança de saúde da tora: 1.0 se limpa, ou (1.0 - conf_max_defeito)
            if defeitos:
                maior_conf_defeito = max(d["confianca"] for d in defeitos)
                confianca_saude = round(max(0.0, 1.0 - (maior_conf_defeito * 0.5)), 4)
            else:
                confianca_saude = 1.0

            # --- 2. Lê os sensores ---
            distancia_cm = sensores.ler_distancia_cm()
            forca_bruta = sensores.ler_forca_bruta()

            # --- 3. Extrai contorno da tora e calcula indicadores ---
            contorno_tora = None
            if resultado.masks is not None and len(resultado.masks.xy) > 0:
                contorno_tora = max(resultado.masks.xy, key=len)
            else:
                contorno_tora = extrair_contorno_tora(frame)

            # Altura/Comprimento da tora (pixels -> cm via GSD):
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

            # --- 4. Classifica e salva ---
            status = classificar_qualidade(confianca_saude, defeitos, cfg)
            uuid_gerado = salvar_inspecao(conexao, cfg, indicadores, defeitos, status, confianca_saude)

            emoji_status = {"aprovado": "✅", "quarentena": "⚠️", "reprovado": "❌"}[status]
            print(
                f"{emoji_status} [{uuid_gerado[:8]}] status={status} "
                f"saúde={confianca_saude:.2%} altura={altura_cm}cm "
                f"densidade={densidade}kg/m3 tortuosidade={tortuosidade} "
                f"defeitos={len(defeitos)}"
            )

            # Exibe janela visual em tempo real se solicitado
            if args.simulado or args.gui:
                frame_visual = resultado.plot()
                cor_hud = {"aprovado": (0, 255, 0), "quarentena": (0, 255, 255), "reprovado": (0, 0, 255)}[status]
                cv2.putText(frame_visual, f"Status: {status.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor_hud, 2)
                cv2.putText(frame_visual, f"Saude: {confianca_saude:.1%}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_visual, f"Altura: {altura_cm:.1f}cm | Densidade: {densidade:.0f}kg/m3", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Omni-Root | John Deere Wood Inspection", frame_visual)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(cfg.intervalo_captura_seg)

    except KeyboardInterrupt:
        print("\n🛑 Encerrando por solicitação do usuário...")
    finally:
        captura.release()
        cv2.destroyAllWindows()
        conexao.close()
        print("✅ Recursos liberados. Até a próxima inspeção!")


if __name__ == "__main__":
    main()
