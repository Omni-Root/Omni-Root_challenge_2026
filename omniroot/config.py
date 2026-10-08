"""
config.py — Configuração da máquina: o dataclass Config (com os padrões comentados) e a leitura do config.json.

Extraído do main.py sem mudança de comportamento (ver tests/testar_equivalencia.py).
"""

import json
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

@dataclass
class Config:
    # --- Identidade da máquina (deve bater com o numero_serie no Postgres) ---
    maquina_id: str = "RASPI-DEMO-01"
    talhao_id: str = "Talhão Demo"
    clone_id: str = "SP3108"             # Clone genético do eucalipto (ex: SP3108, SP2974, CO41H_TEST)
    idade_talhao_anos: float = 5.3       # Idade do plantio na data da colheita

    # --- Banco local ---
    sqlite_path: str = "./omni_root_local.db"

    # --- Modelo de IA ---
    # Ordem de busca: modelo_path (se existir) > OpenVINO INT8 > OpenVINO
    # FP32 > NCNN > .pt. OpenVINO é o runtime da Intel para CPU (o harvester
    # NÃO tem GPU). Medido neste notebook, mesmo estado térmico: .pt a 1024
    # 2.400 ms; .pt a 640 1.050 ms; OpenVINO FP32 640 ~1.000 ms; OpenVINO
    # INT8 640 ~500 ms. Gere com `python exportar_modelo.py` (uma vez por máquina).
    modelo_path: str = "./models/wood_best_int8_openvino_model"
    modelo_ncnn_path: str = "./models/wood_ncnn_model"   # compatibilidade
    conf_threshold: float = 0.40
    iou_threshold: float = 0.55
    # Limiar de confiança POR CLASSE (sobrepõe conf_threshold para a classe).
    # O modelo foi treinado em madeira serrada: em tora de eucalipto ele
    # confunde casca com "resin" e a borda escura com "Live_Knot" — são as
    # duas classes que mais aparecem "no nada" na demo. Exigir mais delas (e
    # só delas) corta a maior parte dos falsos positivos sem retreinar e sem
    # esconder rachadura/nó morto, que continuam no limiar geral.
    conf_por_classe: dict = None  # ex.: {"resin": 0.80, "Live_Knot": 0.70}
    # A webcam entrega 640x480: inferir a 1024 só amplia pixel, sem informação
    # nova, e custa 3x. O export OpenVINO/NCNN é gerado com tamanho FIXO --
    # tem que ser o mesmo daqui (exportar_modelo.py lê este valor).
    imgsz: int = 640

    # --- Câmera ---
    camera_index: int = 0
    intervalo_captura_seg: float = 2.0  # tempo entre análises

    # --- Região de interesse (ROI), em fração do frame [x, y, w, h] ---
    # A câmera é fixa na garra: a tora SEMPRE aparece na mesma região do
    # frame. Tudo fora dela (fundo, mão de quem segura a peça na demo,
    # mesa, teclado...) é cortado ANTES da inferência — o modelo nem vê.
    # É o filtro mais barato e mais eficaz contra detecção fora da madeira.
    # null/None desliga (usa o frame inteiro). Ex.: [0.2, 0.05, 0.6, 0.9]
    roi: list | None = None

    # --- Regra de negócio (limiar de aprovação) ---
    limiar_aprovacao: float = 0.85       # 85% de confiança mínima
    limiar_quarentena: float = 0.60      # abaixo disso já é reprovado direto

    # --- Filtros de inferência (reduzem falso positivo na demo) ---
    # Ver as funções filtrar_* e FiltroPersistencia.
    filtro_area_minima: float = 0.0008   # fração da área do frame; 0 desliga
    filtro_dentro_contorno: bool = True  # descarta defeito fora da madeira
    filtro_persistencia: int = 2         # análises consecutivas p/ confirmar; 1 desliga

    # --- Distância câmera-tora, FIXA e calibrada (sem sensor) ---
    # A câmera é montada numa posição fixa em relação à tora (no braço do
    # harvester ou na maquete), então a distância não muda entre leituras.
    # CALIBREM ESSE VALOR com a montagem real antes da apresentação: meçam
    # a distância física câmera-tora uma vez, com fita métrica, e coloquem
    # o número aqui. É o único "hardware" que isso substitui — nenhum
    # sensor físico é necessário.
    distancia_camera_tora_cm: float = 45.0

    # --- Conversão pixel -> cm ---
    # Duas formas, em ordem de preferência:
    #
    #   (a) cm_por_px > 0: calibração DIRETA com régua. Com a câmera na
    #       posição final, coloquem uma régua onde a tora fica, meçam
    #       quantos pixels correspondem a 10 cm no frame e dividam:
    #       cm_por_px = 10 / pixels. É o método mais preciso e o mais
    #       fácil de explicar para a banca. Quando definido, ignora (b).
    #
    #   (b) cm_por_px = 0: fórmula de GSD a partir da óptica da câmera
    #       (distância x largura do sensor / foco x largura do frame).
    #       Os valores abaixo são genéricos de webcam — só servem como
    #       ordem de grandeza até calibrar com a régua.
    #
    # A largura do frame em pixels vem do próprio frame capturado
    # (frame.shape[1]); NÃO é o imgsz do modelo — o Ultralytics devolve
    # as caixas já nas coordenadas do frame original.
    cm_por_px: float = 0.0
    distancia_focal_mm: float = 4.0      # foco da lente
    largura_sensor_mm: float = 6.3       # largura física do sensor da câmera

    # --- Escala AUTOMÁTICA por marcador ArUco (ver escala_por_marcador) ---
    # Lado do marcador impresso, em cm. Se o marcador aparecer no frame, a
    # escala vem dele e ignora cm_por_px/GSD. 0 desliga.
    marcador_aruco_cm: float = 5.0

    # --- Comprimento de traçamento do talhão, em cm ---
    # O harvester corta a tora em comprimento FIXO (setpoint da operação:
    # 6 m é o padrão de celulose no Brasil). Quando a câmera vê só a SEÇÃO
    # (rodela), o comprimento não é mensurável na imagem e o volume usa
    # este valor — é o mesmo que o cabeçote usa para traçar.
    comprimento_corte_cm: float = 600.0

    # --- Balanço de branco pela BORDA do frame (ver balancear_branco) ---
    # A luz do ambiente muda a cor de tudo (na sala de vocês a mesa branca
    # saiu azulada). A câmera é fixa e a tora fica no centro, então a borda
    # do frame é fundo: estima-se o "cinza" ali e corrige o frame inteiro.
    # Só afeta a parte clássica (segmentação, portão, casca) — o modelo
    # recebe gray+CLAHE como sempre. Ganho limitado para não desbotar
    # madeira. DESLIGADO por padrão (0): nos testes com a peça segurada na
    # mão, o deslocamento de cor bagunçou a segmentação por saturação, e o
    # portão já tolera luz fria/quente pela regra "pálido e claro = madeira".
    # Ligue (ex.: 0.10 = 10% de cada lado) só se o fundo do palco sair
    # colorido e a segmentação sofrer — e teste antes com --fonte.
    balanco_branco_borda: float = 0.0

    # --- Fundo de referência (ver segmentar_por_fundo) ---
    # A câmera é FIXA na garra: o fundo é sempre o mesmo; a tora é o que
    # muda. Com um frame de referência da garra vazia, a segmentação vira
    # "o que difere do fundo" — independe da cor da madeira, do lençol, da
    # mesa ou da luz. É o método clássico de inspeção com câmera fixa e o
    # único que funciona quando madeira clara e fundo claro têm a mesma cor
    # para a câmera (testado em casa: auto-balanço da webcam deixou o disco
    # da cor do lençol). Captura: tecla 'b' com a garra vazia, ou automática
    # nos primeiros `fundo_auto_seg` segundos (só para câmera; 0 desliga).
    # Sem fundo capturado, cai na segmentação por cor (mesa neutra).
    # Captura automática DESLIGADA por padrão (0): se a peça já estiver na
    # frente da câmera ao iniciar, ela vira "fundo" e nunca mais é detectada
    # -- aconteceu. Tecle 'b' com a mesa vazia; a miniatura no canto da
    # janela mostra o que foi capturado.
    fundo_auto_seg: float = 0.0
    fundo_limiar: float = 22.0           # distância Lab mínima para "difere do fundo"
    fundo_salvar_em: str = "./capturas/fundo_referencia.jpg"  # cópia do fundo capturado (inspeção); "" não salva

    # --- Portão "isso é madeira?" (ver validar_tora) ---
    # A segmentação SEMPRE acha algum blob no centro (é o trabalho dela);
    # quem decide se aquilo é uma tora é este portão. Sem ele, teclado,
    # notebook e mão viravam "Tora #N". Todos os critérios são baratos e
    # explicáveis; qualquer um reprovado = "Sem tora" (nada é gravado).
    tora_exigir_cor: bool = True        # só aceita segmentação pela COR (fallback de brilho não vale)
    tora_solidez_min: float = 0.80      # área / área do casco convexo: tora é convexa; mão/objetos irregulares não
    tora_fracao_pele_max: float = 0.60  # fração de pixels com matiz de pele; acima disso é mão (mão real: 0,7-0,9; madeira rosada chega a 0,4)
    tora_razao_max: float = 6.0         # lado maior / lado menor acima disso = cabo, borda, ruído
    tora_fracao_madeira_min: float = 0.55  # fração de pixels da máscara com cor de madeira (matiz quente + saturação)
    # Madeira PÁLIDA (creme, pouco saturada) só conta como madeira se o Lab b*
    # (OpenCV, 128 = neutro) for pelo menos isto. Era 128 fixo: com a rodela
    # ocupando o quadro, o auto-balanço da webcam puxa o creme para o neutro
    # e o miolo caiu para b* 126-128 (capturas de 30/09) — metade dos pixels
    # falhava por 1-2 unidades e o portão alternava tora/sem tora. Papel
    # branco nas mesmas capturas: b* ~95 (antes 106-112). 122 fica no meio.
    tora_palida_b_min: float = 122.0

    # --- Gravação por EVENTO DE TORA (ver RastreadorTora) ---
    # "evento": uma tora = um registro. O evento abre quando a tora aparece
    # e se mantém por `evento_frames_abrir` análises seguidas, fecha quando
    # some por `evento_frames_fechar` (garra vazia) ou quando o contorno
    # "pula" (outra tora entrou sem gap), e grava UM registro consolidando
    # todos os quadros: mediana dos indicadores, união dos defeitos.
    # O evento é INCREMENTAL: assim que abre (evento_frames_minimo quadros)
    # o registro já é gravado e enviado; enquanto a tora continua na frente
    # da câmera, o MESMO registro é atualizado a cada intervalo_captura_seg
    # com os quadros acumulados (o dashboard vê a tora em ~1 s e vê os
    # números se refinando); tirou a tora, para. Uma tora = um registro,
    # sem esperar a tora sair para aparecer no dashboard.
    # "intervalo": grava um registro NOVO a cada intervalo_captura_seg
    # enquanto houver tora (janelas de 2 s, não toras) — só comparação.
    # Em máquina real o gatilho natural seria o ciclo de corte do cabeçote.
    modo_gravacao: str = "evento"
    evento_frames_abrir: int = 3         # análises seguidas com tora para abrir
    evento_frames_fechar: int = 6        # análises seguidas sem tora para fechar (~0,6 s a 10 fps)
    evento_frames_minimo: int = 4        # evento mais curto que isso é ruído: descarta
    evento_iou_troca: float = 0.25       # IoU do contorno entre quadros abaixo disso = tora diferente
    evento_frames_troca: int = 3         # ...por tantos quadros seguidos
    evento_defeito_min_quadros: int = 2  # defeito precisa aparecer em N quadros do evento

    # --- Câmera ao vivo no dashboard (ver worker_stream) ---
    # A máquina EMPURRA o frame anotado (caixas + HUD) em JPEG para o
    # servidor do dashboard, a poucos fps, enquanto houver rede — mesma
    # direção do sync_daemon.py (em campo a máquina está atrás de 4G/NAT,
    # a central não consegue "puxar" dela). Sem rede, só recua e tenta de
    # novo; a inspeção nunca espera por isso. URL vazia desliga.
    # O token vai em STREAM_TOKEN no .env (é segredo; config.json vai pro Git).
    stream_url: str = ""                 # ex: http://192.168.1.50:3001/api/camera/frame
    stream_fps: float = 4.0              # quadros por segundo enviados (banda ~ 40 KB x fps)
    stream_largura_px: int = 640         # reduz o frame antes de codificar (0 = tamanho original)
    stream_jpeg_qualidade: int = 70      # 1-100

    # --- Posição da máquina (GNSS), ver omniroot/posicao.py ---
    # Cada tora é gravada com a posição do momento em que o evento ABRE (o
    # corte). Sem fonte configurada, sem satélite ou com leitura mais velha
    # que `gnss_validade_s`, a tora é gravada normalmente, só sem posição.
    #   gnss_porta: porta serial NMEA — o GNSS da máquina ou um receptor USB
    #               ("COM5" no Windows). Precisa do pacote pyserial.
    #   gnss_arquivo: trilha NMEA REAL gravada antes, reproduzida em loop
    #               (plano de demonstração; gravada como pos_fonte "gnss_log",
    #               e o dashboard declara isso). Tem prioridade sobre a porta.
    #   gnss_porta = "windows": Localização do Windows — o notebook da maquete
    #               fazendo o papel da máquina (estimada por Wi-Fi, precisão de
    #               dezenas de metros; gravada como "windows_localizacao").
    #   gnss_porta = "auto": procura sozinho a porta serial que está mandando
    #               NMEA (o número do COM muda quando o receptor troca de USB).
    # Ambos vazios desliga. `--gnss COM5`, `--gnss trilha.nmea`, `--gnss auto`
    # ou `--gnss windows` sobrescreve.
    #   gnss_reserva: outra fonte, usada só enquanto a principal não tem
    #               posição (GNSS sem fix, desconectado ou procurando a porta).
    #               Ex.: gnss_porta "auto" + gnss_reserva "windows" = GNSS com
    #               prioridade e a Localização do Windows de reserva. Cada
    #               posição guarda a fonte de onde veio. Vazio = sem reserva.
    gnss_porta: str = ""
    gnss_reserva: str = ""
    gnss_baud: int = 9600                # padrão da maioria dos receptores (alguns usam 4800)
    gnss_arquivo: str = ""
    gnss_validade_s: float = 5.0         # posição mais velha que isso não é "a posição atual"

    # --- Frota: posição em tempo real + trajeto, ver omniroot/telemetria.py ---
    # Com uma fonte de posição ligada (acima), a máquina:
    #   - grava o TRAJETO no SQLite (rastro_local), com ou sem internet — o
    #     sync_daemon.py manda para o Postgres quando há rede;
    #   - envia a posição ATUAL ao dashboard a cada `telemetria_intervalo_s`,
    #     identificada pelo SN (maquina_id), enquanto houver rede.
    # Mesmo token da câmera (STREAM_TOKEN no .env). URL vazia: deduzida da
    # stream_url (mesmo servidor, /api/maquinas/posicao); as duas vazias
    # desligam só o tempo real — o trajeto continua sendo gravado.
    telemetria_url: str = ""             # ex: http://192.168.1.50:3001/api/maquinas/posicao
    telemetria_intervalo_s: float = 2.0  # posição ao vivo: um POST a cada N s
    rastro_intervalo_s: float = 5.0      # trajeto: no máximo um ponto a cada N s...
    rastro_distancia_min_m: float = 20.0  # ...e só se andou pelo menos isso (ou a precisão informada, se maior):
                                          # GNSS parado em ambiente fechado "passeia" 10-12 m — isso não é trajeto
    rastro_parado_s: float = 60.0        # parada: ainda assim um ponto a cada N s ("estive aqui")

    # --- Pouca luz / operação noturna, ver omniroot/luz.py ---
    # Mede brilho e ruído de cada quadro; em luz BAIXA ou CRÍTICA empilha
    # quadros da tora parada e aplica ganho que preserva a cor. Nunca recusa
    # por luz: em luz crítica mede e marca a tora como de baixa confiança
    # ("_luz_critica" no método). Nada disso retreina o modelo. Os limiares
    # são ponto de partida: o HUD mostra brilho/ruído medidos — calibrem
    # apagando a luz da bancada.
    luz_realce: bool = True
    luz_limiar_boa: float = 90.0            # brilho (percentil 95, 0-255) mínimo de luz "boa"
    luz_limiar_critica: float = 20.0        # abaixo disso: luz crítica (sintético: o realce ainda acha a tora em ~15)
    luz_ruido_baixa: float = 6.0            # ruído acima disso conta como luz baixa
    luz_ruido_critico: float = 16.0         # ruído acima disso: luz crítica
    luz_ganho_max: float = 4.0
    luz_empilhar_max: int = 8               # quadros da tora parada somados (ruído cai ~raiz de N)


def carregar_configuracao_json(caminho_json: str = "./config.json") -> Config:
    """Carrega as configurações operacionais da máquina a partir de um arquivo JSON estático."""
    cfg = Config()
    p = Path(caminho_json)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                dados = json.load(f)
                if isinstance(dados, dict):
                    for k, v in dados.items():
                        if hasattr(cfg, k):
                            setattr(cfg, k, v)
                        elif k == "largura_imagem_px":
                            print(
                                "ℹ️  'largura_imagem_px' não é mais usado: a largura vem do "
                                "próprio frame da câmera. Pode remover do config.json."
                            )
        except Exception as e:
            print(f"⚠️ Erro ao carregar {caminho_json}: {e}")

    # ROI precisa ser [x, y, w, h] em fração do frame, tudo em (0, 1].
    if cfg.roi is not None:
        ok = (
            isinstance(cfg.roi, (list, tuple)) and len(cfg.roi) == 4
            and all(isinstance(v, (int, float)) for v in cfg.roi)
            and 0.0 <= cfg.roi[0] < 1.0 and 0.0 <= cfg.roi[1] < 1.0
            and 0.0 < cfg.roi[2] <= 1.0 - cfg.roi[0]
            and 0.0 < cfg.roi[3] <= 1.0 - cfg.roi[1]
        )
        if not ok:
            print(f"⚠️ 'roi' inválida ({cfg.roi}); esperado [x, y, w, h] em fração do frame. Ignorando.")
            cfg.roi = None
    return cfg
