"""
luz.py — pouca luz / operação noturna, só no software (sem retreinar nada).

A colheita pode ser noturna. O modelo YOLO já recebe cinza + CLAHE, mas
isso só redistribui o contraste que existe; e os indicadores do desafio
(casca, tortuosidade, o portão "é madeira?") são visão clássica sobre COR
e BRILHO — justamente o que a pouca luz destrói. Aqui, antes de qualquer
análise, cada quadro passa por:

  1. MEDIDOR DE LUZ — dois números por quadro:
       brilho: percentil 95 do cinza (a parte mais clara da cena: sobre a
               mesa preta/garra, é a madeira iluminada, não o fundo);
       ruido:  desvio do ruído do sensor (método de Immerkær, 1996) — câmera
               no escuro sobe o ganho sozinha e a imagem "clara" vem granulada.
     -> nível "boa" / "baixa" / "critica" (com histerese, para não piscar).

  2. REALCE, em luz baixa ou crítica:
       - empilhamento: média dos últimos quadros da tora PARADA (até N). O
         ruído cai ~raiz(N). Se a cena mexe (tora entrando/girando), zera e
         recomeça — nunca borra movimento;
       - ganho multiplicando B, G e R igualmente: clareia SEM mudar matiz nem
         saturação (o portão e a casca dependem de cor). Limitado.

  3. NUNCA RECUSA POR LUZ — a tora é colhida de qualquer jeito; se não for
     registrada, o inventário deixa de ser "100% das toras". Quem decide se
     há tora é o portão "é madeira?" (no escuro total ele mesmo não acha
     contorno). Em luz crítica a tora é MEDIDA e marcada como de baixa
     confiança — o dashboard mostra e não dispara alerta só com ela.

  4. PROVENIÊNCIA — o sufixo "_luz_baixa" ou "_luz_critica" vai no método
     de medição dos indicadores de imagem (como a densidade diz de onde veio
     e a posição diz a fonte).

Os limiares são pontos de partida, não verdade: o HUD mostra brilho e ruído
medidos para calibrar na bancada (config.json: luz_*).
"""

import math

import cv2
import numpy as np

NIVEIS = ("boa", "baixa", "critica")

# Kernel do estimador de ruído de Immerkær (anula imagem suave, sobra o ruído).
_KERNEL_RUIDO = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def medir_luz(frame: np.ndarray) -> tuple[float, float]:
    """(brilho = percentil 95 do cinza, ruído = sigma estimado), no quadro reduzido à metade."""
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    meio = cinza[::2, ::2].astype(np.float32)  # metade da resolução: rápido, e o ruído continua visível
    brilho = float(np.percentile(meio, 95))
    h, w = meio.shape
    if h < 3 or w < 3:
        return brilho, 0.0
    resposta = cv2.filter2D(meio, -1, _KERNEL_RUIDO, borderType=cv2.BORDER_REFLECT)
    ruido = math.sqrt(math.pi / 2.0) * float(np.abs(resposta[1:-1, 1:-1]).sum()) / (6.0 * (w - 2) * (h - 2))
    return brilho, ruido


def nivel_do_evento(niveis: list[str]) -> str:
    """
    Nível de luz de uma tora a partir dos níveis dos seus quadros: "critica"
    se a maioria foi crítica; "baixa" se a maioria foi baixa OU crítica;
    senão "boa". (Uma tora com metade dos quadros em luz ruim não é de luz boa.)
    """
    if not niveis:
        return "boa"
    if niveis.count("critica") * 2 > len(niveis):
        return "critica"
    if (niveis.count("baixa") + niveis.count("critica")) * 2 > len(niveis):
        return "baixa"
    return "boa"


class RealceLuz:
    """Mede a luz, realça quando ela está ruim e informa o nível. Um por câmera, na thread de captura."""

    def __init__(
        self,
        ligado: bool = True,
        limiar_boa: float = 90.0,      # brilho (p95, 0-255) a partir do qual a luz é "boa"
        limiar_critica: float = 20.0,  # abaixo disso: luz crítica (mede, com baixa confiança)
        ruido_baixa: float = 6.0,      # ruído acima disso já conta como luz baixa (câmera no ganho alto)
        ruido_critico: float = 16.0,   # ruído acima disso: luz crítica
        ganho_max: float = 4.0,
        alvo_brilho: float = 170.0,
        empilhar_max: int = 8,
    ):
        self.ligado = ligado
        self.limiar_boa = limiar_boa
        self.limiar_critica = limiar_critica
        self.ruido_baixa = ruido_baixa
        self.ruido_critico = ruido_critico
        self.ganho_max = ganho_max
        self.alvo_brilho = alvo_brilho
        self.empilhar_max = max(1, int(empilhar_max))
        self._brilho: float | None = None  # médias móveis (suavizam o medidor)
        self._ruido: float | None = None
        self.nivel = "boa"
        self.ganho = 1.0
        self._acum: np.ndarray | None = None
        self._acum_peq: np.ndarray | None = None
        self._n = 0

    # ---------- classificação ----------

    def _classificar(self, brilho: float, ruido: float) -> str:
        """Nível com histerese de 10%: perto do limiar ele não fica trocando a cada quadro."""
        folga_critica = 1.10 if self.nivel == "critica" else 1.0  # para SAIR de crítica precisa passar 10% acima
        folga_boa = 1.10 if self.nivel != "boa" else 1.0          # para VOLTAR a boa, idem
        if brilho < self.limiar_critica * folga_critica or ruido > self.ruido_critico:
            return "critica"
        if brilho < self.limiar_boa * folga_boa or ruido > self.ruido_baixa:
            return "baixa"
        return "boa"

    # ---------- empilhamento ----------

    def _empilhar(self, frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32)
        peq = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._acum is None or self._acum.shape != f.shape:
            self._acum, self._acum_peq, self._n = f, peq, 1
            return frame
        # Mexeu? Conta os pixels (na miniatura, onde o ruído já caiu) que
        # mudaram MAIS do que o ruído explica. Média do quadro inteiro não
        # serve: numa cena escura a tora mexendo se dilui no fundo preto.
        limiar_px = 8.0 + (self._ruido or 0.0)
        mudou = float(np.mean(np.abs(peq - self._acum_peq) > limiar_px))
        if mudou > 0.015:  # mais de 1,5% da imagem mudou => cena nova
            self._acum, self._acum_peq, self._n = f, peq, 1
            return frame
        self._n = min(self._n + 1, self.empilhar_max)
        a = 1.0 / self._n
        self._acum += a * (f - self._acum)
        self._acum_peq += a * (peq - self._acum_peq)
        return np.clip(self._acum, 0, 255).astype(np.uint8)

    def _zerar_pilha(self) -> None:
        self._acum = None
        self._acum_peq = None
        self._n = 0

    # ---------- interface ----------

    def processar(self, frame: np.ndarray) -> np.ndarray:
        """Mede e, se a luz estiver ruim, devolve o quadro realçado (senão, o próprio)."""
        if frame is None or not self.ligado:
            return frame
        brilho, ruido = medir_luz(frame)
        self._brilho = brilho if self._brilho is None else 0.8 * self._brilho + 0.2 * brilho
        self._ruido = ruido if self._ruido is None else 0.8 * self._ruido + 0.2 * ruido
        self.nivel = self._classificar(self._brilho, self._ruido)

        if self.nivel == "boa":
            self._zerar_pilha()
            self.ganho = 1.0
            return frame

        saida = self._empilhar(frame)
        # Ganho igual nos três canais: clareia sem mudar a cor (matiz e
        # saturação são razões entre canais — o portão e a casca continuam
        # vendo "madeira"). Limitado: ganho alto demais só amplia ruído.
        self.ganho = float(min(self.ganho_max, max(1.0, self.alvo_brilho / max(self._brilho, 1.0))))
        if self.ganho > 1.01:
            saida = cv2.convertScaleAbs(saida, alpha=self.ganho, beta=0)
        return saida

    @property
    def estado(self) -> dict:
        return {
            "nivel": self.nivel if self.ligado else "boa",
            "brilho": round(self._brilho or 0.0, 1),
            "ruido": round(self._ruido or 0.0, 1),
            "ganho": round(self.ganho, 2),
            "empilhados": self._n,
            "ligado": self.ligado,
        }


def descrever_luz(estado: dict | None) -> str:
    """Linha de HUD (só ASCII: o cv2.putText não desenha acentos)."""
    if not estado or not estado.get("ligado"):
        return "Luz: realce desligado"
    base = f"Luz: {estado['nivel'].upper()} (brilho {estado['brilho']:.0f}, ruido {estado['ruido']:.1f})"
    if estado["nivel"] == "boa":
        return base
    realce = f" | realce: ganho {estado['ganho']:.1f}x, {estado['empilhados']} quadros"
    if estado["nivel"] == "critica":
        return base + realce + " | BAIXA CONFIANCA - ilumine a tora"
    return base + realce
