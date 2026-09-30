"""
posicao.py — posição da máquina (GNSS) para gravar junto com cada tora.

A posição entra como a densidade: um dado com PROVENIÊNCIA, nunca
obrigatório. Sem receptor, sem satélite ou com leitura velha, a tora é
gravada normalmente, só sem coordenada (e fora do mapa).

De onde vem a posição — sempre sentenças NMEA 0183, o formato que
receptores GNSS falam há décadas:

  - porta serial ("COM5", "/dev/ttyUSB0"): o GNSS da máquina, se exposto
    em serial, ou um receptor USB. pos_fonte = "gnss_serial".
  - arquivo (.nmea / .txt): uma trilha REAL gravada antes, reproduzida no
    ritmo em que foi gravada e em loop — o equivalente do `--fonte video.mp4`
    da câmera. pos_fonte = "gnss_log" (o dashboard declara isso na tela).
  - "windows": o serviço de Localização do Windows (no notebook da maquete,
    que faz o papel da máquina). Sem GPS, o Windows estima pela rede Wi-Fi:
    precisão de dezenas a centenas de metros, que ele mesmo informa e nós
    gravamos em pos_precisao_m. pos_fonte = "windows_localizacao".
    Precisa da Localização ligada em Configurações > Privacidade.

Só GGA e RMC são lidas (todos os receptores mandam pelo menos uma delas).
Sentença com checksum errado é descartada — linha serial corrompida não
vira coordenada.

A precisão NÃO é inventada: gravamos o HDOP e o número de satélites que o
próprio receptor informa. E a coordenada é da máquina, não da árvore: a
árvore está no alcance da grua (~10 m), o que sobra para mapa de talhão.

Nada aqui depende de rede: GNSS é recebido do satélite, offline.
"""

import math
import subprocess
import threading
import time
from pathlib import Path

FONTE_SERIAL = "gnss_serial"
FONTE_LOG = "gnss_log"
FONTE_WINDOWS = "windows_localizacao"
ORIGEM_WINDOWS = "windows"  # valor de --gnss / gnss_porta que liga a Localização do Windows

# Lê a Localização do Windows (API System.Device, presente em todo Windows
# 10/11 com o PowerShell 5.1) e escreve UMA linha por segundo:
#   POS;<lat>;<lon>;<precisão em m>   |   SEM_FIX;<status>   |   NEGADO
# Números em cultura invariante (ponto decimal), para o Python ler sem
# depender do idioma do Windows.
_SCRIPT_WINDOWS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Device
$ci = [Globalization.CultureInfo]::InvariantCulture
$w = New-Object System.Device.Location.GeoCoordinateWatcher([System.Device.Location.GeoPositionAccuracy]::High)
[void]$w.TryStart($false, [TimeSpan]::FromSeconds(10))
while ($true) {
  $l = $w.Position.Location
  if ($w.Permission -eq 'Denied') { $linha = 'NEGADO' }
  elseif ($l.IsUnknown) { $linha = 'SEM_FIX;' + $w.Status }
  else { $linha = 'POS;' + $l.Latitude.ToString('R', $ci) + ';' + $l.Longitude.ToString('R', $ci) + ';' + $l.HorizontalAccuracy.ToString('R', $ci) }
  [Console]::Out.WriteLine($linha)
  [Console]::Out.Flush()
  Start-Sleep -Seconds 1
}
"""


# ============================================================
# PARSER NMEA (funções puras — testadas sem receptor)
# ============================================================

def checksum_ok(linha: str) -> bool:
    """`$<corpo>*HH`: HH é o XOR de todos os caracteres do corpo, em hexa."""
    linha = linha.strip()
    if not linha.startswith("$") or "*" not in linha:
        return False
    corpo, _, cs = linha[1:].rpartition("*")
    if len(cs) != 2:
        return False
    calculado = 0
    for c in corpo:
        calculado ^= ord(c)
    try:
        return calculado == int(cs, 16)
    except ValueError:
        return False


def _graus(valor: str, hemisferio: str, digitos_graus: int) -> float | None:
    """NMEA dá (d)ddmm.mmmm + hemisfério; devolve graus decimais com sinal (S e W negativos)."""
    if not valor or hemisferio not in ("N", "S", "E", "W"):
        return None
    try:
        graus = int(valor[:digitos_graus])
        minutos = float(valor[digitos_graus:])
    except ValueError:
        return None
    if minutos >= 60.0:
        return None
    decimal = graus + minutos / 60.0
    return -decimal if hemisferio in ("S", "W") else decimal


def _float(valor: str) -> float | None:
    try:
        return float(valor) if valor else None
    except ValueError:
        return None


def _int(valor: str) -> int | None:
    try:
        return int(valor) if valor else None
    except ValueError:
        return None


def ler_sentenca(linha: str) -> dict | None:
    """
    Interpreta UMA sentença GGA ou RMC (qualquer talker: GP, GN, GL, GA, BD).
    Devolve None para o que não é GGA/RMC ou tem checksum errado. Para
    GGA/RMC válida devolve {"tipo", "hora", "valida", "lat", "lon",
    "hdop", "satelites"} — `valida=False` quando o receptor diz que não tem fix.
    """
    if not checksum_ok(linha):
        return None
    campos = linha.strip()[1:].rpartition("*")[0].split(",")
    identificador = campos[0]
    if len(identificador) != 5:
        return None
    tipo = identificador[2:]

    if tipo == "GGA" and len(campos) >= 10:
        # hora, lat, N/S, lon, E/W, qualidade (0 = sem fix), satélites, HDOP, altitude...
        qualidade = _int(campos[6]) or 0
        lat = _graus(campos[2], campos[3], 2)
        lon = _graus(campos[4], campos[5], 3)
        return {
            "tipo": "GGA",
            "hora": campos[1] or None,
            "valida": qualidade > 0 and lat is not None and lon is not None,
            "lat": lat,
            "lon": lon,
            "hdop": _float(campos[8]),
            "satelites": _int(campos[7]),
        }

    if tipo == "RMC" and len(campos) >= 7:
        # hora, status (A = válido, V = sem fix), lat, N/S, lon, E/W, ...
        lat = _graus(campos[3], campos[4], 2)
        lon = _graus(campos[5], campos[6], 3)
        return {
            "tipo": "RMC",
            "hora": campos[1] or None,
            "valida": campos[2] == "A" and lat is not None and lon is not None,
            "lat": lat,
            "lon": lon,
            "hdop": None,
            "satelites": None,
        }
    return None


def _segundos_do_dia(hora: str | None) -> float | None:
    """'hhmmss.ss' -> segundos desde 00:00 (para reproduzir um log no ritmo original)."""
    if not hora or len(hora) < 6:
        return None
    try:
        return int(hora[0:2]) * 3600 + int(hora[2:4]) * 60 + float(hora[4:])
    except ValueError:
        return None


# ============================================================
# LEITOR EM THREAD
# ============================================================

class LeitorPosicao:
    """
    Lê NMEA numa thread própria e guarda a ÚLTIMA posição válida. O loop de
    inspeção só chama `atual()` (não bloqueia, não toca em porta serial):
    receptor travado ou desconectado nunca atrasa a medição da tora.
    """

    def __init__(self, origem: str, baud: int = 9600):
        self.origem = origem
        self.baud = baud
        self.e_windows = origem.strip().lower() == ORIGEM_WINDOWS
        self.e_arquivo = not self.e_windows and Path(origem).is_file()
        if self.e_windows:
            self.fonte = FONTE_WINDOWS
        else:
            self.fonte = FONTE_LOG if self.e_arquivo else FONTE_SERIAL
        self._proc: subprocess.Popen | None = None  # PowerShell da Localização do Windows
        self._lock = threading.Lock()
        self._ultima: dict | None = None      # posição válida mais recente
        self._recebida_em = 0.0               # time.monotonic() da última posição válida
        self._tem_gga = False                 # receptor manda GGA: ela manda (tem HDOP/satélites)
        self._estado = "iniciando"            # para o HUD: iniciando / sem_fix / ok / sem_receptor
        self._parar = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------- interface usada pelo main.py ----------

    def iniciar(self) -> "LeitorPosicao":
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gnss")
        self._thread.start()
        return self

    def parar(self, timeout_s: float = 2.0) -> None:
        self._parar.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()  # destrava o readline da thread e não deixa PowerShell órfão
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def atual(self, validade_s: float) -> dict | None:
        """Última posição, se tiver no máximo `validade_s` segundos; senão None."""
        with self._lock:
            if self._ultima is None:
                return None
            idade = time.monotonic() - self._recebida_em
            if idade > validade_s:
                return None
            return {**self._ultima, "idade_s": round(idade, 1)}

    def descricao(self, validade_s: float) -> str:
        """Linha de HUD (só ASCII: o cv2.putText não desenha acentos)."""
        p = self.atual(validade_s)
        rotulo = "log gravado" if self.e_arquivo else "Windows (Wi-Fi)" if self.e_windows else self.origem
        if p is not None:
            extra = ""
            if p.get("precisao_m") is not None:
                extra += f" | +-{p['precisao_m']:.0f} m"
            if p.get("hdop") is not None:
                extra += f" | HDOP {p['hdop']:.1f}"
            if p.get("satelites") is not None:
                extra += f" | {p['satelites']} sat"
            return f"GNSS: {p['lat']:.5f}, {p['lon']:.5f}{extra} | {rotulo}"
        with self._lock:
            estado = self._estado
        if estado == "sem_receptor":
            if self.e_windows:
                return "GNSS: Localizacao do Windows indisponivel - toras sem posicao"
            return f"GNSS: sem receptor ({rotulo}) - toras sem posicao"
        if self.e_windows:
            return "GNSS: aguardando Localizacao do Windows - toras sem posicao"
        return f"GNSS: aguardando satelites ({rotulo}) - toras sem posicao"

    # ---------- interno ----------

    def processar_linha(self, linha: str) -> dict | None:
        """Aplica uma linha NMEA ao estado. Público para os testes."""
        s = ler_sentenca(linha)
        if s is None:
            return None
        if s["tipo"] == "GGA":
            self._tem_gga = True
        elif self._tem_gga:
            return s  # quem manda GGA tem HDOP/satélites: RMC só serviria para piorar
        with self._lock:
            if s["valida"]:
                self._ultima = {
                    "lat": round(s["lat"], 7),
                    "lon": round(s["lon"], 7),
                    "hdop": s["hdop"],
                    "satelites": s["satelites"],
                    "precisao_m": None,  # NMEA dá HDOP, não metros
                    "fonte": self.fonte,
                }
                self._recebida_em = time.monotonic()
                self._estado = "ok"
            else:
                # O receptor disse explicitamente que perdeu o fix: a posição
                # antiga deixa de ser "a posição atual" na hora.
                self._ultima = None
                self._estado = "sem_fix"
        return s

    def processar_windows(self, linha: str) -> None:
        """Aplica uma linha do script da Localização do Windows. Público para os testes."""
        partes = linha.strip().split(";")
        if partes[0] == "POS" and len(partes) == 4:
            try:
                lat, lon, precisao = float(partes[1]), float(partes[2]), float(partes[3])
            except ValueError:
                return
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return
            with self._lock:
                self._ultima = {
                    "lat": round(lat, 7),
                    "lon": round(lon, 7),
                    "hdop": None,
                    "satelites": None,
                    # O Windows diz o raio de incerteza em metros; NaN = não informou
                    "precisao_m": round(precisao, 1) if math.isfinite(precisao) and precisao > 0 else None,
                    "fonte": self.fonte,
                }
                self._recebida_em = time.monotonic()
                self._estado = "ok"
        elif partes[0] == "SEM_FIX":
            with self._lock:
                self._ultima = None
                self._estado = "sem_fix"
        elif partes[0] == "NEGADO":
            with self._lock:
                self._ultima = None
            self._mudar_estado(
                "sem_receptor",
                "⚠️  Localização do Windows NEGADA. Ligue em Configurações > Privacidade e segurança > "
                "Localização (e 'Permitir que aplicativos da área de trabalho acessem sua localização'). "
                "Toras seguem sem posição.",
            )

    def _mudar_estado(self, estado: str, mensagem: str) -> None:
        with self._lock:
            mudou = self._estado != estado
            self._estado = estado
        if mudou:
            print(mensagem)

    def _loop(self) -> None:
        while not self._parar.is_set():
            try:
                if self.e_windows:
                    self._ler_windows()
                elif self.e_arquivo:
                    self._reproduzir_arquivo()
                else:
                    self._ler_serial()
            except ImportError:
                self._mudar_estado(
                    "sem_receptor",
                    "⚠️  GNSS: pacote 'pyserial' não instalado (pip install pyserial). Toras seguem sem posição.",
                )
                return
            except Exception as e:  # porta sumiu, cabo solto, arquivo ilegível...
                with self._lock:
                    self._ultima = None
                self._mudar_estado(
                    "sem_receptor",
                    f"📡 GNSS indisponível em {self.origem} ({e}). Tentando de novo a cada 3 s; toras seguem sem posição.",
                )
                self._parar.wait(3.0)

    def _ler_serial(self) -> None:
        import serial  # só aqui: sem receptor configurado, o pyserial nem é necessário

        with serial.Serial(self.origem, self.baud, timeout=1.0) as porta:
            print(f"🛰️  GNSS conectado em {self.origem} ({self.baud} baud).")
            with self._lock:
                if self._estado == "sem_receptor":
                    self._estado = "iniciando"
            while not self._parar.is_set():
                bruto = porta.readline()
                if bruto:
                    self.processar_linha(bruto.decode("ascii", errors="ignore"))

    def _ler_windows(self) -> None:
        """Roda o script da Localização do Windows num PowerShell filho e lê uma linha por segundo."""
        print("🛰️  Posição pela Localização do Windows (Wi-Fi/rede; precisão informada pelo próprio Windows).")
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", _SCRIPT_WINDOWS],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
            errors="ignore",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for linha in self._proc.stdout:
                if self._parar.is_set():
                    return
                self.processar_windows(linha)
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
        if not self._parar.is_set():
            raise RuntimeError("o PowerShell da Localização do Windows encerrou")

    def _reproduzir_arquivo(self) -> None:
        """Reproduz o log no ritmo em que foi gravado (pela hora das sentenças), em loop."""
        print(f"🛰️  GNSS reproduzindo log gravado: {self.origem} (posição declarada como '{FONTE_LOG}').")
        while not self._parar.is_set():
            hora_anterior = None
            lidas = 0
            with open(self.origem, "r", encoding="ascii", errors="ignore") as f:
                for linha in f:
                    if self._parar.is_set():
                        return
                    s = self.processar_linha(linha)
                    if s is None:
                        continue
                    lidas += 1
                    agora = _segundos_do_dia(s["hora"])
                    if agora is not None and hora_anterior is not None and agora != hora_anterior:
                        # Espera o intervalo real entre as leituras (limitado:
                        # um buraco no log não congela a reprodução).
                        delta = agora - hora_anterior
                        if delta < 0:
                            delta += 86400  # virou meia-noite UTC
                        self._parar.wait(min(max(delta, 0.0), 2.0))
                    elif agora is None:
                        self._parar.wait(1.0)
                    if agora is not None:
                        hora_anterior = agora
            if lidas == 0:
                raise ValueError("nenhuma sentença GGA/RMC válida no arquivo")
