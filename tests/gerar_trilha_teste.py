"""
gerar_trilha_teste.py — trilha NMEA SINTÉTICA para testar o mapa sem receptor

Gera um arquivo .nmea com a máquina andando em faixas ("vai e volta") sobre
um talhão pequeno, a ~0,5 m/s, 1 leitura por segundo — o bastante para as
toras caírem em várias zonas de 25 m do mapa.

ATENÇÃO: é uma trilha INVENTADA, só para testar a tela e o pipeline. Na
apresentação, use uma trilha REAL gravada com um receptor/app de GPS (o
dashboard mostra "Trilha gravada" para qualquer arquivo — quem garante que
ela é real são vocês).

USO (raiz do repositório):
    python tests/gerar_trilha_teste.py                       # -> tests/trilha_teste_sintetica.nmea
    python main.py --gui --gnss tests/trilha_teste_sintetica.nmea
"""

import argparse
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


def nmea(corpo: str) -> str:
    cs = 0
    for c in corpo:
        cs ^= ord(c)
    return f"${corpo}*{cs:02X}"


def graus_para_nmea(valor: float, digitos_graus: int) -> tuple[str, bool]:
    negativo = valor < 0
    v = abs(valor)
    graus = int(v)
    minutos = (v - graus) * 60.0
    return f"{graus:0{digitos_graus}d}{minutos:07.4f}", negativo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, default=-23.5641, help="canto do talhão (padrão: região da FIAP Paulista)")
    p.add_argument("--lon", type=float, default=-46.6524)
    p.add_argument("--faixas", type=int, default=4, help="quantas faixas de vai-e-volta")
    p.add_argument("--comprimento-m", type=float, default=120.0, help="comprimento de cada faixa")
    p.add_argument("--espacamento-m", type=float, default=25.0, help="distância entre faixas")
    p.add_argument("--velocidade", type=float, default=0.5, help="m/s")
    p.add_argument("--saida", type=str, default=str(Path(__file__).parent / "trilha_teste_sintetica.nmea"))
    args = p.parse_args()

    m_por_grau_lat = 111_320.0
    m_por_grau_lon = 111_320.0 * math.cos(math.radians(args.lat))
    hora = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
    linhas = []
    passos = int(args.comprimento_m / args.velocidade)
    for f in range(args.faixas):
        for s in range(passos + 1):
            x = s * args.velocidade if f % 2 == 0 else args.comprimento_m - s * args.velocidade
            y = -f * args.espacamento_m  # faixas para o sul
            lat = args.lat + y / m_por_grau_lat
            lon = args.lon + x / m_por_grau_lon
            lat_s, lat_neg = graus_para_nmea(lat, 2)
            lon_s, lon_neg = graus_para_nmea(lon, 3)
            hhmmss = hora.strftime("%H%M%S") + ".00"
            linhas.append(nmea(
                f"GPGGA,{hhmmss},{lat_s},{'S' if lat_neg else 'N'},{lon_s},{'W' if lon_neg else 'E'},"
                f"1,09,0.9,760.0,M,-5.0,M,,"
            ))
            hora += timedelta(seconds=1)

    Path(args.saida).write_text("\n".join(linhas) + "\n", encoding="ascii")
    print(f"Trilha SINTÉTICA de teste: {len(linhas)} leituras (~{len(linhas) // 60} min) -> {args.saida}")
    print("Só para testar a tela. Na demo, use uma trilha real gravada.")


if __name__ == "__main__":
    main()
