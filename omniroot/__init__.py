"""
omniroot — código da máquina de campo, um módulo por responsabilidade.

O ponto de entrada é o main.py (argumentos, threads e loop de captura).
O caminho de um quadro da câmera até o banco local:

    fontes        câmera / vídeo / pasta de imagens
      │
    luz           medidor de luz + realce em pouca luz (empilhamento, ganho)
      │
    analise       pipeline de UM quadro (função pura):
      ├─ segmentacao    máscara da tora e portão "isso é madeira?"
      ├─ deteccao       modelo de defeitos (YOLO) + filtros de falso positivo
      ├─ medidas        escala px->cm, diâmetro, casca, tortuosidade, volume, massa
      ├─ inventario     densidade por clone (cache sincronizado do Postgres)
      └─ classificacao  saúde da tora e aprovado / quarentena / reprovado
      │
    evento        uma tora = um registro (consolida os quadros da mesma tora)
      │
    registro      grava no SQLite local (+ posicao: GNSS/NMEA/Windows)
      │
    banco_local   schema/migração compartilhados com o sync_daemon.py

    config  (Config e config.json)   hud (desenho na tela)   stream (câmera ao vivo no dashboard)

Garantia de comportamento: tests/testar_equivalencia.py compara todo o
pipeline com uma referência gravada (números, textos do HUD e pixels).
"""
