# Walkthrough — Arquitetura de Qualidade da Madeira (Refatorada)

Em total alinhamento com os retornos técnicos da **John Deere** e o **Regulamento (Item 1.3)**, o sistema calcula os indicadores de qualidade da madeira combinando **Visão Computacional na Borda** (para medição geométrica e defeitos) com **Lookup Cadastral por Clone** (para densidade de referência de laboratório).

---

## 🛠️ Alterações e Melhorias Realizadas

### 1. Lookup de Densidade por Material Genético (`data/clones_densidade.json`)
- **Sem Fórmulas Inventadas**: A densidade básica é obtida diretamente por busca cadastral por clone (`obter_densidade_estimada`), sem fórmulas arbitrárias derivadas do DAP.
- **Metadados Transparentes**: O arquivo `data/clones_densidade.json` inclui o aviso de que os valores são referências de laboratório/demonstração.

### 2. Cálculo da Massa Seca de Madeira (`calcular_massa_seca_kg`)
- **Implementação do Indicador de Massa**:
  $$\text{Massa Seca Estimada (kg)} = \text{Volume Útil Medido (m}^3\text{)} \times \text{Densidade Cadastrada (kg/m}^3\text{)}$$
- Salvo automaticamente nas tabelas de indicadores no SQLite e PostgreSQL (`metodo: calculado_volume_x_densidade`).

### 3. Remoção de Código Legado
- Removido completamente o código não utilizado do sensor de força (ADC/spidev), mantendo o código enxuto e focado no sensor ultrassônico HC-SR04 para conversão de pixels em centímetros reais (GSD).

---

## 🧪 Validação dos Testes

Execução do simulador florestal:
```bash
.venv/bin/python simular_cenario.py --num-toras 2
```

**Resultado dos Registros Armazenados no SQLite (`omni_root_local.db`):**
- **Tora 1:** Volume `0.053 m³` $\times$ Densidade `510.0 kg/m³` $\rightarrow$ **Massa Seca:** `27.0 kg`
- **Tora 2:** Volume `0.158 m³` $\times$ Densidade `510.0 kg/m³` $\rightarrow$ **Massa Seca:** `80.6 kg`
- **Indicadores Processados:** `densidade`, `massa_seca`, `altura`, `diametro`, `tortuosidade`, `porcentagem_casca`, `volume_util`, `apodrecimento_pragas`.

---

## 🎤 Discurso Ajustado para o Pitch (5 Minutos)

Use esta frase refinada no pitch para evitar exageros e transmitir 100% de honestidade técnica:

> *"Os dados de Diâmetro, Altura, Volume Útil, Tortuosidade, Porcentagem de Casca e Defeitos Visuais (nós e podridão) são medidos em tempo real pela nossa **Visão Computacional na Câmera**. Conforme nos foi explicado por e-mail pelo contato técnico da John Deere, a **densidade** é um atributo genético do Clone cadastrado em laboratório. Nosso software cruza o Clone do Talhão com o cadastro de densidade para calcular a **Massa Seca de Madeira** ($Massa = Volume_{CV} \times Densidade_{Clone}$). Em ambiente fabril real, esse cadastro é alimentado automaticamente pelos laudos oficiais de laboratório da fábrica."*
