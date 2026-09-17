# Omni-Root — Challenge 2026 (John Deere / Suzano)

**Desafio recebido:** "Qualidade da madeira" (enunciado curto, sem detalhar espécie).

**Nossa proposta:** sistema de inspeção de tora por visão computacional, acoplado
à máquina de colheita, funcionando **offline em campo** e sincronizando os dados
para um banco central quando houver internet.

Este README existe pra qualquer pessoa do time se atualizar rápido sem reler tudo
do zero. Se você sumiu um tempo do projeto, comece por aqui.

---

## 1. Ideia em uma frase

Câmera na máquina → visão computacional avalia a tora em tempo real → grava local
(SQLite, funciona sem internet) → sincroniza para um PostgreSQL central quando há
rede → dashboard mostra, analisa e exporta.

---

## 2. Arquitetura

```
        MÁQUINA DE CAMPO (Windows)                    CENTRAL (escritório / cloud)
 ┌──────────────────────────────────────┐          ┌────────────────────────────┐
 │ main.py                              │          │                            │
 │  ├─ Câmera (OpenCV)   thread 1       │          │   PostgreSQL               │
 │  ├─ YOLO + indicadores  thread 2     │  sync    │   desafio_madeira          │
 │  └─ SQLite local (omni_root_local.db)│ ───────► │        ▲                   │
 │                                      │ quando   │        │ leitura           │
 │         sync_daemon.py ──────────────┼─ há rede │   Dashboard (repo à parte) │
 └──────────────────────────────────────┘          └────────────────────────────┘
```

**Só a câmera é hardware obrigatório.** Sensor de força e ADC (que existiam numa
versão anterior, para "medir" densidade) foram **removidos** — densidade não vem
de sensor, vem de tabela de referência (seção 7). A distância câmera-tora é uma
**constante calibrada** em `config.json`, não um sensor ultrassônico.

**Por que Windows e não Raspberry Pi:** as máquinas reais da John Deere rodam
Windows Embedded. Rodar a maquete em Windows é mais fiel ao ambiente real, e sem
dependência de sensor físico não há motivo para manter o Raspberry na demo.

**A interface é um repositório separado:**
[omni-root-dashboard](https://github.com/Omni-Root/omni-root-dashboard) — ele lê
o **mesmo** PostgreSQL que o `sync_daemon.py` alimenta. Os nomes das variáveis
`PG_*` são idênticos nos dois `.env` de propósito.

---

## 3. Como rodar

### Preparar o ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> ⚠️ O `requirements.txt` pede `opencv-python-headless`, que **não tem janela**.
> Para usar o modo com visualização (`--gui`), instale a versão com GUI:
> `pip install opencv-python`.

> ⚠️ Os pesos do modelo (`models/wood_best.pt`, ~84 MB) **não estão no Git**
> (`*.pt` está no `.gitignore`). Peça o arquivo ao time e coloque em `models/`.
> Sem ele, o `main.py` cai no `yolov8n.pt` genérico e não detecta defeito de
> madeira nenhum.

### Escala automática — imprimir o marcador (uma vez)

```powershell
python calibrar.py --gerar-marcador
```

Gera `marcador_aruco_5cm.png`. Imprima em **100%** (sem "ajustar à página"),
confira com régua que o quadrado preto tem 5 cm e cole num papelão ao lado
da tora (na garra real, fixo no cabeçote). O `main.py` detecta o marcador em
cada frame e calcula cm/px **sozinho** — sem ninguém clicar em nada, e
robusto a mudança de posição da câmera. O quadrado magenta "escala" na tela
confirma que ele foi visto.

Sem marcador no frame, a escala cai para `cm_por_px` do config (calibração
de fábrica da montagem; `python calibrar.py --regua` obtém o valor na
maquete) e, por último, para a fórmula de GSD com óptica genérica.

> Em máquina real a fonte preferencial de diâmetro/comprimento nem é a
> câmera: o cabeçote do harvester já mede os dois (sensores nas facas e no
> rolo) e exporta no StanForD `.hpr`. A câmera entra para o que o cabeçote
> **não** vê: casca residual, defeitos e tortuosidade.

### Exportar o modelo para CPU (uma vez por máquina)

```powershell
python exportar_modelo.py
```

O harvester não tem GPU. O `.pt` rodando direto no PyTorch custa ~2,4 s por
frame em CPU; o mesmo modelo em **OpenVINO INT8** (runtime da Intel para
CPU, quantizado com as fotos de `capturas/`) custa ~0,5 s. O `main.py`
carrega `models/wood_best_int8_openvino_model` automaticamente se existir.
O export (~170 MB) fica fora do Git — cada máquina gera o seu.

### Inspecionar com a câmera

```powershell
python main.py --gui
```

Abre a janela com o vídeo ao vivo e as detecções; grava uma inspeção no SQLite a
cada `intervalo_captura_seg`. Sai com **`q`** na janela ou `Ctrl+C`.
Sem `--gui`, roda em modo silencioso (só console + banco).

Na tela: caixas **coloridas** são defeitos que contaram; caixas **cinza
"(ignorado)"** são detecções que o modelo fez mas os filtros descartaram
(fora da tora ou na faixa de casca, área mínima, não persistiu). A linha
branca é o contorno segmentado da tora; a terceira linha do HUD diz se a
câmera está vendo a **seção** (rodela) ou a **lateral** da tora — os
indicadores mudam de significado (ver seção 5). Tecle **`s`** para salvar o
frame cru em `capturas/` (serve para o dataset de eucalipto e para testar
offline com `--fonte .\capturas`).

**Mão no frame:** pele e casca têm a mesma cor para a câmera; segurar a
peça com a mão dentro do enquadramento gruda a mão na máscara e distorce
casca e diâmetro. Na demo, apoie a peça (mesa neutra ou garra da maquete).

**Fundo da maquete:** a segmentação separa madeira de fundo pela cor, então
use uma superfície **neutra** (cinza, branca, preta, metal). Mesa de madeira
ou terra marrom se confundem com a casca — limitação conhecida.

**Plano B para a apresentação** — se a webcam falhar no palco, o mesmo
código roda sobre um vídeo gravado ou uma pasta de fotos, sem nada mocado:

```powershell
python main.py --gui --fonte .\demo.mp4
python main.py --gui --fonte .\fotos_demo\
```

Outras flags: `--config-file <caminho>`, `--clone <ID>`, `--idade <anos>`.

### Subir o PostgreSQL central (local, para testes)

```bash
docker compose up -d
```

Lê as mesmas variáveis do `.env`. Na primeira subida, aplica
`Banco de dados/setup_completo.sql` automaticamente.

### Sincronizar campo → central

```powershell
python sync_daemon.py
```

### Testar o modelo sem câmera

```powershell
python tests\testar_modelo_eucalipto.py --pasta .\fotos_eucalipto --modelo .\models\wood_best.pt
```

Roda o modelo numa pasta de imagens, salva as versões anotadas em
`resultados_teste/` e resume o que não foi detectado ou saiu com confiança baixa.

### Simular a operação inteira sem câmera

```powershell
python tests\simular_cenario.py --num-toras 15 --pasta .\fotos_eucalipto
```

Roda o **mesmo** `analisar_frame()` do `main.py` sobre imagens de disco (ou
frames sintéticos, se a pasta não existir) e grava no SQLite como a máquina
faria. Útil para popular o Postgres/dashboard com dados produzidos pelo código
real (`--sem-banco` só analisa).

---

## 4. O que cada peça faz

### `main.py` — inspeção em campo
O coração do projeto. Roda em **três threads** para a imagem nunca congelar:

- **Thread principal**: só captura e exibe o vídeo, na velocidade da câmera.
- **Thread do modelo**: roda o YOLO no frame mais recente, no ritmo que a
  CPU der (~0,5 s com OpenVINO INT8), e publica as caixas.
- **Thread de análise**: visão clássica (~70 ms — contorno, casca, diâmetro,
  rachadura, status) a ~10 fps, reaproveitando as últimas caixas do modelo;
  grava no SQLite a cada `intervalo_captura_seg`.

Assim a inferência (pesada na CPU) nunca bloqueia o vídeo — no máximo as caixas
de detecção aparecem uma fração de segundo atrás da imagem. O buffer da câmera é
fixado em 1 frame para não acumular atraso.

Antes de ir ao modelo, cada frame passa por `preprocessar_para_modelo()`:
**escala de cinza + CLAHE**, exatamente o mesmo pré-processamento usado no
treino. Isso importa muito: mandar o frame BGR cru degrada a precisão **sem
gerar erro nenhum** — o tipo de bug que passa despercebido. Se mudarem o
`clipLimit`/`tileGridSize` no notebook de treino, mudem aqui também.

As medições por OpenCV (contorno, porcentagem de casca) continuam usando o frame
**colorido original**, onde a informação de cor ajuda.

### `sync_daemon.py` — sincronização
Substitui o antigo `sync.go` (o Go existia para compilar sem CGO no Raspberry;
como o Raspberry saiu do escopo e a máquina já tem Python por causa do `main.py`,
virou uma linguagem a menos para manter). Ele:

- verifica se o PostgreSQL responde (proxy para "tem internet?");
- envia as toras com `sync_status = 0`, de forma **idempotente**
  (`ON CONFLICT (uuid_local) DO NOTHING`), então um retry após queda de rede não
  duplica registro;
- só marca como sincronizado **depois** que a tora inteira (registro +
  indicadores + defeitos) entrou — nunca fica "meio sincronizada";
- **também baixa** a tabela `clones_densidade` do Postgres e regrava o cache
  local `data/clones_densidade.json`. Por isso esse arquivo **não deve ser
  editado à mão**: a fonte de verdade é o banco central. O `main.py`
  **recarrega o cache sozinho** quando o arquivo muda em disco
  (`inventario_atual()`, confere o mtime 1×/s) e imprime a mudança
  (`📥 ... SP3108: 490.0 -> 512.0 kg/m3`) — cadastrou no banco, em ~1 s a
  máquina já usa o valor novo, sem reiniciar. É a demonstração que a JD pediu.

### `config.json` — parâmetros da operação
Identidade da máquina e do talhão, clone, caminhos, limiares da IA
(`conf_threshold`, `imgsz`), índice da câmera, intervalo entre gravações e:

| Chave | O que faz |
|---|---|
| `marcador_aruco_cm` | Lado (cm) do marcador ArUco impresso. Se aparecer no frame, a escala px→cm vem dele, automaticamente. `0` desliga. |
| `comprimento_corte_cm` | Comprimento de traçamento do talhão (padrão 600 = 6 m). Usado quando a câmera vê só a seção da tora. |
| `roi` | `[x, y, w, h]` em fração do frame: só essa região vai para o modelo. `null` = frame inteiro. (`calibrar.py --roi`) |
| `cm_por_px` | Escala fixa da montagem, usada quando não há marcador no frame (`calibrar.py --regua`). `0` = fórmula de GSD com `distancia_camera_tora_cm`, `distancia_focal_mm` e `largura_sensor_mm`. |
| `filtro_area_minima` | Descarta caixas menores que essa fração do frame (`0` desliga). |
| `filtro_dentro_contorno` | Descarta detecção cujo centro cai fora do contorno da tora. |
| `filtro_persistencia` | Nº de análises consecutivas em que o defeito precisa aparecer (`1` desliga). |

### Demais diretórios

| Caminho | O que é |
|---|---|
| `data/clones_densidade.json` | Cache da tabela de densidade por clone (com fontes documentadas) |
| `data/inventario_johndeere.json` | DAP/idade por clone, da planilha real da JD (contexto dendrométrico) |
| `Banco de dados/` | Schemas SQLite e PostgreSQL, seed de teste e `setup_completo.sql` |
| `models/` | Pesos do YOLO (`wood_best.pt`) e imagens de avaliação do treino |
| `tests/` | Scripts de teste sem hardware (ver seção 3) |
| `OmniRoot_Challenge_*.ipynb` | Notebooks de treino do modelo (Colab e VSCode local) |
| `docker-compose.yml` | PostgreSQL local para testar o `sync_daemon.py` |

---

## 5. Os indicadores de qualidade

| Indicador | Como é obtido |
|---|---|
| Diâmetro | Câmera: **seção** → diâmetro equivalente pela área segmentada; **lateral** → lado menor do retângulo de área mínima. Escala px→cm: marcador ArUco (automático) > `cm_por_px` > GSD |
| Comprimento | **Lateral** → lado maior medido. **Seção** → não é visível na imagem; usa `comprimento_corte_cm` (o comprimento fixo de traçamento do talhão — o mesmo que o cabeçote usa para cortar) e marca `metodo = comprimento_tracamento_config` |
| Tortuosidade | OpenCV — **flecha do eixo da tora / comprimento** (%). O eixo é a linha dos pontos médios entre as bordas, fatia a fatia; 0 = reta. Só na vista **lateral**; na seção grava 0 com `metodo = nao_aplicavel_secao` em vez de inventar número. *(A versão anterior media meio diâmetro, não curvatura: tora reta e grossa dava ~60.)* |
| Porcentagem de casca | OpenCV — **casca residual**: % da superfície da tora (dentro do contorno) no grupo escuro de um Otsu — madeira descascada é clara, casca é escura. *(A versão anterior media um anel de largura fixa em pixels.)* |
| Apodrecimento / pragas | YOLO — ver seção 8, é o item mais crítico do projeto |
| Rachadura radial (seção) | OpenCV — black-hat morfológico no miolo + filtro geométrico (longa, fina, reta, passando pelo centro). Cobre o que o modelo atual não vê: rachadura de secagem na face de corte. Aparece como `Crack (cv)` na tela |
| Densidade | **Lookup por clone/material genético** (não sensor, não fórmula) — seção 7 |
| Volume útil | Geometria (cilindro), descontado pela severidade dos defeitos |
| Massa seca estimada | Volume útil × densidade de referência |

São gravados **8 indicadores por inspeção**. Importante ao apresentar: volume é
**medido**, densidade é **estimada** — logo a massa seca herda a incerteza da
densidade. É uma estimativa, não uma pesagem.

---

## 6. Classificação por severidade

Nem todo defeito pesa igual para a indústria de celulose, então a triagem é
ponderada em vez de "detectou algo = reprovado":

| Nível | Classes | Peso |
|---|---|---|
| **Grave** | `Dead_Knot`, `Knot_missing`, `knot_with_crack`, `resin` | 1.0 |
| **Moderado** | `Crack`, `Marrow`, `Quartzity` | 0.5 |
| **Leve** | `Live_Knot` | 0.2 |

Os nomes são exatamente as classes do modelo treinado. *(Uma versão anterior
procurava por `apodrecimento`/`praga`/`rot`, que **não existem** no modelo — ou
seja, aquela regra nunca disparava.)*

Regras, em ordem: defeito grave com confiança ≥ 70% reprova; defeito extenso
(≥ 5% da área, ignorando leves) reprova; saúde abaixo de 60% reprova; tora limpa
com saúde ≥ 85% aprova; **só defeitos leves com saúde alta também aprova**; o
resto vai para quarentena (revisão manual).

Essa última regra é a diferença prática do modelo ponderado: uma tora só com nós
vivos — o defeito mais comum e de menor impacto — não gera fila de revisão
desnecessária.

Os pesos são uma **decisão de projeto** baseada em características conhecidas da
madeira, não uma norma publicada. Se a Suzano/JD fornecer o critério oficial, é
só ajustar os conjuntos no topo de `main.py` — nada mais muda.

---

## 7. Densidade — por que é assim, e o que dizer se perguntarem

Densidade básica da madeira **não dá para medir com câmera**. Dois caminhos foram
testados e **descartados de propósito** — importante saber para não repetir:

- ❌ Fórmula inventada a partir de DAP/idade (`440 + DAP×3.8 + idade×4.5`) — sem
  base científica, e contraria o que o contato da JD disse por e-mail (densidade
  é definida por material genético, não por medida de campo).
- ❌ Valor "mockado" fixo por clone, sem fonte — pareceria dado real sem ser.

**O que fizemos:** os clones cadastrados usam valores de literatura reais e
citáveis (faixa de **485–490 kg/m³**) para o híbrido comercial *Eucalyptus
grandis × E. urophylla*, o material genético mais comum em plantios de celulose
no Brasil. Não é o valor exato de nenhum clone específico — isso é dado
proprietário da Suzano, como eles confirmaram por e-mail — mas é a aproximação
mais honesta que os dados disponíveis permitem. As fontes estão em
`data/clones_densidade.json`.

Se um clone não tiver densidade cadastrada, o código usa **500 kg/m³** e **avisa
no console**. Isso é intencional: melhor um aviso visível do que mascarar dado
faltante com um número inventado.

**Frase para a banca:**
> "A densidade não vem de sensor nem é calculada a partir de DAP — vem de uma
> referência de literatura para o material genético típico de plantios
> industriais de eucalipto no Brasil, já que o dado específico do clone é
> proprietário do parceiro, como eles próprios confirmaram. Em operação real,
> isso seria substituído pelo laudo de laboratório de cada clone."

**Não digam:** "o e-mail da JD validou 100% nossa abordagem" (exagero — apoia a
*direção*, não confirma o número) nem "é padrão da indústria" (não existe padrão
numérico documentado para isso).

---

## 8. Visão computacional — o item mais crítico do projeto

### O que já foi corrigido
- O dataset tinha **8 classes de defeito** sendo esmagadas numa só
  (`wood_defect`) por um bug — corrigido.
- Confirmado, testando no dataset de origem, que o mapeamento de classe está
  correto e o modelo generaliza bem **dentro do domínio em que foi treinado**.
- O pré-processamento de inferência foi alinhado ao do treino (grayscale +
  CLAHE), eliminando uma perda silenciosa de precisão.

### O problema real, confirmado com teste controlado
O dataset de treino atual (Kaggle,
`nomihsa965/large-scale-image-dataset-of-wood-surface-defects`) é **madeira
serrada europeia** (provavelmente pinheiro/abeto), fotografada em esteira
industrial — **não é eucalipto, não é casca, é outro domínio visual inteiro**:

- No dataset de origem → distribuição plausível, boa confiança.
- Em imagens de tora com casca → o modelo confunde casca normal com `resin`,
  quase sempre.

**Diagnóstico:** não é bug, não é falta de epoch, não é peso inicial errado
(testamos) — é **domain mismatch**. O modelo nunca viu casca de eucalipto
saudável, então não distingue "textura normal" de "defeito".

### O plano
1. **Fine-tuning em duas etapas**, complementando o modelo (não substituindo):
   fotos reais de eucalipto saudável **e** com defeito, depois as 8 classes.
2. **Fonte de fotos:** dataset da UTFPR (tora real de eucalipto brasileiro) +
   fotos próprias compradas de madeireira.
3. **Descoberta importante:** o setor escaneia a **superfície/casca** ao longo do
   comprimento, não a face cortada — as fotos devem priorizar casca.

**Se o fine-tuning não sair a tempo,** a defesa honesta é: "usamos um dataset
público europeu como prova de conceito de arquitetura; identificamos e
documentamos a lacuna de domínio". Isso é defensável — o oposto seria fingir que
está resolvido.

---

## 9. Escopo — "madeira" vs. "eucalipto"

O enunciado diz só "qualidade de madeira". Decisão: **não usar isso como desculpa
formal** (a JD mencionou eucalipto verbalmente numa visita; fingir que não
sabíamos não resiste a pergunta de banca) — mas **usar como vantagem de design
real**: o pipeline é agnóstico de espécie por arquitetura (densidade é tabela
trocável, detecção é generalizável), e o eucalipto é uma **especialização por
fine-tuning**, não uma reformulação do projeto.

---

## 10. Estrutura de arquivos

```
main.py                            # Campo: câmera + YOLO + indicadores + SQLite
calibrar.py                        # Gera o marcador ArUco de escala (e ROI/régua opcionais p/ maquete)
sync_daemon.py                     # SQLite local → PostgreSQL central (idempotente)
config.json                        # Parâmetros de máquina/talhão/câmera/IA
docker-compose.yml                 # PostgreSQL local de teste
requirements.txt                   # Dependências de main.py e sync_daemon.py
data/clones_densidade.json         # Cache da densidade por clone (gerado pelo sync)
data/inventario_johndeere.json     # DAP/idade/clone da planilha real da JD
Banco de dados/schema_sqlite.sql   # Schema local (campo)
Banco de dados/schema_postgres.sql # Schema central
Banco de dados/setup_completo.sql  # Schema + seed, usado pelo docker-compose
Banco de dados/migration_clones_densidade.sql
models/wood_best.pt                # Pesos do YOLO (fora do Git — pedir ao time)
tests/testar_modelo_eucalipto.py   # Roda o modelo numa pasta de imagens
tests/testar_densidade_clones.py   # Confere o lookup de densidade
tests/simular_cenario.py           # Roda o pipeline real (analisar_frame) sem câmera
OmniRoot_Challenge_*.ipynb         # Notebooks de treino (Colab e VSCode)
walkthrough.md                     # ⚠️ desatualizado — revisar antes de usar
```

---

## 11. Pendências, em ordem de prioridade

1. **[Crítico] Fine-tuning de eucalipto** — coletar fotos (UTFPR + compra),
   anotar no Roboflow e retreinar a partir do checkpoint atual. É o maior risco
   do projeto hoje. *(O script `fine_tuning_eucalipto.py` citado em versões
   anteriores deste README **não está neste repositório** — precisa ser
   recuperado ou reescrito.)*
2. **[Importante] Validar o StanForD** — o Export StanForD já existe no
   dashboard, mas é baseado na documentação pública da Skogforsk e **não** foi
   validado contra o XSD oficial. Não declarar como "certificado".
3. **[Limpeza] Atualizar `walkthrough.md`** — tem número de densidade e nome de
   função desatualizados.
4. **[Antes de mandar para fora] Nota metodológica de densidade** — o arquivo
   `.docx` citado antes não está no repositório; se ainda for entregue, precisa
   ser gerado e preenchido com nome da equipe/instituição.

### Já concluído
- ✅ **Dashboard** — existe e está funcionando
  ([omni-root-dashboard](https://github.com/Omni-Root/omni-root-dashboard)), com
  login, painéis e exportações.
- ✅ **Export StanForD** — implementado no dashboard (`.hpr` StanForD 2010 por
  máquina, em ZIP), lendo direto do PostgreSQL.
- ✅ **Tempo real sem congelamento** — `main.py` reestruturado em duas threads.
- ✅ **Pré-processamento alinhado ao treino** (grayscale + CLAHE).
- ✅ **ROI + filtros de inferência** — o modelo só vê a região da tora; o que
  ele detecta fora dela aparece em cinza como "ignorado" (transparente na demo).
- ✅ **Tortuosidade e casca corrigidas** — eixo/flecha e casca residual por Otsu
  (seção 5). Testadas com toras sintéticas de curvatura conhecida.
- ✅ **Inferência em CPU 5x mais rápida** — OpenVINO INT8 a 640 px
  (`exportar_modelo.py`) e visão clássica desacoplada do modelo em thread
  própria: contorno/casca/diâmetro/rachadura atualizam a ~10 fps, o YOLO no
  ritmo da CPU.
- ✅ **Segmentação por cor** (madeira x fundo) com detecção de vista
  seção/lateral — resolve o contorno seguindo sombra/mesa das primeiras demos.
- ✅ **Escala automática por marcador ArUco** (sem calibração manual) e
  **`--fonte` vídeo/pasta** como plano B da apresentação.

---

## 12. Decisões já tomadas

Não precisa reabrir a discussão, a não ser que surja informação nova:

- Windows na demo, não Raspberry.
- Sem sensor de força/ADC — densidade é lookup, distância câmera-tora é constante
  calibrada.
- Densidade por literatura documentada (485–490 kg/m³), não por clone específico.
- CV: o problema é **dado de domínio**, não hiperparâmetro nem peso inicial —
  confirmado com teste controlado.
- Sincronização em Python (`sync_daemon.py`), não Go — uma linguagem a menos.
- Dashboard em TypeScript, repositório separado, **somente leitura** sobre o
  PostgreSQL central.
- Pipeline "species-agnostic" por design — eucalipto é especialização.

---

*Dúvidas sobre alguma decisão aqui? Pergunte antes de mudar algo que já foi
validado com mais contexto do que cabe num README — mas, claro, se surgir
informação nova, tudo é revisável.*
