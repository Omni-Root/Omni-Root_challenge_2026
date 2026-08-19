# Omni-Root — Challenge 2026 (John Deere / Suzano)

**Desafio recebido:** "Qualidade da madeira" (enunciado curto, sem detalhar espécie).

**Nossa proposta:** sistema de inspeção de tora por visão computacional, acoplado à máquina de colheita, funcionando offline em campo e sincronizando dados quando houver internet.

Este README existe pra qualquer pessoa do time conseguir se atualizar rápido sem precisar reler tudo do zero. Se você sumiu um tempo do projeto, comece por aqui.

---

## 1. Ideia em uma frase

Câmera na máquina → visão computacional avalia a tora em tempo real (4 indicadores) → grava local (SQLite, funciona sem internet) → sincroniza pra um banco central (PostgreSQL) quando há rede → dashboard mostra os dados.

---

## 2. Arquitetura

```
┌─────────────────────────┐         ┌──────────────────────┐
│   MÁQUINA DE CAMPO       │  sync   │   CENTRAL (escritório │
│  (Windows — não precisa  │ ──────► │   / cloud, com internet)│
│   mais de Raspberry)     │ quando  │                       │
│                          │  há     │  PostgreSQL           │
│  main.py                 │ internet│  Dashboard (a definir)│
│  ├─ Câmera (OpenCV)      │         │                       │
│  ├─ YOLO (detecção)      │         └──────────────────────┘
│  └─ SQLite local         │
└─────────────────────────┘
```

**Decisão importante:** só a câmera é hardware obrigatório. Sensor de força e ADC (que existiam numa versão anterior, pra "medir" densidade) foram **removidos** — densidade não vem de sensor, vem de uma tabela de referência (ver seção 4).

**Por que Windows e não Raspberry Pi:** as máquinas reais da John Deere rodam Windows Embedded. Rodar a maquete em Windows também é mais fiel ao ambiente real do que simular em Linux/ARM — e como o código já não depende de sensor físico específico, não tem motivo pra manter o Raspberry na demo.

---

## 3. Os 4 indicadores de qualidade

| Indicador | Como é medido | Status |
|---|---|---|
| Diâmetro / Altura | Câmera + conversão pixel→cm (GSD), usa distância do sensor ultrassônico | ✅ funcionando |
| Tortuosidade | OpenCV, contorno da tora | ✅ funcionando |
| Apodrecimento / pragas (defeito) | YOLO (nó, rachadura, etc.) | ⚠️ ver seção 5 — problema de domínio identificado, em correção |
| Densidade | **Lookup por clone/material genético** (não sensor, não fórmula) | ✅ funcionando, ver seção 4 |

Também calculamos **volume útil** (geometria) e **massa seca estimada** (volume × densidade) como indicadores derivados.

---

## 4. Densidade — por que é assim, e o que dizer se perguntarem

Densidade básica da madeira **não dá pra medir com câmera**. Tentamos dois caminhos que **não funcionaram** e foram descartados de propósito — importante saber disso pra não repetir o erro:

- ❌ Fórmula inventada a partir de DAP/idade (`440 + DAP×3.8 + idade×4.5`) — não tem base científica, contraria o que o próprio contato da JD disse por e-mail (densidade é definida por material genético, não por medida de campo).
- ❌ Valor "mockado" fixo por clone, sem fonte — pareceria dado real sem ser.

**O que fizemos:** todos os clones do inventário usam hoje **490 kg/m³**, valor de literatura real e citável para o híbrido comercial *Eucalyptus grandis x E. urophylla* (o material genético mais comum em plantios de celulose no Brasil) — não é o valor exato de nenhum clone específico (isso é dado proprietário da Suzano, confirmado por e-mail), mas é a aproximação mais honesta que os dados disponíveis permitem. As fontes estão documentadas em `data/clones_densidade.json` (campo `_leiame` e `fontes`) e na nota metodológica (ver `nota_metodologica_densidade.docx`, gerada à parte).

**Frase pra usar se perguntarem na banca:**
> "A densidade não vem de sensor nem é calculada a partir de DAP — vem de uma referência de literatura para o material genético típico de plantios industriais de eucalipto no Brasil, já que o dado específico do clone é proprietário do parceiro, como eles próprios confirmaram. Em operação real, isso seria substituído pelo laudo de laboratório de cada clone."

**Não digam:** "o e-mail da JD validou 100% nossa abordagem" (exagero — o e-mail apoia a *direção*, não confirma o número) nem "é padrão da indústria" (não existe padrão numérico documentado pra isso).

---

## 5. Visão computacional — situação atual (o item mais crítico do projeto)

### O que já foi corrigido
- O dataset de treino tinha **8 classes de defeito** (`Quartzity`, `Live_Knot`, `Marrow`, `resin`, `Dead_Knot`, `knot_with_crack`, `Knot_missing`, `Crack`) sendo **esmagadas numa classe só** (`wood_defect`) por um bug de código — corrigido.
- Confirmamos, testando no próprio dataset de origem (não no eucalipto), que o **mapeamento de classe está correto** e o modelo generaliza bem *dentro do domínio em que foi treinado*.

### O problema real, confirmado com teste controlado
O dataset de treino atual (Kaggle, `nomihsa965/large-scale-image-dataset-of-wood-surface-defects`) é **madeira serrada europeia** (provavelmente pinheiro/abeto), fotografada em esteira industrial — **não é eucalipto, não é casca, é outro domínio visual inteiro**. Testado nas mesmas condições:
- No dataset de origem → distribuição de classes plausível, boa confiança.
- Em imagens de tora com casca (Roboflow, genérico) → o modelo confunde casca normal com a classe `resin`, quase sempre.

**Diagnóstico:** não é bug, não é falta de epoch, não é peso inicial errado (testamos essa hipótese também) — é **domain mismatch** real. O modelo nunca viu casca de eucalipto saudável, então não sabe distinguir "isso é textura normal" de "isso é defeito".

### O plano em andamento
1. **Fine-tuning em duas etapas**, não substituindo o modelo, complementando:
   - Etapa 1: fotos reais de eucalipto (saudável **e** com defeito) — ensina o domínio visual certo.
   - Etapa 2 (já feita): 8 classes de defeito reais.
2. **Fonte de fotos:** dataset da UTFPR (link no Google Drive, tora real de eucalipto brasileiro — ver referência completa abaixo) + fotos próprias do time (compradas de madeireira/lenheiro, não precisa sair caçando árvore).
3. **Descoberta importante:** o setor de detecção de defeito escaneia a **superfície/casca** da tora ao longo do comprimento, não a face cortada — então fotos devem priorizar casca, não a ponta circular da tora.
4. **Script pronto:** `fine_tuning_eucalipto.py` organiza fotos anotadas (Roboflow) em treino/validação e roda o fine-tuning a partir do checkpoint atual, sem precisar treinar do zero.

**Se esse fine-tuning não sair a tempo:** a defesa honesta é "usamos um dataset público europeu como prova de conceito de arquitetura; identificamos e documentamos a lacuna de domínio; o pipeline de fine-tuning para eucalipto já está pronto (`fine_tuning_eucalipto.py`), faltando só o dado de treino real". Isso é uma resposta defensável — o oposto seria fingir que já está resolvido.

---

## 6. Escopo do desafio — "madeira" vs. "eucalipto"

O enunciado diz só "qualidade de madeira", não especifica espécie. Decisão: **não usar isso como desculpa formal** (a JD já mencionou eucalipto verbalmente numa visita, então fingir que não sabíamos não resiste a pergunta de banca) — mas **usar como vantagem de design real**: o pipeline é agnóstico de espécie por arquitetura (densidade é uma tabela trocável por clone, detecção é generalizável), e a especialização em eucalipto é uma etapa de fine-tuning já mapeada e com script pronto, não uma reformulação do projeto.

---

## 7. Estrutura de arquivos

```
main.py                          # Roda no campo: câmera + YOLO + grava SQLite
sync.go                          # Sincroniza SQLite → PostgreSQL quando há internet
config.json                      # Parâmetros da máquina/talhão/câmera
data/clones_densidade.json       # Tabela de referência de densidade (com fontes documentadas)
data/inventario_johndeere.json   # Dados reais de DAP/idade/clone (da planilha da JD)
Banco de dados/schema_*.sql      # Schemas SQLite e PostgreSQL
tests/testar_modelo_eucalipto.py # Testa o modelo YOLO numa pasta de imagens (sem precisar de webcam)
fine_tuning_eucalipto.py         # Organiza fotos anotadas + roda fine-tuning
stanford_export.py               # Exporta dados no formato StanForD (padrão real da indústria) — pronto, não conectado a UI ainda
simular_cenario.py               # Simula cenário de inspeção sem hardware real
OmniRoot_Challenge_*.ipynb       # Notebooks de treino do modelo (Colab e VSCode local)
walkthrough.md                   # ⚠️ desatualizado — tem número antigo de densidade (510) e nome de função antigo, ignorar até revisar
```

---

## 8. Pendências / próximos passos, em ordem de prioridade

1. **[Crítico] Fine-tuning de eucalipto** — coletar fotos (UTFPR + compra de madeira), anotar no Roboflow, rodar `fine_tuning_eucalipto.py`. É o maior risco do projeto hoje.
2. **[Importante] Dashboard** — ainda não existe. Decisão em aberto: provavelmente algo simples (Streamlit ou Next.js + API fina), rodando perto do PostgreSQL central, não na máquina de campo. Ver decisões de arquitetura na íntegra da conversa se precisar retomar essa discussão.
3. **[Bônus] Conectar `stanford_export.py` ao dashboard** — já funciona isolado, só falta um botão que chame ele.
4. **[Limpeza] Atualizar `walkthrough.md`** — tem número de densidade e nome de função desatualizados.
5. **[Antes de mandar pra fora] Preencher a nota metodológica de densidade** com nome da equipe/instituição (hoje tem placeholder).

---

## 9. Decisões já tomadas (não precisa reabrir discussão, a não ser que surja informação nova)

- Windows na demo, não Raspberry.
- Sem sensor de força/ADC — densidade é lookup, não medição.
- Densidade = 490 kg/m³ uniforme hoje, com fonte de literatura documentada, não por clone específico.
- CV: problema é dado de domínio, não hiperparâmetro nem peso inicial — confirmado com teste controlado.
- TypeScript é opção viável pro backend/dashboard, se o time tiver mais afinidade que com Python/Java/C# pra essa parte.
- Pipeline é "species-agnostic" por design — eucalipto é especialização, não reformulação.

---

*Dúvidas sobre qualquer decisão aqui? Bora perguntar antes de mudar alguma coisa que já foi validada com mais contexto do que cabe num README — mas claro, se surgir informação nova, tudo é revisável.*
