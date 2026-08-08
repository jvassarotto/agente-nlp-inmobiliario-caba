# Agente LLM + NLP para avisos inmobiliarios (ZonaProp / CABA)

**Trabajo Final — Deep Learning · Maestría en Management & Analytics (MMA), ITBA**
Alumno: Joaquín Héctor Vassarotto — Legajo 106442

---

## Qué es esto

Una arquitectura de **dos capas** que convierte avisos inmobiliarios en variables estructuradas:

1. **Capa de agente (orquestación).** Un agente **ReAct / tool-use** con un LLM **local y gratuito**
   (vía Ollama) navega ZonaProp, decide cómo paginar y qué extraer, y se recupera ante errores.
   Usa Playwright para el contenido dinámico y un *parser resiliente* de dos niveles.
2. **Capa de enriquecimiento (NLP — el núcleo entrenable).** Dos *transformers* **BETO**
   (BERT-base-spanish) *fine-tuneados*:
   - **NER** (token classification): `AMENITY`, `ESTADO`, `ANTIGUEDAD`, `ORIENTACION`, `EXPENSAS`.
   - **Clasificación multilabel**: señales del vendedor `DUENO_DIRECTO`, `OPORTUNIDAD`, `URGENCIA`, `REFACCION`.

> El LLM del agente se usa **pre-entrenado**: es sólo orquestación. Los modelos que se **entrenan y
> evalúan** como entregable de la materia son los dos *transformers* de la capa 2.

**El valor no es obtener datos, sino generar variables latentes a partir del lenguaje natural** que
en una etapa posterior puedan enriquecer un modelo de valuación más allá de los atributos tabulares.

---

## 1. Reproducir el trabajo en ~10 minutos

Este camino **no requiere GPU, ni Ollama, ni scrapear nada**. Es el recomendado para corregir.

```bash
git clone https://github.com/<usuario>/zonaprop-agent-nlp.git
cd zonaprop-agent-nlp

python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

scripts\run_demo.bat --quick      # Windows   (Linux/Mac: bash scripts/run_demo.sh --quick)
```

`--quick` entrena 1 epoch sobre 200 ejemplos para verificar que **todo el pipeline corre**.
Sin ese flag, corre el entrenamiento completo según `configs/config.yaml` (varios minutos con GPU,
bastante más en CPU).

El script encadena: generar dataset sintético → armar *splits* → *fine-tuning* NER → *fine-tuning*
clasificación → evaluación → medición de robustez del parser.

Para ver el resultado sobre una descripción cualquiera:

```bash
python -m src.models.infer --text "Depto a estrenar con pileta y cochera. Dueno directo."
```

### Notebook con el recorrido completo

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) recorre el proyecto entero con los resultados
inline: problema, agente, métricas, esquema de etiquetas, dataset, entrenamiento, justificación de
las métricas e inferencia final.

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## 2. Acceso a los datos

Todos los datos necesarios para reproducir el trabajo **están incluidos en este repositorio**.

| Conjunto | Ubicación | Cómo se obtiene | En el repo |
|---|---|---|---|
| **Sintético** (entrenamiento) | `data/synthetic/listings.jsonl` | `python -m src.data.generate_synthetic` — determinístico con `seed: 42` | Sí |
| **Splits** train/val/test | `data/annotated/{ner,cls}_{train,val,test}.jsonl` | `python -m src.annotation.prepare_dataset` | Sí |
| **Fixtures HTML** | `data/fixtures/` | Se guardan solos durante el scrape real | Sí, si se corrió el scrape |
| **Avisos reales** | `data/raw/zonaprop_caba.jsonl` | `python -m src.agent.run_scrape` (requiere Playwright) | Muestra acotada |
| **Modelos entrenados** | `models/ner-beto/`, `models/cls-beto/` | Se generan al entrenar | **No** — ver nota |
| **Reportes de métricas** | `reports/` | Los generan el entrenamiento y la evaluación | Sí |

> **Por qué los modelos no están en el repo.** Cada checkpoint de BETO pesa ~440 MB y GitHub
> rechaza archivos de más de 100 MB. Se versionan los **reportes de métricas** (`reports/`), que es
> lo que permite verificar los resultados; los pesos se regeneran con `scripts\run_demo.bat`.

> **Sobre los datos reales.** Se incluye una **muestra acotada** con fines de reproducibilidad
> académica. No se redistribuye el contenido completo del portal (ver sección 7).

---

## 3. Resultados

Los reportes con las métricas están en [`reports/`](reports/):

| Archivo | Contenido |
|---|---|
| `ner_sintetico.md` | F1 por entidad (seqeval) sobre el test sintético |
| `cls_sintetico.md` | Precisión / recall / F1 por clase y macro |
| `parser_robustness.json` | Retención de campos del parser ante cambios de layout |
| `agent_metrics_*.json` | Tasa de éxito de extracción del agente |

### Métricas: definición y justificación

**NER — F1 por entidad con `seqeval`, no accuracy por token.** Por dos motivos: (a) la enorme
mayoría de los tokens son `O`, así que un modelo que no detecte nada igual sacaría un accuracy
altísimo; (b) lo que importa es el **span completo** — si la anotación dice `a estrenar` y el modelo
marca sólo `estrenar`, para construir la variable *estado de la propiedad* eso está mal. `seqeval`
exige que coincidan el tipo y los límites exactos.

**Clasificación — F1 macro además del micro.** Las clases están desbalanceadas. El micro queda
dominado por las frecuentes; el **macro promedia las cuatro con igual peso**, y por eso penaliza que
el modelo ignore una clase rara. Justamente las señales raras (`URGENCIA`, `OPORTUNIDAD`) son las
más relevantes para detectar subvaluación. Se reporta también el F1 **por clase**, porque el
promedio solo esconde dónde falla.

**Agente — tasa de éxito de extracción y robustez.** La primera es *avisos extraídos con descripción
utilizable / avisos detectados*. La segunda mide qué fracción de los campos sobrevive cuando se
degrada el HTML imitando cambios reales del portal (`python -m src.agent.robustness`).

---

## 4. Instalación completa (para correr el agente y el scrape)

### 4.1 GPU (opcional, acelera el entrenamiento)

`requirements.txt` instala la build de CPU. Si tenés GPU NVIDIA, instalá torch con CUDA **antes**:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Esperado: True NVIDIA GeForce GTX 1650
```

Si `cuda.is_available()` da `False`, entrena en CPU (más lento pero funcional) y `fp16` se desactiva solo.

### 4.2 Navegador (sólo para el scrape)

```bash
playwright install chromium
```

### 4.3 LLM local gratuito (sólo para el agente y la pre-anotación)

1. Instalar Ollama desde <https://ollama.com>.
2. Descargar el modelo:

```bash
ollama pull llama3.2:3b-instruct-q4_K_M     # entra 100% en 4 GB de VRAM (recomendado)
# alternativa con mejor extraccion, mas lenta:
ollama pull qwen2.5:7b-instruct-q4_K_M
```

3. Dejar Ollama corriendo. El modelo se elige en `configs/config.yaml → agent.model`.

> **Gratis y sin API keys.** Todo el LLM corre local vía Ollama; no se usa ni se paga ninguna API.

---

## 5. Pipeline completo, etapa por etapa

Desde la raíz del repo, con el entorno activado. Los parámetros salen de `configs/config.yaml`.

```bash
# 1. Construir el dataset real con el agente
python -m src.agent.run_scrape --mode deterministic --max-listings 300   # robusto, para corridas largas
python -m src.agent.run_scrape --mode agent --max-pages 3                # patron ReAct: el LLM orquesta

# 2. Anotacion semiautomatica (el LLM pre-anota; despues se revisa a mano)
python -m src.annotation.preannotate --input data/raw/zonaprop_caba.jsonl --limit 300

# 3. Armar los splits
python -m src.annotation.prepare_dataset --input data/annotated/preannotated.jsonl
python -m src.annotation.prepare_dataset --input data/synthetic/listings.jsonl   # o desde el sintetico

# 4. Fine-tuning (nucleo Deep Learning)
python -m src.models.train_ner            # -> models/ner-beto
python -m src.models.train_classifier     # -> models/cls-beto

# 5. Evaluacion
python -m src.models.evaluate --task ner --model_dir models/ner-beto
python -m src.models.evaluate --task cls --model_dir models/cls-beto
#    ...y sobre el set real, para medir generalizacion sintetico -> real:
python -m src.models.evaluate --task ner --model_dir models/ner-beto \
    --input data/annotated/real_ner.jsonl --tag real

# 6. Enriquecimiento
python -m src.models.infer --text "Depto a estrenar con pileta y cochera. Dueno directo."
```

Los dos modos de scrape comparten las mismas *tools* y el mismo *parser*: el **modo agente**
demuestra el patrón ReAct de la propuesta, el **determinístico** es más robusto para corridas largas.

---

## 6. Estructura del proyecto

```
zonaprop-agent-nlp/
├── configs/config.yaml           # UNICO lugar para hiperparametros y rutas
├── notebooks/demo.ipynb          # recorrido completo con resultados inline
├── docs/guia_github.md           # guia para publicar el repo
├── reports/                      # metricas generadas (versionadas)
├── scripts/run_demo.bat / .sh    # pipeline demo end-to-end (--quick disponible)
├── src/
│   ├── agent/                    # CAPA 1 — agente
│   │   ├── browser_tools.py      #   tools Playwright (navegar / extraer / paginar)
│   │   ├── parser.py             #   HTML -> campos estructurados + descripcion
│   │   ├── metrics.py            #   tasa de exito de extraccion + fixtures
│   │   ├── robustness.py         #   robustez del parser ante cambios de layout
│   │   ├── react_agent.py        #   agente ReAct (LangGraph + ChatOllama)
│   │   └── run_scrape.py         #   entrypoint (modo agente / deterministico)
│   ├── annotation/               # anotacion semiautomatica
│   │   ├── label_schema.py       #   entidades NER + clases de senal
│   │   ├── llm_client.py         #   cliente Ollama (JSON)
│   │   ├── preannotate.py        #   el LLM pre-anota (BIO + clases)
│   │   └── prepare_dataset.py    #   splits train/val/test
│   ├── data/
│   │   ├── schema.py             #   modelo Pydantic del aviso
│   │   └── generate_synthetic.py #   dataset sintetico con etiquetas gold
│   ├── models/                   # CAPA 2 — NLP entrenable
│   │   ├── train_ner.py          #   fine-tuning BETO (token classification)
│   │   ├── train_classifier.py   #   fine-tuning BETO (multilabel)
│   │   ├── evaluate.py           #   F1 por entidad/clase + reportes
│   │   └── infer.py              #   texto libre -> atributos estructurados
│   └── utils/                    # config, IO (jsonl), tokenizacion/BIO
├── tests/test_pipeline.py        # 7 tests de la logica propia, sin GPU
├── data/                         # raw / synthetic / annotated / fixtures
└── models/                       # checkpoints (no versionados)
```

---

## 7. El dataset sintético: por qué existe

ZonaProp tiene protección anti-bot fuerte y cambia de *layout* seguido, así que una corrida real de
scraping lleva horas y puede bloquearse. Para poder **desarrollar, testear y evaluar todo el
pipeline de forma reproducible sin depender del scraping**, el proyecto incluye un generador de
avisos sintéticos realistas de CABA (`src/data/generate_synthetic.py`).

Como se controla la generación, cada aviso trae **etiquetas *gold* por construcción**, lo que
permite medir F1 real de punta a punta. Usa **el mismo tokenizador** que el camino real, así que los
dos son intercambiables.

---

## 8. Limitaciones (documentadas explícitamente)

- **Volumen del scrape.** La propuesta apuntaba a 8.000–15.000 avisos; se trabajó con un volumen
  mucho menor. El objetivo fue demostrar que la arquitectura funciona, no maximizar el dataset.
- **Entrenamiento sobre sintético.** Los modelos se entrenan con el generador y el conjunto real
  anotado se usa como **evaluación externa**, midiendo generalización sintético → real.
- **Anotación semiautomática.** El pre-anotador es un LLM chico (`llama3.2:3b`, por la restricción
  de 4 GB de VRAM); sus errores se propagan, y por eso la revisión manual del subconjunto.
- **Anti-bot.** ZonaProp emplea DataDome/Cloudflare. Se maneja con navegador real, `user-agent` y
  `locale` de escritorio, y **rate-limiting cortés** (`min_delay_s`/`max_delay_s`). Aun así la tasa
  de éxito varía, y los selectores del parser pueden requerir ajuste (por eso los *fallbacks*).
- **Términos de uso.** Scraping con fines **académicos**, a ritmo razonable y sin redistribuir el
  contenido del portal.

---

## 9. Troubleshooting

**`CUDA out of memory`** (GTX 1650, 4 GB) — en `configs/config.yaml`:
1. Bajar `batch_size` a `4` y subir `grad_accum` a `4` (mantiene el batch efectivo).
2. Bajar `max_length` (NER `160`, clasificación `128`).
3. Confirmar `fp16: true`.
4. Cerrar Ollama mientras entrenás: compite por la misma VRAM.

**Disco lleno durante el entrenamiento.** Cada checkpoint incluye el estado del optimizador
(~1,3 GB). Por eso `save_total_limit=1`. Si aun así falta espacio, bajá `epochs` o cambiá `out_dir`
a otro disco.

**El agente alucina la navegación.** Usá el modo determinístico para el scrape y reservá el LLM
para la pre-anotación.

---

## 10. Verificación del código

```bash
python tests/test_pipeline.py
# ==== TODOS LOS TESTS PASARON ====
```

7 tests, **sin GPU ni internet**: alineación de etiquetas subword→palabra, métricas NER y multilabel,
*matching* BIO, agrupación de entidades, parser HTML, consistencia de splits y robustez del parser.

---

## 11. Mapeo propuesta → entregable

| Ítem de la propuesta | Dónde está |
|---|---|
| Agente ReAct / tool-use | `src/agent/react_agent.py`, `browser_tools.py`, `run_scrape.py` |
| Parser HTML/texto resiliente | `src/agent/parser.py` |
| Dataset primario | `data/raw/` (esquema `src/data/schema.py`) |
| Conjunto de anotación (LLM + revisión) | `src/annotation/preannotate.py` |
| Modelo NER (BETO fine-tuning) | `src/models/train_ner.py` |
| Modelo de clasificación (BETO fine-tuning) | `src/models/train_classifier.py` |
| Evaluación (F1 por entidad/clase, macro) | `src/models/evaluate.py`, `reports/` |
| **Agente: tasa de éxito y robustez** | `src/agent/metrics.py`, `src/agent/robustness.py` |
| Enriquecimiento (texto → variables) | `src/models/infer.py` |

---

## 12. Trabajo futuro

Los atributos generados por esta capa de NLP están pensados para integrarse, en el marco de la
**tesis de la maestría**, como *features* de un modelo hedónico de valuación para detectar activos
subvaluados en CABA. Esa integración **excede el alcance de este Trabajo Final** y se menciona sólo
para dar cuenta de la continuidad del proyecto.
