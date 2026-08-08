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

### El resultado principal: la brecha sintético → real

| Modelo | Test **sintético** | Conjunto **real** |
|---|---|---|
| **NER** — F1 micro (seqeval) | **0.997** | **0.222** |
| **Clasificación** — F1 macro | **0.971** | **0.176** |

**Este contraste es el hallazgo central del trabajo, y no se disimula.** El F1 casi perfecto sobre
datos sintéticos no mide qué tan bueno es el modelo: mide qué tan fácil es el test. Los avisos
sintéticos salen de plantillas, así que el modelo aprende la plantilla en lugar del concepto.

Los dos modelos fallan de manera **distinta**, y eso es informativo:

- **El NER sobre-etiqueta.** Aprendió "sustantivo después de *Cuenta con*" en vez del vocabulario
  de amenities. Sobre texto real marca cosas como `AMENITY → ventilación` o `ORIENTACION → Scalabrini`
  (un nombre de calle).
- **El clasificador casi no dispara.** Precisión macro 0.55 pero recall 0.16: cuando predice, suele
  acertar, pero se pierde la mayoría. Los avisos reales expresan «dueño directo» o «urgencia» con
  formas que el generador nunca produjo.

**Cómo leer estos números.** Las etiquetas del conjunto real vienen del pre-anotador LLM
(`llama3.2:3b`) **sin revisión humana completa**, así que miden concordancia con un anotador
imperfecto, no con verdad de referencia. Además son sólo 65 avisos. La **magnitud de la caída** es
sólida; los valores exactos, no.

La conclusión práctica es el trabajo futuro más urgente: **anotar un conjunto real de tamaño
razonable**. Es lo que separa este pipeline funcionando de un modelo utilizable.

### Reportes

Todo está en [`reports/`](reports/):

| Archivo | Contenido |
|---|---|
| `ner_sintetico.md` / `ner_real.md` | F1 por entidad (seqeval), sintético y real |
| `cls_sintetico.md` / `cls_real.md` | Precisión / recall / F1 por clase y macro |
| `parser_robustness.json` | Retención de campos del parser ante cambios de layout |
| `agent_metrics_*.json` | Tasa de éxito de extracción del agente, por fuente |

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
python -m src.agent.run_scrape --mode grid --max-listings 2500           # RECOMENDADO (ver abajo)
python -m src.agent.run_scrape --mode deterministic --max-listings 300   # paginacion clasica
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

Los tres modos comparten las mismas *tools* y el mismo *parser*.

### El anti-bot: qué se probó y qué se aprendió

Este fue el hallazgo más concreto sobre la viabilidad de la "adquisición autónoma" que plantea la
propuesta. Se probaron tres estrategias, en este orden:

| Estrategia | Resultado |
|---|---|
| **1. Visitar la página de detalle de cada aviso** | Bloqueado. Devuelve un *challenge* de Cloudflare ("Un momento…") en lugar del contenido |
| **2. Extraer desde las tarjetas del listado** | **Funciona.** Las tarjetas ya traen la descripción completa (1.500-3.500 caracteres). Además una request rinde 25-30 avisos en vez de uno |
| **3. Recorrer búsquedas distintas (barrio × ambientes) en lugar de paginar** | Bloqueado, pero **no por lo que parecía** |

Sobre el punto 3, el detalle importa. Primero se creyó que el bloqueo era de las **URLs paginadas**
(`-pagina-2.html`), y que usar búsquedas distintas lo evitaría. La evidencia lo desmintió:

- `/departamentos-venta-palermo-2-ambientes.html` devolvió **30 avisos** en una prueba aislada.
- La **misma URL**, dentro de la corrida de la grilla, devolvió **0**.

Es decir: **el bloqueo es por sesión, no por URL**. Cloudflare detecta el navegador automatizado y
lo desafía a partir de la segunda request, sin importar a qué página apunte. Subir los delays a
20-40 segundos tampoco cambió nada, lo que confirma que no es un límite de tasa sino de detección.

**Dónde se puso el límite.** Seguir habría requerido renovar sesión o *fingerprint* en cada request
—proxies rotativos, plugins *stealth*, resolución de CAPTCHA—, y eso ya no es consultar el sitio de
otra manera sino **evadir un control de seguridad**. Contradice el uso académico que declara este
trabajo, así que no se hizo.

**Consecuencia para el dataset:** en la práctica se obtiene una tanda de ~25-30 avisos reales por
sesión de navegador. Muy lejos de los 8.000-15.000 de la propuesta.

### La solución: cambiar de fuente, no de método

Ante ese límite había dos caminos: **evadir la protección** (proxies rotativos, *fingerprints*
falsos, resolución de CAPTCHA) o **buscar una fuente que permita el acceso**. Se tomó el segundo.

**Argenprop habilita explícitamente el acceso automatizado en su `robots.txt`:**

```
Allow: /*?pagina-1$
...
Allow: /*?pagina-10$
Disallow: /*?pagina-
```

Las páginas 1 a 10 de cada búsqueda están permitidas; de la 11 en adelante, no. **Ese tope se
valida en el código**, no sólo en la documentación (`src/agent/argenprop.py → MAX_PAGINA`).

Ventajas adicionales de la fuente:

| | ZonaProp | Argenprop |
|---|---|---|
| Necesita navegador | Sí (Playwright) | No, HTTP directo |
| Avisos por request | 25-30 | 20 |
| Requests por sesión | **1** (después bloquea) | Sin bloqueo |
| Antigüedad en la tarjeta | No | **Sí** |
| Acceso automatizado | Bloqueado | **Permitido por robots.txt** |

La arquitectura del agente no cambió: siguen siendo las mismas *tools* y el mismo patrón de
parseo resiliente; lo único que cambió es la puerta por la que entra.

```bash
python -m src.agent.run_scrape --mode argenprop --max-listings 1000
```

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

- **Anti-bot: el límite real del scrape.** ZonaProp protege el sitio con Cloudflare. En la práctica:
  - Las **páginas de detalle** de cada aviso devuelven un *challenge* ("Un momento…") en lugar del
    contenido. Por eso la extracción se hace **desde las tarjetas del listado**, que ya traen la
    descripción completa. Además de esquivar el bloqueo, es más cortés: una request rinde 25-30
    avisos en vez de uno.
  - Las **URLs paginadas** (`-pagina-2.html` en adelante) también quedan bloqueadas, incluso
    subiendo los delays a 20-40 segundos. No se insistió contra el bloqueo.
  - Resultado: el volumen quedó muy por debajo de los 8.000-15.000 avisos de la propuesta. El
    objetivo fue demostrar que la arquitectura funciona, no maximizar el dataset.
- **Entrenamiento sobre sintético.** Los modelos se entrenan con el generador y el conjunto real
  anotado se usa como **evaluación externa**, midiendo generalización sintético → real.
  Ver la sección "Generalización sintético → real" del notebook: el F1 perfecto sobre datos
  sintéticos **no** se traslada a texto real, y ese es un resultado del trabajo, no un accidente.
- **Anotación semiautomática.** El pre-anotador es un LLM chico (`llama3.2:3b`, por la restricción
  de 4 GB de VRAM); sus errores se propagan, y por eso la revisión manual del subconjunto.
- **Longitud del texto: resuelto con ventanas deslizantes.** BETO tiene un límite
  **arquitectónico** de 512 sub-tokens — son las posiciones que aprendió al pre-entrenarse, no un
  parámetro ajustable. Los avisos reales llegan a 1.173 sub-tokens, así que ni con el máximo se
  leerían enteros (entrarían completos 86 de 105). La solución es partir el aviso en ventanas que
  sí entren y unir las predicciones (`src/models/chunking.py`), con dos cuidados:
  las ventanas **se solapan**, para no cortar una entidad al medio; y al unir **gana la ventana
  donde la palabra está más al centro**, porque ahí tiene contexto de los dos lados.
  Para clasificación se toma el **máximo** por clase entre ventanas: si la señal aparece en alguna
  parte del aviso, el aviso la tiene (promediar la diluiría en los avisos largos).
- **Términos de uso.** Scraping con fines **académicos**, a ritmo razonable y sin redistribuir el
  contenido del portal: el repositorio incluye una muestra acotada de datos ya estructurados, no
  volcados de páginas.

---

## 9. Troubleshooting

**`CUDA out of memory`** (GTX 1650, 4 GB) — en orden de probabilidad:

1. **Descargar el modelo de Ollama.** Es la causa más común y la menos evidente: Ollama deja su
   modelo residente en VRAM varios minutos después de usarlo, ocupando ~2,5 GB de los 4 GB. No
   alcanza con cerrar la terminal.
   ```bash
   ollama stop llama3.2:3b-instruct-q4_K_M
   nvidia-smi   # confirmar que bajó a ~500 MiB antes de entrenar
   ```
2. Bajar `batch_size` a `2` y subir `grad_accum` a `8` (mantiene el batch efectivo).
3. Bajar `max_length`. Con *chunking* esto **no** pierde texto: sólo achica la ventana, y el aviso
   se sigue leyendo entero en más pasadas.
4. Confirmar `fp16: true`.
5. Si el error menciona fragmentación: `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

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
