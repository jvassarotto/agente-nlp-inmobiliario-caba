# Evaluacion — Clasificacion multilabel

- **Modelo:** `models/cls-beto`
- **Conjunto evaluado:** `data/annotated/real_cls.jsonl` (real)
- **Ejemplos:** 105

## Reporte por clase

```
               precision    recall  f1-score   support

DUENO_DIRECTO     0.5200    0.2281    0.3171        57
  OPORTUNIDAD     0.6552    0.6129    0.6333        31
     URGENCIA     0.1111    0.0303    0.0476        33
    REFACCION     1.0000    0.0952    0.1739        21

    micro avg     0.5385    0.2465    0.3382       142
    macro avg     0.5716    0.2416    0.2930       142
 weighted avg     0.5255    0.2465    0.3023       142
  samples avg     0.2492    0.1302    0.1648       142
```

## Resumen

```json
{
  "f1_macro": 0.2929846487905873,
  "f1_micro": 0.33816425120772947,
  "n_ejemplos": 105
}
```
