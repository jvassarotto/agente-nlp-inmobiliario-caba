# Evaluacion — Clasificacion multilabel

- **Modelo:** `models/cls-beto`
- **Conjunto evaluado:** `data/annotated/cls_test.jsonl` (sintetico)
- **Ejemplos:** 120

## Reporte por clase

```
               precision    recall  f1-score   support

DUENO_DIRECTO     1.0000    1.0000    1.0000        39
  OPORTUNIDAD     1.0000    1.0000    1.0000        41
     URGENCIA     1.0000    1.0000    1.0000        25
    REFACCION     1.0000    1.0000    1.0000        30

    micro avg     1.0000    1.0000    1.0000       135
    macro avg     1.0000    1.0000    1.0000       135
 weighted avg     1.0000    1.0000    1.0000       135
  samples avg     0.6917    0.6917    0.6917       135
```

## Resumen

```json
{
  "f1_macro": 1.0,
  "f1_micro": 1.0,
  "n_ejemplos": 120
}
```
