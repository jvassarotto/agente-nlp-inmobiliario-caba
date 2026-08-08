# Evaluacion — Clasificacion multilabel

- **Modelo:** `models/cls-beto`
- **Conjunto evaluado:** `data/annotated/cls_test.jsonl` (sintetico)
- **Ejemplos:** 120

## Reporte por clase

```
               precision    recall  f1-score   support

DUENO_DIRECTO     0.9744    0.9744    0.9744        39
  OPORTUNIDAD     0.9744    0.9268    0.9500        41
     URGENCIA     0.9600    0.9600    0.9600        25
    REFACCION     1.0000    1.0000    1.0000        30

    micro avg     0.9774    0.9630    0.9701       135
    macro avg     0.9772    0.9653    0.9711       135
 weighted avg     0.9774    0.9630    0.9700       135
  samples avg     0.6521    0.6542    0.6516       135
```

## Resumen

```json
{
  "f1_macro": 0.9710897435897435,
  "f1_micro": 0.9701492537313433,
  "n_ejemplos": 120
}
```
