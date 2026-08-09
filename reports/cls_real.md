# Evaluacion — Clasificacion multilabel

- **Modelo:** `models/cls-beto`
- **Conjunto evaluado:** `data/annotated/real_cls.jsonl` (real)
- **Ejemplos:** 45

## Reporte por clase

```
               precision    recall  f1-score   support

DUENO_DIRECTO     0.0000    0.0000    0.0000         0
  OPORTUNIDAD     0.1818    1.0000    0.3077         2
     URGENCIA     0.0000    0.0000    0.0000         0
    REFACCION     0.0000    0.0000    0.0000         0

    micro avg     0.0645    1.0000    0.1212         2
    macro avg     0.0455    0.2500    0.0769         2
 weighted avg     0.1818    1.0000    0.3077         2
  samples avg     0.0444    0.0444    0.0444         2
```

## Resumen

```json
{
  "f1_macro": 0.07692307692307693,
  "f1_micro": 0.12121212121212122,
  "n_ejemplos": 45
}
```
