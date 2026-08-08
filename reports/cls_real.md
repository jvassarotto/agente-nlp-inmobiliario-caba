# Evaluacion — Clasificacion multilabel

- **Modelo:** `models/cls-beto`
- **Conjunto evaluado:** `data/annotated/real_cls.jsonl` (real)
- **Ejemplos:** 65

## Reporte por clase

```
               precision    recall  f1-score   support

DUENO_DIRECTO     0.6667    0.0513    0.0952        39
  OPORTUNIDAD     0.5263    0.5263    0.5263        19
     URGENCIA     0.0000    0.0000    0.0000        15
    REFACCION     1.0000    0.0435    0.0833        23

    micro avg     0.5200    0.1354    0.2149        96
    macro avg     0.5482    0.1553    0.1762        96
 weighted avg     0.6146    0.1354    0.1628        96
  samples avg     0.1615    0.0923    0.1118        96
```

## Resumen

```json
{
  "f1_macro": 0.1762218045112782,
  "f1_micro": 0.21487603305785125,
  "n_ejemplos": 65
}
```
