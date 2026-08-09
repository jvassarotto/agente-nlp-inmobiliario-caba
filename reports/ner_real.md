# Evaluacion — NER (token classification)

- **Modelo:** `models/ner-beto`
- **Conjunto evaluado:** `data/annotated/real_ner.jsonl` (real)
- **Ejemplos:** 45

## Reporte por clase

```
              precision    recall  f1-score   support

     AMENITY     0.2389    0.2450    0.2419       351
  ANTIGUEDAD     0.2308    0.6000    0.3333         5
      ESTADO     0.0485    0.1515    0.0735        33
    EXPENSAS     0.0625    0.5000    0.1111         4
 ORIENTACION     0.0822    0.1765    0.1121        34

   micro avg     0.1756    0.2389    0.2024       427
   macro avg     0.1326    0.3346    0.1744       427
weighted avg     0.2100    0.2389    0.2184       427
```

## Resumen

```json
{
  "f1_micro": 0.20238095238095238,
  "n_ejemplos": 45
}
```
