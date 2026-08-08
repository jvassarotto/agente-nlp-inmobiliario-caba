# Evaluacion — NER (token classification)

- **Modelo:** `models/ner-beto`
- **Conjunto evaluado:** `data/annotated/ner_test.jsonl` (sintetico)
- **Ejemplos:** 120

## Reporte por clase

```
              precision    recall  f1-score   support

     AMENITY     1.0000    1.0000    1.0000       281
  ANTIGUEDAD     0.9785    1.0000    0.9891        91
      ESTADO     1.0000    0.9818    0.9908       110
    EXPENSAS     1.0000    1.0000    1.0000       102
 ORIENTACION     1.0000    1.0000    1.0000        91

   micro avg     0.9970    0.9970    0.9970       675
   macro avg     0.9957    0.9964    0.9960       675
weighted avg     0.9971    0.9970    0.9970       675
```

## Resumen

```json
{
  "f1_micro": 0.997037037037037,
  "n_ejemplos": 120
}
```
