# Evaluacion — NER (token classification)

- **Modelo:** `models/ner-beto`
- **Conjunto evaluado:** `data/annotated/ner_test.jsonl` (sintetico)
- **Ejemplos:** 120

## Reporte por clase

```
              precision    recall  f1-score   support

     AMENITY     1.0000    1.0000    1.0000       281
  ANTIGUEDAD     0.9677    0.9890    0.9783        91
      ESTADO     0.9908    0.9818    0.9863       110
    EXPENSAS     1.0000    1.0000    1.0000       102
 ORIENTACION     1.0000    1.0000    1.0000        91

   micro avg     0.9941    0.9956    0.9948       675
   macro avg     0.9917    0.9942    0.9929       675
weighted avg     0.9942    0.9956    0.9948       675
```

## Resumen

```json
{
  "f1_micro": 0.9948186528497409,
  "n_ejemplos": 120
}
```
