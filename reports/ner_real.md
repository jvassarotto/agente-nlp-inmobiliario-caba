# Evaluacion — NER (token classification)

- **Modelo:** `models/ner-beto`
- **Conjunto evaluado:** `data/annotated/real_ner.jsonl` (real)
- **Ejemplos:** 65

## Reporte por clase

```
              precision    recall  f1-score   support

     AMENITY     0.2248    0.3268    0.2664       205
  ANTIGUEDAD     0.0909    0.1667    0.1176         6
      ESTADO     0.0455    0.1500    0.0698        20
    EXPENSAS     0.0526    0.2500    0.0870         4
 ORIENTACION     0.1591    0.2414    0.1918        58

   micro avg     0.1784    0.2935    0.2219       293
   macro avg     0.1146    0.2270    0.1465       293
weighted avg     0.1945    0.2935    0.2327       293
```

## Resumen

```json
{
  "f1_micro": 0.22193548387096776,
  "n_ejemplos": 65
}
```
