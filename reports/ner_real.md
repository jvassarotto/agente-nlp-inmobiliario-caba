# Evaluacion — NER (token classification)

- **Modelo:** `models/ner-beto`
- **Conjunto evaluado:** `data/annotated/real_ner.jsonl` (real)
- **Ejemplos:** 105

## Reporte por clase

```
              precision    recall  f1-score   support

     AMENITY     0.1926    0.3381    0.2454       417
  ANTIGUEDAD     0.1905    0.2667    0.2222        15
      ESTADO     0.0739    0.4054    0.1250        37
    EXPENSAS     0.0294    0.2222    0.0519         9
 ORIENTACION     0.0769    0.1308    0.0969       107

   micro avg     0.1459    0.3009    0.1965       585
   macro avg     0.1127    0.2727    0.1483       585
weighted avg     0.1614    0.3009    0.2071       585
```

## Resumen

```json
{
  "f1_micro": 0.19653824678950305,
  "n_ejemplos": 105
}
```
