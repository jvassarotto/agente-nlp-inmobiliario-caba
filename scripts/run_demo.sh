#!/usr/bin/env bash
# ============================================================
#  Pipeline DEMO end-to-end sobre el dataset sintetico.
#  NO requiere scrape, ni Ollama, ni GPU.
#
#    ./scripts/run_demo.sh            -> pipeline completo (5 epochs)
#    ./scripts/run_demo.sh --quick    -> version rapida para verificar que
#                                        todo corre (1 epoch, 200 ejemplos).
# ============================================================
set -e
cd "$(dirname "$0")/.."

TRAIN_ARGS=""
if [ "$1" = "--quick" ]; then
  TRAIN_ARGS="--epochs 1 --max_train 200"
  echo "=== MODO RAPIDO: 1 epoch sobre 200 ejemplos ==="
else
  echo "=== MODO COMPLETO: segun configs/config.yaml ==="
fi

echo "[1/6] Generando dataset sintetico..."
python -m src.data.generate_synthetic

echo "[2/6] Armando splits train/val/test..."
python -m src.annotation.prepare_dataset --input data/synthetic/listings.jsonl

echo "[3/6] Fine-tuning NER (BETO)..."
python -m src.models.train_ner $TRAIN_ARGS

echo "[4/6] Fine-tuning clasificacion (BETO)..."
python -m src.models.train_classifier $TRAIN_ARGS

echo "[5/6] Evaluacion detallada..."
python -m src.models.evaluate --task ner --model_dir models/ner-beto
python -m src.models.evaluate --task cls --model_dir models/cls-beto

echo "[6/6] Robustez del parser ante cambios de layout..."
python -m src.agent.robustness

echo ""
echo "============================================================"
echo " LISTO."
echo "   Modelos   -> models/ner-beto y models/cls-beto"
echo "   Reportes  -> reports/"
echo ""
echo " Probar inferencia sobre texto libre:"
echo '   python -m src.models.infer --text "Depto a estrenar con pileta y cochera. Dueno directo."'
echo "============================================================"
