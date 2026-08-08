@echo off
REM ============================================================
REM  Pipeline DEMO end-to-end sobre el dataset sintetico.
REM  NO requiere scrape, ni Ollama, ni GPU.
REM
REM    scripts\run_demo.bat            -> pipeline completo (5 epochs)
REM    scripts\run_demo.bat --quick    -> version rapida para verificar que
REM                                       todo corre (1 epoch, 200 ejemplos).
REM                                       Pensado para correr en CPU.
REM ============================================================
setlocal
cd /d "%~dp0.."

set QUICK=0
if /I "%~1"=="--quick" set QUICK=1

if "%QUICK%"=="1" (
    set TRAIN_ARGS=--epochs 3 --max_train 300
    REM Los reportes van a un tag propio para NO pisar los resultados reales
    REM versionados en reports/ner_sintetico.md y reports/cls_sintetico.md.
    set EVAL_TAG=quick
    echo ============================================================
    echo  MODO RAPIDO: 3 epochs sobre 300 ejemplos.
    echo  Sirve para verificar que el pipeline corre de punta a punta.
    echo  Las metricas que salgan de aca NO son los resultados del
    echo  trabajo: con tan pocos datos el modelo apenas aprende.
    echo  Los resultados reales estan en reports\ner_sintetico.md,
    echo  cls_sintetico.md, ner_real.md y cls_real.md.
    echo ============================================================
) else (
    set TRAIN_ARGS=
    set EVAL_TAG=sintetico
    echo === MODO COMPLETO: segun configs/config.yaml ===
)

echo.
echo [1/6] Generando dataset sintetico...
python -m src.data.generate_synthetic || goto :err

echo [2/6] Armando splits train/val/test...
python -m src.annotation.prepare_dataset --input data/synthetic/listings.jsonl || goto :err

echo [3/6] Fine-tuning NER (BETO)...
python -m src.models.train_ner %TRAIN_ARGS% || goto :err

echo [4/6] Fine-tuning clasificacion (BETO)...
python -m src.models.train_classifier %TRAIN_ARGS% || goto :err

echo [5/6] Evaluacion detallada...
python -m src.models.evaluate --task ner --model_dir models/ner-beto --tag %EVAL_TAG% || goto :err
python -m src.models.evaluate --task cls --model_dir models/cls-beto --tag %EVAL_TAG% || goto :err

echo [6/6] Robustez del parser ante cambios de layout...
python -m src.agent.robustness || goto :err

echo.
echo ============================================================
echo  LISTO.
echo    Modelos   -^> models\ner-beto y models\cls-beto
echo    Reportes  -^> reports\
echo.
echo  Probar inferencia sobre texto libre:
echo    python -m src.models.infer --text "Depto a estrenar con pileta y cochera. Dueno directo."
echo ============================================================
goto :eof

:err
echo.
echo ERROR en el pipeline. Revisa el mensaje anterior.
exit /b 1
