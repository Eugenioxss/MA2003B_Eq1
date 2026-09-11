#!/bin/bash
# =============================================================================
# SCRIPT MAESTRO: Evaluación por enmascaramiento de bloques
# =============================================================================
# Ejecutar desde tu terminal SSH (NO desde Antigravity) para tener GPU.
#
# Uso:
#   cd /home/emv/Multivariados
#   source .venv/bin/activate
#   bash eval_bloques/run_all.sh
#
# El script se detiene ante cualquier error.
# =============================================================================
set -euo pipefail

EVAL_DIR="/home/emv/Multivariados/eval_bloques"
SIMA_DIR="/home/emv/Multivariados/MA2003B-Equipo-6-MIMA-v6"

echo "============================================"
echo " PASO 0: Verificar GPU"
echo "============================================"
python -c "import torch; assert torch.cuda.is_available(), 'GPU NO DISPONIBLE'; print('✅ GPU:', torch.cuda.get_device_name(0))"

echo ""
echo "============================================"
echo " PASO 1: Construir CSV de entrada"
echo "============================================"
python "$EVAL_DIR/paso1_construir_csv.py"

echo ""
echo "============================================"
echo " PASO 2: Análisis de cobertura"
echo "============================================"
python "$EVAL_DIR/paso2_cobertura.py"

echo ""
echo "============================================"
echo " PASO 2b: Confirmar estación a usar"
echo "============================================"
echo ""
echo ">>> Revisa la salida arriba. El script de enmascaramiento usará"
echo ">>> la estación y año indicados como parámetros."
echo ">>> Default: SE3 2023. Para cambiar, edita paso3 o pasa argumentos."
echo ""
read -p "¿Estación a usar? (default SE3): " STATION
STATION=${STATION:-SE3}
read -p "¿Año? (default 2023): " YEAR
YEAR=${YEAR:-2023}
echo "Usando: $STATION / $YEAR"

echo ""
echo "============================================"
echo " PASO 3: Enmascarar escenarios A y B"
echo "============================================"
python "$EVAL_DIR/paso3_enmascarar.py" "$STATION" "$YEAR"

echo ""
echo "============================================"
echo " PASO 4+5: Entrenar e imputar"
echo "============================================"
echo "Esto puede tomar 15-60 minutos con GPU..."
python "$EVAL_DIR/paso456_entrenar_evaluar.py"

echo ""
echo "============================================"
echo " PASO 6: Comparar y generar reporte"
echo "============================================"
python "$EVAL_DIR/paso6_comparar.py"

echo ""
echo "============================================"
echo " ✅ PIPELINE COMPLETO"
echo "============================================"
echo "Reporte: $EVAL_DIR/evaluacion_imputacion_bloques.md"
echo "Copia:   /home/emv/Multivariados/MA2003B_Eq1/results/evaluacion_imputacion_bloques.md"
echo "Gráficas: $EVAL_DIR/resultados/"
