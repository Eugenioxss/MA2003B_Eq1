#!/usr/bin/env python3
"""
PASO 4+5+6: Entrenar, imputar, y evaluar — todo en un solo runner script.

Este script:
1. Inspecciona el dataset enmascarado con sima.py inspect
2. Entrena el modelo con parámetros por defecto (cronometra 1ª época)
3. Imputa el dataset completo
4. Extrae las reconstrucciones para los escenarios A y B
5. Compara vs ground truth: RMSE, R², RMSE-picos, gráficas overlay

Uso:
  source /home/emv/Multivariados/.venv/bin/activate
  cd /home/emv/Multivariados/MA2003B-Equipo-6-MIMA-v6
  python /home/emv/Multivariados/eval_bloques/paso456_entrenar_evaluar.py
"""
import subprocess
import sys
import os
import time

SIMA_DIR = "/home/emv/Multivariados/MA2003B-Equipo-6-MIMA-v6"
SIMA_PY = os.path.join(SIMA_DIR, "sima.py")
EVAL_DIR = "/home/emv/Multivariados/eval_bloques"
INPUT_CSV = os.path.join(EVAL_DIR, "sima_input_masked.csv")
CHECKPOINT = os.path.join(EVAL_DIR, "models", "sima_eval.pt")
OUTPUT_DIR = os.path.join(EVAL_DIR, "outputs")
IMPUTED_CSV = os.path.join(EVAL_DIR, "outputs", "sima_imputado.csv")

os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run(cmd, description):
    """Run command, print output, and return exit code."""
    print(f"\n{'='*70}")
    print(f">>> {description}")
    print(f">>> {' '.join(cmd)}")
    print('='*70)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=SIMA_DIR)
    elapsed = time.time() - t0
    print(f"\n[{description}] terminó en {elapsed:.1f}s (exit code: {result.returncode})")
    if result.returncode != 0:
        print(f"⚠️  Error en: {description}")
        sys.exit(1)
    return elapsed

# === PASO 4a: Inspect ===
run(
    [sys.executable, SIMA_PY, "inspect", "--input", INPUT_CSV],
    "Inspeccionar dataset enmascarado"
)

# === PASO 4b: Train ===
# Parámetros por defecto del repo:
#   --epochs 50, --seq-len 168, --hidden-size 64, --n-heads 4, --n-layers 2
#   --diffusion-steps 50, --dropout 0.1, --residual-scale 0.1
#   --learning-rate 1e-3, --mask-ratio 0.2, --train-stride 24, --batch-size 32
train_time = run(
    [sys.executable, SIMA_PY, "train",
     "--input", INPUT_CSV,
     "--checkpoint", CHECKPOINT,
     "--epochs", "50",
     "--device", "auto",
     "--output-dir", os.path.join(OUTPUT_DIR, "training"),
    ],
    "Entrenar modelo (50 épocas, parámetros por defecto)"
)

# === PASO 5: Impute ===
impute_time = run(
    [sys.executable, SIMA_PY, "impute",
     "--input", INPUT_CSV,
     "--checkpoint", CHECKPOINT,
     "--output", IMPUTED_CSV,
     "--device", "auto",
    ],
    "Imputar dataset completo"
)

print(f"\n{'='*70}")
print(f"✅ Entrenamiento completado en {train_time:.1f}s")
print(f"✅ Imputación completada en {impute_time:.1f}s")
print(f"✅ Dataset imputado: {IMPUTED_CSV}")
print(f"\nAhora ejecuta: python {EVAL_DIR}/paso6_comparar.py")
print('='*70)
