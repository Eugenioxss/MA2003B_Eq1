"""
experiments/planB_picos/run_pipeline_remaining.py
=================================================
Ejecuta secuencialmente los módulos restantes del experimento planB_picos:
 1. 02_prediction_intervals.py (Intervalos conformes, condicionales y CQR al 80%)
 2. 03_exceedance_classification.py (Alertas NOM-020 90 ppb y calibración p*)
 3. 04_peak_correction.py (Regresión isotónica y tail scaling)
 4. 05_generate_figures_and_report.py (Generación de todas las figuras diagnósticas)
"""

import os, sys, time, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
EXP_DIR = os.path.join(ROOT, 'experiments', 'planB_picos')

def run_step(script_name, desc):
    print("\n" + "=" * 70, flush=True)
    print(f"EJECUTANDO: {desc} ({script_name})", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    script_path = os.path.join(EXP_DIR, script_name)
    ret = subprocess.run([PY, '-u', script_path], cwd=ROOT)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"ERROR: {script_name} falló con código {ret.returncode}", flush=True)
        sys.exit(ret.returncode)
    else:
        print(f"EXITOSO: {script_name} completado en {elapsed:.1f}s", flush=True)

def main():
    print("Iniciando pipeline integral planB_picos...", flush=True)
    run_step('02_prediction_intervals.py', 'Evaluación de Intervalos de Incertidumbre y CQR')
    run_step('03_exceedance_classification.py', 'Clasificación de Alertas NOM-020 y Calibración')
    run_step('04_peak_correction.py', 'Correcciones Post-Hoc Isotónicas y de Cola')
    run_step('05_generate_figures_and_report.py', 'Generación de Figuras Diagnósticas')
    print("\nPIPELINE COMPLETO FINALIZADO.", flush=True)

if __name__ == '__main__':
    main()
