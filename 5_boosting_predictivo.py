import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

print("=======================================================")
print(" SCRIPT 5: PREDICCIÓN CON GRADIENT BOOSTING (LightGBM)")
print("=======================================================")
print("Este modelo resuelve el problema de los 'huecos dispersos'")
print("al no exigir ventanas temporales continuas, usando lags puntuales.")

# 1. CARGA Y UNIÓN DE VARIABLES BASE (Conservando NaNs)
print("\n[1/4] Cargando variables y reindexando el tiempo...")
variables = ['O3', 'NOX', 'SR', 'TOUT', 'WSR', 'WDR', 'PM10', 'PM2.5']
dfs = []
for var in variables:
    path = f"data/processed/variables/{var}_clean.parquet"
    if os.path.exists(path):
        dfs.append(pd.read_parquet(path))

df_ml = reduce(lambda left, right: pd.merge(left, right, on=['Date', 'Estacion'], how='outer'), dfs)
df_ml.sort_values(by=['Estacion', 'Date'], inplace=True)
df_ml.reset_index(drop=True, inplace=True)

# Excluir estaciones con demasiados problemas estructurales
df_ml = df_ml[~df_ml['Estacion'].isin(['NE3', 'NO3'])]

# Vectores de Viento
radianes = df_ml['WDR'] * (np.pi / 180)
df_ml['U_Wind'] = -df_ml['WSR'] * np.sin(radianes)
df_ml['V_Wind'] = -df_ml['WSR'] * np.cos(radianes)
df_ml.drop(columns=['WDR', 'WSR'], inplace=True)

# 2. GENERACIÓN DE REZAGOS (LAGS) PUNTUALES
start_date = df_ml['Date'].min()
end_date = df_ml['Date'].max()
full_idx = pd.date_range(start=start_date, end=end_date, freq='h')

def process_station(group):
    # Reindexar estrictamente cada hora calendario (introduce NaNs en huecos reales)
    group = group.set_index('Date')
    group = group[~group.index.duplicated(keep='first')]
    group = group.reindex(full_idx)
    
    # Lags de Ozono (1h, 2h, y el ciclo exacto del día anterior: 24h)
    for lag in [1, 2, 24]:
        group[f'O3_lag_{lag}'] = group['O3'].shift(lag)
        
    # Lags de precursores y clima (corto plazo y ciclo diario)
    for lag in [1, 2, 4, 12, 24]:
        for feature in ['NOX', 'SR', 'TOUT', 'PM10', 'PM2.5', 'U_Wind', 'V_Wind']:
            group[f'{feature}_lag_{lag}'] = group[feature].shift(lag)
            
    # Variables de calendario (estacionalidad y ciclo circadiano)
    group['hour'] = group.index.hour
    group['month'] = group.index.month
    group['dayofweek'] = group.index.dayofweek
    
    return group

print("\n[2/4] Generando variables rezagadas (Lags puntuales)...")
df_full = df_ml.groupby('Estacion').apply(process_station).reset_index()
if 'Date' not in df_full.columns:
    df_full.rename(columns={df_full.columns[1]: 'Date'}, inplace=True)

# Preparar Target (Predeciremos O3 actual)
# Eliminamos filas donde el TARGET sea NaN (no se puede evaluar lo que no se midió)
df_model = df_full.dropna(subset=['O3']).copy()

features = [c for c in df_model.columns if 'lag_' in c] + ['hour', 'month', 'dayofweek']
target = 'O3'

# 3. PARTICIÓN DE DATOS Y ENTRENAMIENTO
print("\n[3/4] Entrenando LightGBM...")
df_model['year'] = df_model['Date'].dt.year
train = df_model[df_model['year'] <= 2023]
val = df_model[df_model['year'] == 2024]
test = df_model[df_model['year'] == 2025]

X_train, y_train = train[features], train[target]
X_val, y_val = val[features], val[target]
X_test, y_test = test[features], test[target]

print(f"  --> Train (2020-2023): {len(X_train):,} observaciones")
print(f"  --> Val (2024):        {len(X_val):,} observaciones")
print(f"  --> Test (2025):       {len(X_test):,} observaciones")

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'seed': 42,
    'verbose': -1
}

# Entrenamiento con Early Stopping
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1500,
    valid_sets=[lgb_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0) # Silenciar logs por iteración
    ]
)

# 4. EVALUACIÓN Y EXPORTACIÓN
print("\n[4/4] Evaluando modelo en la partición Test (2025)...")
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test = r2_score(y_test, y_pred)

print(f"\n==============================================")
print(f" MÉTRICAS FINALES TEST 2025 (Out-of-Sample)")
print(f"==============================================")
print(f" RMSE: {rmse_test:.4f} ppb")
print(f" R²:   {r2_test:.4f}")
print(f" Cobertura Test: {len(y_test):,} horas evaluadas")
print(f"==============================================\n")

# Guardar modelo
os.makedirs("results/models", exist_ok=True)
model.save_model('results/models/lightgbm_o3_predictor.txt')
print("Modelo guardado en 'results/models/lightgbm_o3_predictor.txt'")

# Gráfica de Importancia de Variables
os.makedirs("results/figures", exist_ok=True)
imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importance()})
imp = imp.sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=imp, ax=ax, palette='viridis')
ax.set_title('Top 15 Variables Predictoras (Feature Importance)', fontsize=14)
ax.set_xlabel('Ganancia / Importancia')
ax.set_ylabel('')
plt.tight_layout()
fig.savefig('results/figures/lgbm_feature_importance.png', dpi=300)
plt.close(fig)
print("Gráfica de importancia guardada en 'results/figures/lgbm_feature_importance.png'")

print("\n¡Pipeline predictivo completado con éxito!")
