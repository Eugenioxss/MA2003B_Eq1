import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from functools import reduce

print("1. Cargando y uniendo variables base...")
variables = ['O3', 'NOX', 'SR', 'TOUT', 'WSR', 'WDR', 'PM10', 'PM2.5']
dfs = []
for var in variables:
    path = f"data/processed/variables/{var}_clean.parquet"
    if os.path.exists(path):
        dfs.append(pd.read_parquet(path))

df_ml = reduce(lambda left, right: pd.merge(left, right, on=['Date', 'Estacion'], how='outer'), dfs)
df_ml.sort_values(by=['Estacion', 'Date'], inplace=True)
df_ml.reset_index(drop=True, inplace=True)

# Excluir NE3 y NO3
df_ml = df_ml[~df_ml['Estacion'].isin(['NE3', 'NO3'])]

# U y V
radianes = df_ml['WDR'] * (np.pi / 180)
df_ml['U_Wind'] = -df_ml['WSR'] * np.sin(radianes)
df_ml['V_Wind'] = -df_ml['WSR'] * np.cos(radianes)
df_ml.drop(columns=['WDR', 'WSR'], inplace=True)

print("2. Reindexando y creando lags (manteniendo NaNs)...")
start_date = df_ml['Date'].min()
end_date = df_ml['Date'].max()
full_idx = pd.date_range(start=start_date, end=end_date, freq='h')

def process_station(group):
    group = group.set_index('Date')
    group = group[~group.index.duplicated(keep='first')]
    group = group.reindex(full_idx)
    
    # Target: O3 futuro o actual? Asumiremos predecir O3_actual basado en lags, o predecir a futuro.
    # El modelo ARX usualmente predice O3_t usando variables en t, t-1, etc.
    # Como queremos predecir O3, usaremos Lags ESTRICTOS (t-1, t-2, t-24) para no hacer trampa.
    
    # Lags de O3
    for lag in [1, 2, 24]:
        group[f'O3_lag_{lag}'] = group['O3'].shift(lag)
        
    # Lags de otras variables
    for lag in [1, 2, 4, 12, 24]:
        group[f'NOX_lag_{lag}'] = group['NOX'].shift(lag)
        group[f'SR_lag_{lag}'] = group['SR'].shift(lag)
        group[f'TOUT_lag_{lag}'] = group['TOUT'].shift(lag)
        group[f'PM10_lag_{lag}'] = group['PM10'].shift(lag)
        group[f'PM2.5_lag_{lag}'] = group['PM2.5'].shift(lag)
        group[f'U_lag_{lag}'] = group['U_Wind'].shift(lag)
        group[f'V_lag_{lag}'] = group['V_Wind'].shift(lag)
        
    # Extraer variables temporales
    group['hour'] = group.index.hour
    group['month'] = group.index.month
    group['dayofweek'] = group.index.dayofweek
    
    return group

df_full = df_ml.groupby('Estacion').apply(process_station).reset_index()
df_full.rename(columns={'level_1': 'Date'}, inplace=True)
# Si level_1 no está, usar df_full = ... y reset_index asume 'Date' u otro nombre
if 'Date' not in df_full.columns:
    df_full.rename(columns={df_full.columns[1]: 'Date'}, inplace=True)

print(f"Dimensiones del dataset con NaNs conservados: {df_full.shape}")

# El Target será predecir O3 actual basado en lags (no podemos usar O3, ni variables en t porque en la vida real no las tenemos hasta medir)
# NOTA: Si el modelo ARX usa NOX_t para predecir O3_t, eso no es pronóstico. Aquí usaremos puramente lags.
target = 'O3'

# Quitar filas donde el TARGET sea NaN (no podemos entrenar ni evaluar si no sabemos la respuesta real)
df_model = df_full.dropna(subset=[target]).copy()

# Features
features = [c for c in df_model.columns if 'lag_' in c] + ['hour', 'month', 'dayofweek']
print(f"Número de features: {len(features)}")

df_model['year'] = df_model['Date'].dt.year
train = df_model[df_model['year'] <= 2023]
val = df_model[df_model['year'] == 2024]
test = df_model[df_model['year'] == 2025]

X_train, y_train = train[features], train[target]
X_val, y_val = val[features], val[target]
X_test, y_test = test[features], test[target]

print(f"N Train: {len(X_train):,}, N Val: {len(X_val):,}, N Test: {len(X_test):,}")

print("3. Entrenando LightGBM...")
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

print("\n4. Evaluando en Test 2025...")
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

print("==================================================")
print(f"RMSE en Test 2025 (Puros Lags): {rmse_test:.4f} ppb")
print(f"Muestras de Test Evaluadas: {len(y_test):,} (Esto es cuántos picos/horas sobrevivieron gracias a que XGBoost maneja los NaNs)")
print("==================================================")

# Guardar un feature importance rápido
imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importance()}).sort_values('Importance', ascending=False)
print("\nTop 10 Features:")
print(imp.head(10).to_markdown())

with open('reporte_lgbm.md', 'w') as f:
    f.write(f"# Entrenamiento Directo LightGBM (Con NaNs)\n")
    f.write(f"- RMSE en Test 2025: **{rmse_test:.4f} ppb**\n")
    f.write(f"- Horas evaluadas en Test: **{len(y_test):,}**\n")
    f.write(f"\n## Top 10 Features\n")
    f.write(imp.head(10).to_markdown())
