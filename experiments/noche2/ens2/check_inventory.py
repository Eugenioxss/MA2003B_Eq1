"""
Script para explorar el inventario de modelos honestos disponibles en noche y noche2.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('experiments/noche2/inventory.csv')
print(f"Total entries in inventory: {len(df)}")

# Filtrar modelos no honestos o ensambles
excl = df['var'].str.contains('_pp|pp_|leak_test|ensemble_', case=False, regex=True)
honest = df[~excl].copy()

print(f"Total honest candidates: {len(honest)}")
print("\nBreakdown by experiment directory:")
print(honest.groupby('exp')['var'].nunique())

print("\nBreakdown by experiment and horizon:")
print(honest.groupby(['exp', 'h'])['var'].count().unstack(fill_value=0))

print("\nModel coverage stats:")
print(f"Min val coverage: {honest['val_cov'].min()}, Min test coverage: {honest['test_cov'].min()}")
missing_cov = honest[(honest['val_cov'] < 1.0) | (honest['test_cov'] < 1.0)]
if len(missing_cov) > 0:
    print(f"Models with <100% coverage: {len(missing_cov)}")
    print(missing_cov[['exp', 'var', 'h', 'val_cov', 'test_cov']])
else:
    print("All honest models have 100% coverage!")

print("\nTop 5 models by val_rmse for each horizon:")
for h, g in honest.groupby('h'):
    print(f"\n--- Horizon h={h} ---")
    top = g.sort_values('val_rmse').head(8)
    print(top[['exp', 'var', 'val_rmse', 'test_rmse']].to_string(index=False, float_format='%.4f'))
