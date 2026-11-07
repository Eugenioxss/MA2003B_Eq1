"""Inventario de todas las predicciones val/test disponibles en experiments/noche/*/preds."""
import glob, os, re
import numpy as np, pandas as pd

rows = []
for f in glob.glob('experiments/noche/*/preds/*.parquet') + glob.glob('experiments/noche2/*/preds/*.parquet'):
    exp = f.replace('\\', '/').split('/')[-3]
    m = re.match(r'(.+)_h(\d+)\.parquet$', os.path.basename(f))
    if not m:
        continue
    var, h = m.group(1), int(m.group(2))
    try:
        d = pd.read_parquet(f, columns=['split', 'y_true', 'y_pred'])
    except Exception as e:
        print('ERR', f, e); continue
    out = {'exp': exp, 'var': var, 'h': h}
    for s in ['val', 'test']:
        x = d[(d.split == s) & d.y_true.notna()]
        ok = x.y_pred.notna()
        out[f'{s}_n'] = len(x); out[f'{s}_cov'] = ok.mean() if len(x) else np.nan
        out[f'{s}_rmse'] = np.sqrt(((x.y_true[ok] - x.y_pred[ok]) ** 2).mean()) if ok.any() else np.nan
    rows.append(out)
r = pd.DataFrame(rows).sort_values(['h', 'val_rmse'])
r.to_csv('experiments/noche2/inventory.csv', index=False)
pd.set_option('display.width', 250)
for h, g in r.groupby('h'):
    print(f'--- h={h}')
    print(g.head(12).to_string(index=False, float_format='%.3f'))
