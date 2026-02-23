"""
knn_classifier.py

Lê os datasets N_compound_maxenv dos arquivos HDF5 especificados,
monta um DataFrame com as 16 medidas (colunas val0..val15) e a altura (height),
separa em treino/teste/validação e treina um KNN regressor para estimar a altura.

Uso: python knn_classifier.py

Requer: numpy, pandas, h5py, scikit-learn
"""

import h5py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

FILES = [
    'model_compound_4_10_34_1_interface.h5',
    'model_compound_4_10_68_1_interface.h5',
]

DATA_DIR = Path('.')
rows = []

for fname in FILES:
    file_path = DATA_DIR / fname
    print(f'Processing file: {file_path}')
    if not file_path.exists():
        print('  File not found, skipping:', file_path)
        continue
    with h5py.File(file_path, 'r') as f:
        # locate dataset by suffix
        target = [None]
        def finder(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith('N_compound_maxenv'):
                target[0] = name
        f.visititems(finder)
        if target[0] is None:
            print('  N_compound_maxenv not found in', fname)
            continue
        d = f[target[0]]
        shape = getattr(d, 'shape', None)
        print('  found', target[0], 'shape=', shape)

        axis_H = np.asanyarray(d.attrs.get('axis_H', np.arange(shape[0])))
        axis_K = np.asanyarray(d.attrs.get('axis_K', np.arange(shape[2])))

        # iterate H x K and read 16 measurements
        for hi, hval in enumerate(axis_H):
            for ki, kval in enumerate(axis_K):
                try:
                    vals = d[hi, 0, ki, 0, :]
                except Exception as e:
                    print(f'  skipping H={hi} K={ki} read error: {e}')
                    continue
                vals = np.asanyarray(vals).ravel()
                if vals.size != 16:
                    print(f'  warning: expected 16 values but got {vals.size} at H={hi} K={ki}, skipping')
                    continue
                row = {f'val{i}': float(vals[i]) for i in range(16)}
                row['height'] = float(hval)
                rows.append(row)

if not rows:
    print('No data collected, exiting.')
    raise SystemExit(1)

# build DataFrame
df = pd.DataFrame(rows)
print('\nDataFrame shape:', df.shape)
print('Sample:')
print(df.head())

X = df[[f'val{i}' for i in range(16)]].values
y = df['height'].values

# split into train/test/val -> train 70%, val 15%, test 15%
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# split train_val into train and val
val_rel = 0.15 / 0.85  # relative fraction of train_val to get final 15%
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_rel, random_state=42)

print(f"\nSplit sizes: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

# scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# train KNN regressor
knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_s, y_train)

# evaluate
def eval_set(name, Xs, ys):
    preds = knn.predict(Xs)
    mae = metrics.mean_absolute_error(ys, preds)
    rmse = np.sqrt(metrics.mean_squared_error(ys, preds))
    r2 = metrics.r2_score(ys, preds)
    print(f"{name}: n={len(ys)} MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f}")
    return preds

print('\nEvaluation:')
pred_val = eval_set('Validation', X_val_s, y_val)
pred_test = eval_set('Test', X_test_s, y_test)

# show a few predictions vs truth
print('\nSample predictions (val):')
for i in range(min(10, len(y_val))):
    print(f'  true={y_val[i]:.2f} pred={pred_val[i]:.3f} diff={pred_val[i]-y_val[i]:.3f}')

print('\nDone')
