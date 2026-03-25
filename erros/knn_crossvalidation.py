"""
knn_crossvalidation.py

Lê os datasets N_compound_maxenv dos arquivos HDF5 especificados,
monta um DataFrame com as 16 medidas (colunas val0..val15) e a altura (height),
separa em treino/teste e realiza k-fold cross validation nos dados de treino com k=10 folds.
O processo é executado para 1 a 5 vizinhos, e um gráfico das métricas de validação é gerado ao final.

Uso: python knn_crossvalidation.py

Requer: numpy, pandas, h5py, scikit-learn, joblib, matplotlib
"""

import h5py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt

FILES = [
    #'model_compound_4_10_34_1_interface.h5',
    #'model_compound_4_10_68_1_interface.h5',
    'model_compound_4_10_104_1_interface_synthetic_apod.h5'
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

# split into train/test -> train 85%, test 15%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"\nSplit sizes: train={X_train.shape[0]}, test={X_test.shape[0]}")

# scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# k-fold cross validation with k=10
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# metrics to evaluate
scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

# results storage
results = {}

for n_neighbors in range(1, 6):
    print(f'\nEvaluating n_neighbors={n_neighbors}')
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    
    # cross validation scores
    mae_scores = cross_val_score(knn, X_train_s, y_train, cv=kf, scoring='neg_mean_absolute_error')
    mse_scores = cross_val_score(knn, X_train_s, y_train, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(knn, X_train_s, y_train, cv=kf, scoring='r2')
    
    # convert to positive
    mae_scores = -mae_scores
    mse_scores = -mse_scores
    
    # calculate means
    mean_mae = np.mean(mae_scores)
    mean_rmse = np.sqrt(np.mean(mse_scores))  # RMSE from MSE
    mean_r2 = np.mean(r2_scores)
    
    results[n_neighbors] = {
        'MAE': mean_mae,
        'RMSE': mean_rmse,
        'R2': mean_r2
    }
    
    print(f'  Mean MAE: {mean_mae:.6f}')
    print(f'  Mean RMSE: {mean_rmse:.6f}')
    print(f'  Mean R2: {mean_r2:.6f}')

# plot the results
n_neighbors_list = list(results.keys())
mae_list = [results[n]['MAE'] for n in n_neighbors_list]
rmse_list = [results[n]['RMSE'] for n in n_neighbors_list]
r2_list = [results[n]['R2'] for n in n_neighbors_list]

fig, axs = plt.subplots(3, 1, figsize=(8, 12))

axs[0].plot(n_neighbors_list, mae_list, marker='o')
axs[0].set_title('Mean Absolute Error (MAE) vs n_neighbors')
axs[0].set_xlabel('n_neighbors')
axs[0].set_ylabel('MAE')
axs[0].grid(True)

axs[1].plot(n_neighbors_list, rmse_list, marker='o')
axs[1].set_title('Root Mean Squared Error (RMSE) vs n_neighbors')
axs[1].set_xlabel('n_neighbors')
axs[1].set_ylabel('RMSE')
axs[1].grid(True)

axs[2].plot(n_neighbors_list, r2_list, marker='o')
axs[2].set_title('R² Score vs n_neighbors')
axs[2].set_xlabel('n_neighbors')
axs[2].set_ylabel('R²')
axs[2].grid(True)

plt.tight_layout()
plt.savefig('crossvalidation_results.png')
print('\nGráfico salvo em: crossvalidation_results.png')

print('\nDone')