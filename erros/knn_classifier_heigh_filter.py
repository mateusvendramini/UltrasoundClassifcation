"""
knn_classifier.py

Lê os datasets N_compound_maxenv dos arquivos HDF5 especificados,
monta um DataFrame com as 16 medidas (colunas val0..val15) e a altura (height),
separa em treino/teste/validação e treina um KNN regressor para estimar a altura.

O modelo treinado (regressor + scaler) é salvo em knn_checkpoint.joblib.

Uso: python knn_classifier.py

Requer: numpy, pandas, h5py, scikit-learn, joblib
"""

import h5py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import joblib
from error_estimation_height_filter import MIN_HEIGHT, MAX_HEIGHT
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
                #print(f'  H={hval:.2f} K={kval} -> height={row["height"]:.2f}')
                #if kval != 9935.0:
                #    continue
                print(f'  H={hval:.2f} K={kval} -> height={row["height"]:.2f}')
                if MIN_HEIGHT <= hval <= MAX_HEIGHT:
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

idx = np.argsort(y)
X_sorted = X[idx]
y_sorted = y[idx]

# Número de grupos de 2
n_groups = len(y_sorted) // 2

# Trunca para múltiplo de 2 (descarta resto, se houver)
X_trim = X_sorted[:n_groups * 2]
y_trim = y_sorted[:n_groups * 2]

# Reshape para agrupar de 2 em 2
X_grouped = X_trim.reshape(n_groups, 2, -1)
y_grouped = y_trim.reshape(n_groups, 2)

# Média das alturas
y_new = y_grouped.mean(axis=1)

# Agora você tem duas opções para X:

# Opção 1: repetir X original (mais comum em regressão)
X_new = X_trim

#reescreve vetor X e y com os novos valores agrupados
X = X_new
y = np.repeat(y_new, 2)


unique_heights = np.unique(y)
train_heights, test_heights = train_test_split(unique_heights, test_size=0.05)

train_mask = np.isin(y, train_heights)
test_mask = np.isin(y, test_heights)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# # split into train/test/val -> train 70%, val 15%, test 15%
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
# # split train_val into train and val
# val_rel = 0.05 / 0.095  # relative fraction of train_val to get final 15%
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_rel, random_state=42)

print(f"\nSplit sizes: train={X_train.shape[0]}, test={X_test.shape[0]}")

# scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
#X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

from sklearn.svm import SVR

models = {
    "KNN": KNeighborsRegressor(n_neighbors=3, weights='distance'),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "GradientBoosting": GradientBoostingRegressor(),
    "SVR": SVR(kernel='rbf')
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    
    mae = metrics.mean_absolute_error(y_test, preds)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, preds))
    r2 = metrics.r2_score(y_test, preds)
    
    print(f"{name}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
    results[name] = preds
    trained_models[name] = model

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

for name, preds in results.items():
    plt.hist(preds, bins=30, alpha=0.4, label=f'{name}')

plt.hist(y_test, bins=30, alpha=0.6, color='black', label='Real')

plt.legend()
plt.title("Comparação entre modelos (TESTE)")
plt.xlabel("Altura")
plt.ylabel("Frequência")
plt.grid(True, alpha=0.3)
plt.show()
# train KNN regressor
# knn = KNeighborsRegressor(n_neighbors=3, weights='distance', n_jobs=-1)
# knn.fit(X_train_s, y_train)

# # evaluate
# def eval_set(name, Xs, ys):
#     preds = knn.predict(Xs)
#     mae = metrics.mean_absolute_error(ys, preds)
#     rmse = np.sqrt(metrics.mean_squared_error(ys, preds))
#     r2 = metrics.r2_score(ys, preds)
#     print(f"{name}: n={len(ys)} MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f}")
#     return preds

# print('\nEvaluation:')
# pred_val = eval_set('Validation', X_val_s, y_val)
# pred_test = eval_set('Test', X_test_s, y_test)

# show a few predictions vs truth
# print('\nSample predictions (val):')
# for i in range(min(10, len(y_val))):
#     print(f'  true={y_val[i]:.2f} pred={pred_val[i]:.3f} diff={pred_val[i]-y_val[i]:.3f}')

# save model checkpoint
CHECKPOINT_PATH = DATA_DIR / 'knn_checkpoint_synthetic_3neight_group2.joblib'
checkpoint = {
    'knn': trained_models,
    'scaler': scaler,
    'feature_names': [f'val{i}' for i in range(X.shape[1])],
    #'n_neighbors': knn_model.n_neighbors,
    'n_training_samples': X_train.shape[0],
}
joblib.dump(checkpoint, CHECKPOINT_PATH)
print(f'\nCheckpoint salvo em: {CHECKPOINT_PATH}')


# Histograma com comparação entre preditas e esperadas
plt.figure(figsize=(12, 6))
plt.hist(results["KNN"], bins=50, alpha=0.6, color='blue', label='Alturas Preditas (KNN)')
plt.hist(y_test, bins=50, alpha=0.6, color='orange', label='Alturas Esperadas (Reais)')
#plt.axvline(np.mean(predicted_heights_denoised), color='blue', linestyle='--', linewidth=2, label=f'Média Predita: {np.mean(predicted_heights_denoised):.2f}')
#plt.axvline(np.mean(heights), color='orange', linestyle='--', linewidth=2, label=f'Média Esperada: {np.mean(heights):.2f}')
plt.xlabel('Altura')
plt.ylabel('Frequência')
plt.title('Comparação: Distribuição das Alturas Preditas vs Esperadas')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print('\nDone?')

REAL_FILE = 'model_compound_4_10_104_1_interface_realisticfull_apod.h5'

rows_real = []

with h5py.File(DATA_DIR / REAL_FILE, 'r') as f:
    target = [None]
    def finder(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith('N_compound_maxenv'):
            target[0] = name
    f.visititems(finder)

    d = f[target[0]]
    axis_H = np.asanyarray(d.attrs.get('axis_H'))
    axis_K = np.asanyarray(d.attrs.get('axis_K'))

    for hi, hval in enumerate(axis_H):
        if not (MIN_HEIGHT <= hval <= MAX_HEIGHT):
            continue
        for ki, kval in enumerate(axis_K):
            vals = d[hi, 0, ki, 0, :]
            vals = np.asanyarray(vals).ravel() / 50000.0  # ⚠️ conversão

            row = {f'val{i}': float(vals[i]) for i in range(16)}
            row['height'] = float(hval)
            rows_real.append(row)

df_real = pd.DataFrame(rows_real)

X_real = scaler.transform(df_real[[f'val{i}' for i in range(16)]].values)
y_real = df_real['height'].values

print("\n=== Avaliação no dataset REAL ===")

for name, model in trained_models.items():
    preds = model.predict(X_real)

    mae = metrics.mean_absolute_error(y_real, preds)
    rmse = np.sqrt(metrics.mean_squared_error(y_real, preds))
    r2 = metrics.r2_score(y_real, preds)

    print(f"{name}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

plt.figure(figsize=(12,6))

plt.figure(figsize=(12,6))

for name, model in trained_models.items():
    preds = model.predict(X_real)
    plt.hist(preds, bins=30, alpha=0.4, label=name)

plt.hist(y_real, bins=30, alpha=0.6, color='black', label='Real')

plt.legend()
plt.title("Generalização no dataset REAL")
plt.xlabel("Altura")
plt.ylabel("Frequência")
plt.grid(True, alpha=0.3)
plt.show()