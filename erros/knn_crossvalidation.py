"""
knn_crossvalidation.py

Lê os datasets N_compound_maxenv dos arquivos HDF5 especificados,
monta um DataFrame com as 16 medidas (colunas val0..val15) e a altura (height),
filtra alturas no range MIN_HEIGHT <= height <= MAX_HEIGHT,
agrupa os dados em grupos de tamanho group_size (1 a 5),
separa em treino/teste baseado em alturas únicas,
e realiza k-fold cross validation nos dados de treino com k=10 folds para n_neighbors de 1 a 20.
Os resultados das 100 validações (5 group_sizes x 20 n_neighbors) são salvos em um CSV.

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
                rows.append(row)

if not rows:
    print('No data collected, exiting.')
    raise SystemExit(1)

# build DataFrame
df = pd.DataFrame(rows)
print('\nDataFrame shape before filtering:', df.shape)

# Filter heights in range
df = df[(df['height'] >= MIN_HEIGHT) & (df['height'] <= MAX_HEIGHT)]
print('DataFrame shape after filtering:', df.shape)

if df.empty:
    print('No data after filtering, exiting.')
    raise SystemExit(1)

# Sort by height
df = df.sort_values('height')

print('Sample after sorting:')
print(df.head())

X = df[[f'val{i}' for i in range(16)]].values
y = df['height'].values

# results storage
results = []

for group_size in range(1, 6):
    print(f'\nProcessing group_size={group_size}')
    
    # Número de grupos
    n_groups = len(y) // group_size
    
    # Trunca para múltiplo de group_size
    X_trim = X[:n_groups * group_size]
    y_trim = y[:n_groups * group_size]
    
    # Reshape para agrupar
    X_grouped = X_trim.reshape(n_groups, group_size, -1)
    y_grouped = y_trim.reshape(n_groups, group_size)
    
    # Média das alturas
    y_new = y_grouped.mean(axis=1)
    
    # X_new: repetir X original
    X_new = X_trim
    
    # Reescrever y com repetição
    y_final = np.repeat(y_new, group_size)
    
    # Split baseado em alturas únicas
    unique_heights = np.unique(y_final)
    train_heights, test_heights = train_test_split(unique_heights, test_size=0.05, random_state=42)
    
    train_mask = np.isin(y_final, train_heights)
    test_mask = np.isin(y_final, test_heights)
    
    X_train, X_test = X_new[train_mask], X_new[test_mask]
    y_train, y_test = y_final[train_mask], y_final[test_mask]
    
    print(f'  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # k-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    for n_neighbors in range(1, 21):
        print(f'    Evaluating n_neighbors={n_neighbors}')
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
        
        results.append({
            'group_size': group_size,
            'n_neighbors': n_neighbors,
            'MAE': mean_mae,
            'RMSE': mean_rmse,
            'R2': mean_r2
        })
        
        print(f'      Mean MAE: {mean_mae:.6f}, RMSE: {mean_rmse:.6f}, R2: {mean_r2:.6f}')

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('crossvalidation_results_grouped.csv', index=False, float_format='%.6f')
print('\nResults saved to crossvalidation_results_grouped.csv')

# Find best n_neighbors for each group_size
best_neighbors = {}
for group_size in range(1, 6):
    group_results = [r for r in results if r['group_size'] == group_size]
    best = max(group_results, key=lambda x: x['R2'])
    best_neighbors[group_size] = best['n_neighbors']
    print(f'Best n_neighbors for group_size {group_size}: {best["n_neighbors"]} (R2={best["R2"]:.6f})')

# Second validation with realistic data
print('\nStarting second validation with realistic data')

# Use synthetic data with group_size=1 for training
group_size = 1
n_groups = len(y) // group_size
X_trim = X[:n_groups * group_size]
y_trim = y[:n_groups * group_size]
X_new = X_trim
y_final = y_trim

# Split based on unique heights
unique_heights = np.unique(y_final)
train_heights, _ = train_test_split(unique_heights, test_size=0.05, random_state=42)
train_mask = np.isin(y_final, train_heights)
X_train_synth = X_new[train_mask]
y_train_synth = y_final[train_mask]

scaler_synth = StandardScaler()
X_train_synth_s = scaler_synth.fit_transform(X_train_synth)

# Load realistic data
realistic_file = 'model_compound_4_10_104_1_interface_realisticfull_apod.h5'
file_path = DATA_DIR / realistic_file
if not file_path.exists():
    print('Realistic file not found, skipping second validation')
else:
    rows_real = []
    with h5py.File(file_path, 'r') as f:
        target = [None]
        def finder(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith('N_compound_maxenv'):
                target[0] = name
        f.visititems(finder)
        if target[0] is None:
            print('N_compound_maxenv not found in realistic file')
        else:
            d = f[target[0]]
            shape = getattr(d, 'shape', None)
            axis_H = np.asanyarray(d.attrs.get('axis_H', np.arange(shape[0])))
            axis_K = np.asanyarray(d.attrs.get('axis_K', np.arange(shape[2])))
            for hi, hval in enumerate(axis_H):
                for ki, kval in enumerate(axis_K):
                    try:
                        vals = d[hi, 0, ki, 0, :]
                    except Exception as e:
                        continue
                    vals = np.asanyarray(vals).ravel()
                    if vals.size != 16:
                        continue
                    row = {f'val{i}': float(vals[i]) for i in range(16)}
                    row['height'] = float(hval)
                    rows_real.append(row)

    df_real = pd.DataFrame(rows_real)
    df_real = df_real[(df_real['height'] >= MIN_HEIGHT) & (df_real['height'] <= MAX_HEIGHT)]
    if df_real.empty:
        print('No realistic data after filtering')
    else:
        X_real = df_real[[f'val{i}' for i in range(16)]].values / 500000
        y_real = df_real['height'].values

        # Select 15% arbitrarily for validation (test)
        _, X_val, _, y_val = train_test_split(X_real, y_real, test_size=0.15, random_state=42)
        
        print(f'Loaded realistic validation data: {X_val.shape[0]} samples')

        results = []

        for group_size in range(1, 6):
            print(f'\nProcessing group_size={group_size} for validation')
            
            # Número de grupos
            n_groups = len(y) // group_size
            
            # Trunca para múltiplo de group_size
            X_trim = X[:n_groups * group_size]
            y_trim = y[:n_groups * group_size]
            
            # Reshape para agrupar
            X_grouped = X_trim.reshape(n_groups, group_size, -1)
            y_grouped = y_trim.reshape(n_groups, group_size)
            
            # Média das alturas
            y_new = y_grouped.mean(axis=1)
            
            # X_new: repetir X original
            X_new = X_trim
            
            # Reescrever y com repetição
            y_final = np.repeat(y_new, group_size)
            
            # Split baseado em alturas únicas
            unique_heights = np.unique(y_final)
            train_heights, test_heights = train_test_split(unique_heights, test_size=0.05, random_state=42)
            
            train_mask = np.isin(y_final, train_heights)
            test_mask = np.isin(y_final, test_heights)
            
            X_train, X_test = X_new[train_mask], X_new[test_mask]
            y_train, y_test = y_final[train_mask], y_final[test_mask]
            
            print(f'  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')
            
            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            for n_neighbors in range(1, 21):
                print(f'    Evaluating n_neighbors={n_neighbors}')
                knn = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
                knn.fit(X_train_s, y_train)  # Fit on training data
                
                # Transform validation data using the scaler from this group_size
                X_val_s = scaler.transform(X_val)
                y_pred = knn.predict(X_val_s)
    
                # Validation scores (already scalar values)
                mae = metrics.mean_absolute_error(y_val, y_pred)
                mse = metrics.mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                r2 = metrics.r2_score(y_val, y_pred)
                
                results.append({
                    'group_size': group_size,
                    'n_neighbors': n_neighbors,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                })
                
                print(f'      MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}')
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('validation_results_grouped.csv', index=False, float_format='%.6f')
        print('\nResults saved to validation_results_grouped.csv')