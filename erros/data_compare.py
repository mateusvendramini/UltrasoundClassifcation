"""
data_compare.py

Este script compara os valores dos datasets artificial e experimental para avaliar
se é necessária uma transformação linear entre os dois conjuntos.

Para cada match de altura, coleta os valores M_art e M_exp, e realiza regressão linear
para estimar slope, intercept e R².

Uso: python data_compare.py

Requer: numpy, h5py, scikit-learn
"""

import h5py
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Arquivos
ARTIFICIAL_FILE = 'model_compound_4_10_104_1_interface_synthetic_apod.h5'
EXPERIMENTAL_FILE = 'model_compound_4_10_104_1_interface_realisticfull_apod.h5'

DATA_DIR = Path('.')

def load_data_from_h5(fname):
    """Carrega os dados do arquivo HDF5, retornando um dict com alturas e valores."""
    file_path = DATA_DIR / fname
    print(f'Lendo arquivo: {file_path}')
    if not file_path.exists():
        print(f'  Arquivo não encontrado: {file_path}')
        return None

    data = {}
    with h5py.File(file_path, 'r') as f:
        # Encontrar dataset N_compound_maxenv
        target = [None]
        def finder(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith('N_compound_maxenv'):
                target[0] = name
        f.visititems(finder)
        if target[0] is None:
            print(f'  Dataset N_compound_maxenv não encontrado em {fname}')
            return None

        d = f[target[0]]
        shape = getattr(d, 'shape', None)
        print(f'  Encontrado {target[0]}, shape={shape}')

        axis_H = np.asanyarray(d.attrs.get('axis_H', np.arange(shape[0])))
        axis_K = np.asanyarray(d.attrs.get('axis_K', np.arange(shape[2])))
        print(f'  Alturas ({len(axis_H)}): {axis_H[:5]} ... {axis_H[-5:]}')
        print(f'  K ({len(axis_K)}): {axis_K}')

        # Coletar dados por altura e velocidade
        for hi, hval in enumerate(axis_H):
            hval_float = float(hval)
            if hval_float not in data:
                data[hval_float] = {}
            for ki, kval in enumerate(axis_K):
                kval_float = float(kval)
                try:
                    vals = d[hi, 0, ki, 0, :]
                except Exception as e:
                    print(f'  Erro ao ler H={hi} K={ki}: {e}')
                    continue
                vals = np.asanyarray(vals).ravel()
                if vals.size != 16:
                    print(f'  Aviso: esperado 16 valores, mas obteve {vals.size} em H={hi} K={ki}, pulando')
                    continue
                data[hval_float][kval_float] = vals

    return data

# Carregar dados
artificial_data = load_data_from_h5(ARTIFICIAL_FILE)
experimental_data = load_data_from_h5(EXPERIMENTAL_FILE)

if artificial_data is None or experimental_data is None:
    print('Erro ao carregar dados, saindo.')
    raise SystemExit(1)

# Coletar todos os pares M_art vs M_exp
all_art = []
all_exp = []

# Para cada altura no experimental
for h_exp in experimental_data:
    h_match = round(h_exp)
    if h_match not in artificial_data:
        print(f'Altura experimental {h_exp} (arredondada para {h_match}) não encontrada nos dados artificiais, pulando.')
        continue

    print(f'Comparando altura {h_exp} (match com {h_match})')

    # Para cada velocidade no experimental
    for kval_exp, vals_exp in experimental_data[h_exp].items():
        kval_match = round(kval_exp)
        if kval_match not in artificial_data[h_match]:
            print(f'  Velocidade experimental {kval_exp} (arredondada para {kval_match}) não encontrada nos dados artificiais para altura {h_exp}, pulando.')
            continue

        vals_art = artificial_data[h_match][kval_match]
        # Coletar pares
        all_art.extend(vals_art)
        all_exp.extend(vals_exp)

# Converter para arrays
all_art = np.array(all_art)
all_exp = np.array(all_exp)

print(f'\nTotal de pontos coletados: {len(all_art)}')

if len(all_art) == 0:
    print('Nenhum dado coletado.')
    raise SystemExit(1)

# Regressão linear
model = LinearRegression()
model.fit(all_art.reshape(-1, 1), all_exp)

slope = model.coef_[0]
intercept = model.intercept_
r2 = r2_score(all_exp, model.predict(all_art.reshape(-1, 1)))

print(f'\nRegressão linear:')
print(f'  Slope (a): {slope:.6f}')
print(f'  Intercept (b): {intercept:.6f}')
print(f'  R²: {r2:.6f}')

# Estatísticas básicas
print(f'\nEstatísticas:')
print(f'  Média M_art: {np.mean(all_art):.6f}')
print(f'  Média M_exp: {np.mean(all_exp):.6f}')
print(f'  Desvio M_art: {np.std(all_art):.6f}')
print(f'  Desvio M_exp: {np.std(all_exp):.6f}')

print('Done')