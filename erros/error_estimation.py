"""
error_estimation.py

Este script lê os datasets dos arquivos HDF5 artificial e experimental,
encontra matches para as mesmas alturas, calcula a matriz de ganhos K_{i,j} = M_{i,j} / \hat{M}_{i,j},
e estima a média e desvio padrão de todos os ganhos coletados.

Uso: python error_estimation.py

Requer: numpy, h5py
"""

import h5py
from pathlib import Path
import numpy as np

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
        print(f'  Alturas ({len(axis_H)}): {axis_H[:5]} ... {axis_H[-5:]}')  # Mostrar primeiras e últimas
        print(f'  K ({len(axis_K)}): {axis_K}')

        # Coletar dados por altura
        for hi, hval in enumerate(axis_H):
            hval_float = float(hval)
            if hval_float not in data:
                data[hval_float] = []
            for ki, kval in enumerate(axis_K):
                try:
                    vals = d[hi, 0, ki, 0, :]
                except Exception as e:
                    print(f'  Erro ao ler H={hi} K={ki}: {e}')
                    continue
                vals = np.asanyarray(vals).ravel()
                if vals.size != 16:
                    print(f'  Aviso: esperado 16 valores, mas obteve {vals.size} em H={hi} K={ki}, pulando')
                    continue
                data[hval_float].append((ki, vals))  # ki e vals

    return data

# Carregar dados
artificial_data = load_data_from_h5(ARTIFICIAL_FILE)
experimental_data = load_data_from_h5(EXPERIMENTAL_FILE)

if artificial_data is None or experimental_data is None:
    print('Erro ao carregar dados, saindo.')
    raise SystemExit(1)

# Coletar todos os ganhos
all_gains = []

# Para cada altura no experimental
for h_exp in experimental_data:
    h_match = round(h_exp)
    if h_match not in artificial_data:
        print(f'Altura experimental {h_exp} (arredondada para {h_match}) não encontrada nos dados artificiais, pulando.')
        continue

    print(f'Processando altura {h_exp} (match com {h_match})')

    # Para cada K no experimental
    for ki_exp, vals_exp in experimental_data[h_exp]:
        # Encontrar correspondente no artificial (assumindo mesmo ki)
        found = False
        for ki_art, vals_art in artificial_data[h_match]:
            if ki_art == ki_exp:
                found = True
                # Aplicar transformação linear para equalizar escalas
                M_hat_transformed = vals_art * 500000  # Baseado na análise de data_compare.py
                M = vals_exp.reshape(4, 4)
                M_hat = M_hat_transformed.reshape(4, 4)
                # Calcular ganhos: K = M / \hat{M}
                # Evitar divisão por zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    K = M / M_hat
                    K = np.where(M_hat != 0, K, 0)  # Set to 0 where M_hat is 0
                # Flatten e adicionar
                all_gains.extend(K.flatten())
                # Log: mostrar matriz K para algumas
                if len(all_gains) < 16*10:  # Mostrar primeiras 10 matrizes
                    print(f'  Matriz K para H={h_exp}, K={ki_exp}:')
                    print(K)
                break
        if not found:
            print(f'  K={ki_exp} não encontrado para altura {h_exp} nos dados artificiais.')

# Calcular média e desvio padrão
if all_gains:
    gains_array = np.array(all_gains)
    mean_gain = np.mean(gains_array)
    std_gain = np.std(gains_array)
    print(f'\nMédia dos ganhos: {mean_gain:.6f}')
    print(f'Desvio padrão dos ganhos: {std_gain:.6f}')
    print(f'Total de ganhos coletados: {len(all_gains)}')
else:
    print('Nenhum ganho coletado.')

print('Done')