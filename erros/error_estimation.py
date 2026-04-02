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

        # Coletar dados por altura e k
        for hi, hval in enumerate(axis_H):
            hval_float = float(hval)
            data[hval_float] = {}
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
                data[hval_float][float(kval)] = vals

    return data, axis_H, axis_K

# Carregar dados
def main():
    artificial_data, artificial_H, artificial_K = load_data_from_h5(ARTIFICIAL_FILE)
    experimental_data, experimental_H, experimental_K = load_data_from_h5(EXPERIMENTAL_FILE)

    if artificial_data is None or experimental_data is None:
        print('Erro ao carregar dados, saindo.')
        raise SystemExit(1)

    # Coletar todos os ganhos
    all_gains = []
    all_K_matrices = []

    # Para cada altura no experimental, encontrar altura e k mais próximos nos artificiais
    for h_exp in experimental_data:
        # Correspondência de altura mais próxima
        h_match = min(artificial_data.keys(), key=lambda h: abs(h - h_exp))
        print(f'Processando altura experimental {h_exp} (match artificial {h_match})')

        k_exp_values = list(experimental_data[h_exp].keys())
        k_art_values = list(artificial_data[h_match].keys())

        for k_exp in k_exp_values:
            # Correspondência do k mais próximo
            k_match = min(k_art_values, key=lambda k: abs(k - k_exp))
            if abs(k_match - k_exp) > 1e-6:
                print(f'  k experimental {k_exp} mapeado para k artificial {k_match}')

            vals_exp = experimental_data[h_exp][k_exp]
            vals_art = artificial_data[h_match][k_match]

            # Aplicar transformação linear para equalizar escalas
            M_hat_transformed = vals_art * 500000  # Baseado na análise de data_compare.py
            M = vals_exp.reshape(4, 4)
            M_hat = M_hat_transformed.reshape(4, 4)

            # Calcular ganhos: K = M / \hat{M}
            # Evitar divisão por zero
            with np.errstate(divide='ignore', invalid='ignore'):
                K = M_hat / M
                K = np.where(M != 0, K, 0)  # Set to 0 where M_hat is 0

            # Flatten e adicionar
            all_gains.extend(K.flatten())
            all_K_matrices.append(K)

            # Log: mostrar matriz K para algumas
            if len(all_K_matrices) < 10:  # Mostrar primeiras 10 matrizes
                print(f'  Matriz K para H_exp={h_exp}, H_art={h_match}, K_exp={k_exp}, K_art={k_match}:')
                print(K)

    # Calcular média e desvio padrão
    if all_gains:
        gains_array = np.array(all_gains)
        mean_gain = np.mean(gains_array)
        std_gain = np.std(gains_array)
        print(f'\nMédia global dos ganhos: {mean_gain:.6f}')
        print(f'Desvio padrão global dos ganhos: {std_gain:.6f}')
        print(f'Total de ganhos coletados: {len(all_gains)}')

        # Calcular médias e desvios por posição da matriz 4x4
        K_array = np.array(all_K_matrices)  # shape (num_samples, 4, 4)
        mean_matrix = np.mean(K_array, axis=0)
        std_matrix = np.std(K_array, axis=0)

        print('\nMatriz de médias por posição (4x4):')
        print(mean_matrix)
        print('\nMatriz de desvios padrão por posição (4x4):')
        print(std_matrix)
    else:
        print('Nenhum ganho coletado.')

    print('Done')

if __name__ == '__main__':
    main()