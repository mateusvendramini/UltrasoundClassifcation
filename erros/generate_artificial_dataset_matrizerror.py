"""
generate_artificial_dataset.py

Lê os datasets N_compound_maxenv dos arquivos HDF5 especificados,
e para cada ponto de medição gera N_ARTIFICIAL pontos sintéticos
multiplicando element-wise por uma matriz de ganhos aleatória amostrada
de uma distribuição configurável (padrão: normal com média=1, std=0.002).

O dataset artificial é salvo como artificial_<timestamp>.h5 com:
  - measurements:          shape (N_original * N_ARTIFICIAL, 16)  — dados ruidosos
  - heights:               shape (N_original * N_ARTIFICIAL,)      — rótulos de altura
  - original_measurements: shape (N_original, 16)                  — pontos originais
  - original_heights:      shape (N_original,)                     — alturas originais

Uso: python generate_artificial_dataset.py
"""

import h5py
from pathlib import Path
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

FILES = [
    'model_compound_4_10_104_1_interface_synthetic_apod.h5'
]

DATA_DIR = Path('.')

# Número de pontos artificiais por ponto original
N_ARTIFICIAL = 50

# Parâmetros da distribuição de ganhos (global)
GAIN_DISTRIBUTION = 'normal'  # opções: 'normal', 'uniform', 'lognormal'
GAIN_MEAN = 1.0
GAIN_STD = 0.1

# Parâmetros de erro por posição 4x4 (substitui GAIN_MEAN/GAIN_STD quando configurado)
# Ajuste com os valores de média/desvio obtidos em error_estimation.py.
#GAIN_MEAN_MATRIX = np.full((4, 4), 1.0, dtype=np.float64)
#GAIN_STD_MATRIX = np.full((4, 4), 0.1, dtype=np.float64)
GAIN_MEAN_MATRIX = np.array([
    [1.37979424, 0.91479653, 0.84348735, 1.71868952],
    [0.78176418, 1.2936139 , 0.59382913, 1.72507647],
    [0.7323506 , 0.42288392, 0.99975724, 0.82494157],
    [1.00479022, 0.68103193, 0.41581674, 0.82574173],
], dtype=np.float64)

GAIN_STD_MATRIX = np.array([
    [0.16720413, 0.40075454, 0.30979591, 1.22416829],
    [0.52589639, 0.31601479, 0.37821625, 1.0717828 ],
    [0.34299812, 0.14260583, 0.40099375, 0.36728759],
    [0.56270319, 0.26784722, 0.15300275, 0.22316433],
], dtype=np.float64)

#Média global dos ganhos: 0.947398
#Desvio padrão global dos ganhos: 0.648466
#Total de ganhos coletados: 1440

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def load_original_data(files, data_dir):
    """Lê todos os pontos (16 medidas + altura) dos arquivos HDF5."""
    vals_list = []
    heights_list = []

    for fname in files:
        file_path = data_dir / fname
        print(f'Processando: {file_path}')

        if not file_path.exists():
            print(f'  Arquivo não encontrado, ignorando.')
            continue

        with h5py.File(file_path, 'r') as f:
            # Localiza o dataset pelo sufixo
            target = [None]

            def finder(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith('N_compound_maxenv'):
                    target[0] = name

            f.visititems(finder)

            if target[0] is None:
                print(f'  N_compound_maxenv não encontrado em {fname}, ignorando.')
                continue

            d = f[target[0]]
            shape = d.shape
            print(f'  Encontrado {target[0]}, shape={shape}')

            axis_H = np.asanyarray(d.attrs.get('axis_H', np.arange(shape[0])))
            axis_K = np.asanyarray(d.attrs.get('axis_K', np.arange(shape[2])))

            count_before = len(vals_list)
            for hi, hval in enumerate(axis_H):
                for ki, _ in enumerate(axis_K):
                    try:
                        vals = d[hi, 0, ki, 0, :]
                    except Exception as e:
                        print(f'  Ignorando H={hi} K={ki}: {e}')
                        continue

                    vals = np.asanyarray(vals).ravel()
                    if vals.size != 16:
                        print(
                            f'  Aviso: esperado 16 valores, obtido {vals.size} '
                            f'em H={hi} K={ki}, ignorando.'
                        )
                        continue

                    vals_list.append(vals.astype(np.float64))
                    heights_list.append(float(hval))

            added = len(vals_list) - count_before
            print(f'  {added} pontos lidos.')

    if not vals_list:
        return np.empty((0, 16), dtype=np.float64), np.empty(0, dtype=np.float64)

    return np.array(vals_list, dtype=np.float64), np.array(heights_list, dtype=np.float64)


def sample_gains(rng, size, distribution, mean, std):
    """
    Amostra um vetor de ganhos de tamanho `size` a partir da distribuição
    especificada, usando parâmetros globais (escalar) ou por posição 4x4.

    Parâmetros
    ----------
    distribution : str
        'normal'    — N(mean, std)
        'uniform'   — U(mean - std, mean + std)
        'lognormal' — log-normal parametrizada para ter média `mean` e dp `std`
    mean : float or array-like (4x4 / 16)
    std : float or array-like (4x4 / 16)

    Retorna
    -------
    ndarray shape (size,)
    """
    if np.isscalar(mean) and np.isscalar(std):
        mean_arr = np.full(size, float(mean), dtype=np.float64)
        std_arr = np.full(size, float(std), dtype=np.float64)
    else:
        mean_arr = np.asarray(mean, dtype=np.float64)
        std_arr = np.asarray(std, dtype=np.float64)

        if mean_arr.shape == (4, 4):
            mean_arr = mean_arr.ravel()
        if std_arr.shape == (4, 4):
            std_arr = std_arr.ravel()

        if mean_arr.ndim != 1 or std_arr.ndim != 1:
            raise ValueError('mean/std devem ser escalar, 1D com 16 elementos ou 2D 4x4.')

        if mean_arr.shape[0] != size or std_arr.shape[0] != size:
            raise ValueError(
                f'O tamanho de mean/std ({mean_arr.shape[0]}, {std_arr.shape[0]}) '
                f'deve corresponder a size={size}.'
            )

    if np.any(std_arr < 0):
        raise ValueError('std deve ser não negativo para todos os elementos.')

    if distribution == 'normal':
        return rng.normal(mean_arr, std_arr)

    elif distribution == 'uniform':
        low = mean_arr - std_arr
        high = mean_arr + std_arr
        return rng.uniform(low, high)

    elif distribution == 'lognormal':
        if np.any(mean_arr <= 0):
            raise ValueError('mean deve ser > 0 para lognormal')
        sigma2 = np.log(1.0 + (std_arr / mean_arr) ** 2)
        mu = np.log(mean_arr) - 0.5 * sigma2
        return rng.lognormal(mu, np.sqrt(sigma2))

    else:
        raise ValueError(f'Distribuição desconhecida: {distribution!r}. '
                         f'Use: normal, uniform, lognormal.')


def generate_artificial(vals_arr, heights_arr, n_artificial,
                         distribution, gain_mean, gain_std, rng):
    """
    Para cada um dos N pontos originais gera `n_artificial` cópias ruidosas
    multiplicando element-wise por ganhos aleatórios.

    Retorna
    -------
    art_vals   : ndarray shape (N * n_artificial, 16)
    art_heights: ndarray shape (N * n_artificial,)
    """
    N, n_vals = vals_arr.shape
    art_vals = np.empty((N * n_artificial, n_vals), dtype=np.float64)
    art_heights = np.empty(N * n_artificial, dtype=np.float64)

    for i in range(N):
        base = i * n_artificial
        for j in range(n_artificial):
            gains = sample_gains(rng, n_vals, distribution, gain_mean, gain_std)
            art_vals[base + j] = vals_arr[i] * gains
            art_heights[base + j] = heights_arr[i]

    return art_vals, art_heights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng()  # semente aleatória nova a cada execução

    print('=' * 60)
    print('Geração de Dataset Artificial')
    print('=' * 60)
    print(f'Distribuição de ganhos : {GAIN_DISTRIBUTION}')

    if GAIN_MEAN_MATRIX is not None and GAIN_STD_MATRIX is not None:
        print('Usando matriz de ganhos 4x4 (mean/std por posição).')
        print('GAIN_MEAN_MATRIX:')
        print(GAIN_MEAN_MATRIX)
        print('GAIN_STD_MATRIX:')
        print(GAIN_STD_MATRIX)
        mean_param = GAIN_MEAN_MATRIX
        std_param = GAIN_STD_MATRIX
    else:
        print(f'Média dos ganhos       : {GAIN_MEAN}')
        print(f'Desvio padrão dos ganhos: {GAIN_STD}')
        mean_param = GAIN_MEAN
        std_param = GAIN_STD

    print(f'Pontos artificiais/original: {N_ARTIFICIAL}')
    print()

    print('Carregando dados originais...')
    vals_arr, heights_arr = load_original_data(FILES, DATA_DIR)

    if vals_arr.shape[0] == 0:
        print('\nNenhum dado carregado. Encerrando.')
        raise SystemExit(1)

    unique_h = np.unique(heights_arr)
    print(f'\nDataset original: {vals_arr.shape[0]} pontos')
    print(f'Alturas únicas ({len(unique_h)}): {unique_h}')

    total_artificial = vals_arr.shape[0] * N_ARTIFICIAL
    print(f'\nGerando {N_ARTIFICIAL} pontos artificiais por ponto original '
          f'({total_artificial} no total)...')

    art_vals, art_heights = generate_artificial(
        vals_arr, heights_arr,
        N_ARTIFICIAL,
        GAIN_DISTRIBUTION, mean_param, std_param,
        rng,
    )

    print(f'Dataset artificial gerado: {art_vals.shape[0]} pontos')

    # ------------------------------------------------------------------
    # Salvar em HDF5
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = DATA_DIR / f'artificial_{timestamp}.h5'

    with h5py.File(out_path, 'w') as f:
        f.attrs['description'] = (
            'Dataset artificial gerado a partir de medições de ultrassom. '
            'Cada ponto original foi perturbado por ganhos aleatórios element-wise.'
        )
        f.attrs['source_files'] = ', '.join(FILES)
        f.attrs['n_artificial_per_original'] = N_ARTIFICIAL
        f.attrs['gain_distribution'] = GAIN_DISTRIBUTION
        f.attrs['gain_mean'] = GAIN_MEAN
        f.attrs['gain_std'] = GAIN_STD
        f.attrs['timestamp'] = timestamp
        f.attrs['n_original'] = vals_arr.shape[0]
        f.attrs['n_artificial_total'] = art_vals.shape[0]

        f.create_dataset('measurements', data=art_vals,
                         compression='gzip', compression_opts=4)
        f.create_dataset('heights', data=art_heights,
                         compression='gzip', compression_opts=4)
        f.create_dataset('original_measurements', data=vals_arr,
                         compression='gzip', compression_opts=4)
        f.create_dataset('original_heights', data=heights_arr,
                         compression='gzip', compression_opts=4)

    print(f'\nDataset salvo: {out_path}')
    print(f'  measurements shape : {art_vals.shape}')
    print(f'  heights shape      : {art_heights.shape}')

    # ------------------------------------------------------------------
    # Estatísticas de distância original x artificial
    # ------------------------------------------------------------------
    orig_repeated = np.repeat(vals_arr, N_ARTIFICIAL, axis=0)
    diffs = art_vals - orig_repeated
    distances = np.linalg.norm(diffs, axis=1)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    print('\nEstatísticas de distância artificial -> original:')
    print(f'  média da distância (Euclidiana)   : {mean_distance:.6f}')
    print(f'  desvio padrão da distância        : {std_distance:.6f}')
    print(f'  Salvo para o arquivo : {out_path}')

    # salvar no HDF5 também para rastreabilidade
    with h5py.File(out_path, 'a') as f:
        f.attrs['mean_distance'] = float(mean_distance)
        f.attrs['std_distance'] = float(std_distance)

    print('\nConcluído.')


if __name__ == '__main__':
    main()
