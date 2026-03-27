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
import matplotlib.pyplot as plt

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

# Gerar heatmaps
def create_heatmap_data(data_dict):
    """
    Reorganiza data_dict em uma matriz para heatmap.
    Linhas: 16 posições
    Colunas: alturas
    Para cada altura, faz a média entre todos os valores de K
    """
    heights = sorted(data_dict.keys())
    heatmap_data = []
    
    for pos in range(16):
        row = []
        for h in heights:
            # Coletar todos os valores na posição 'pos' para esta altura
            values_at_pos = []
            for k_vals in data_dict[h].values():
                if len(k_vals) > pos:
                    values_at_pos.append(k_vals[pos])
            if values_at_pos:
                row.append(np.mean(values_at_pos))
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    return np.array(heatmap_data), heights

print('\nGerando heatmaps...')

# Gerar dados dos heatmaps
art_heatmap, art_heights = create_heatmap_data(artificial_data)
exp_heatmap, exp_heights = create_heatmap_data(experimental_data)

# Normalizar intensidade do dataset experimental
exp_heatmap = exp_heatmap / 500000

# Criar figura com dois heatmaps lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Heatmap artificial - exibir labels a cada 5 medidas
im1 = ax1.imshow(art_heatmap, aspect='auto', cmap='coolwarm', interpolation='nearest')
ax1.set_xlabel('Altura (Height)', fontsize=12)
ax1.set_ylabel('Posição (0-15)', fontsize=12)
ax1.set_title('Dataset Artificial', fontsize=14, fontweight='bold')
xticks_art = range(0, len(art_heights), 5)
ax1.set_xticks(xticks_art)
ax1.set_xticklabels([f'{art_heights[i]:.1f}' for i in xticks_art], rotation=45)
ax1.set_yticks(range(16))
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Intensidade', fontsize=10)

# Heatmap experimental - exibir labels a cada 5 medidas
im2 = ax2.imshow(exp_heatmap, aspect='auto', cmap='coolwarm', interpolation='nearest')
ax2.set_xlabel('Altura (Height)', fontsize=12)
ax2.set_ylabel('Posição (0-15)', fontsize=12)
ax2.set_title('Dataset Experimental', fontsize=14, fontweight='bold')
xticks_exp = range(0, len(exp_heights), 5)
ax2.set_xticks(xticks_exp)
ax2.set_xticklabels([f'{exp_heights[i]:.1f}' for i in xticks_exp], rotation=45)
ax2.set_yticks(range(16))
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Intensidade', fontsize=10)

plt.tight_layout()
plt.savefig('heatmaps_comparison.png', dpi=150, bbox_inches='tight')
print('Heatmaps salvos em: heatmaps_comparison.png')
plt.show()

print('Done')