"""
pipeline_height_filter_validation.py

Script para validar o pipeline de processamento de dados experimentais usando denoising autoencoder e KNN.

Recebe como parâmetros:
- knn_model_path: Caminho para o modelo KNN (não usado diretamente, mas para consistência)
- dae_model_path: Caminho para o modelo do denoising autoencoder (.pt)
- experimental_file: Caminho para o arquivo experimental HDF5
- dae_checkpoint: Caminho para o checkpoint do KNN (contém scaler e modelo KNN)

O script:
1. Lê dados experimentais, filtra alturas MIN_HEIGHT <= hval <= MAX_HEIGHT
2. Escala os dados por 500000
3. Carrega o denoising autoencoder
4. Carrega o scaler do checkpoint do KNN
5. Escala os dados de entrada com o scaler
6. Passa os dados pelo DAE
7. Escala a saída do DAE com o scaler do KNN
8. Faz previsão com o KNN
9. Imprime MAE e MSE no terminal

Uso: python pipeline_height_filter_validation.py --knn_model_path <path> --dae_model_path <path> --experimental_file <path> --dae_checkpoint <path>
"""

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from error_estimation_height_filter import MIN_HEIGHT, MAX_HEIGHT

# Fator de escala
SCALE_FACTOR = 500000

def extract_dims_from_checkpoint(checkpoint_path):
    """
    Extrai latent_dim e hidden_dim do nome do arquivo de checkpoint.
    
    Formato esperado: dae_checkpoint_<...>_<latent_dim>_<hidden_dim>.joblib
    """
    filename = Path(checkpoint_path).stem  # Remove a extensão .joblib
    parts = filename.split('_')
    print(f"Extraindo dimensões do checkpoint: filename='{filename}', parts={parts}")
    if len(parts) < 7:
        print(f"Nome do checkpoint '{checkpoint_path}' não segue o formato esperado. Usando valores padrão latent_dim=8, hidden_dim=32.")
        return 8, 32
    latent_dim = int(parts[-2])
    hidden_dim = int(parts[-1])
    return latent_dim, hidden_dim

from denoising_autoencoder import DenoisingAutoencoder, split_by_original

def count_chunk_hits(y_true, y_pred, threshold=2.0):
    """Retorna acertos e erros em chunks de tamanho 2 baseado na diferença absoluta."""
    hits, misses = 0, 0
    rows = []
    n = len(y_true)
    for i in range(0, n, 2):
        true_chunk = np.array(y_true[i:i+2])
        pred_chunk = np.array(y_pred[i:i+2])
        diff = np.abs(true_chunk - pred_chunk)
        chunk_hits = np.sum(diff < threshold)
        chunk_misses = np.sum(diff >= threshold)
        hits += int(chunk_hits)
        misses += int(chunk_misses)
        rows.append({
            'chunk': i // 2,
            'n_samples': len(true_chunk),
            'hits': int(chunk_hits),
            'misses': int(chunk_misses)
        })
    return hits, misses, pd.DataFrame(rows)

import seaborn as sns

def plot_height_predictions(height, y_true, pred_knn, pred_knn_dae, output='height_predictions.png'):
    # plt.figure(figsize=(10, 5))
    # plt.hist(pred_knn_dae, bins=30, alpha=0.7, color='blue', label='Preditas (Denoised)')
    # plt.hist(height, bins=30, alpha=0.6, color='red', label='Distribuição esperada (Reais)')
    # plt.hist(pred_knn, bins=30, alpha=0.5, color='orange', label='Preditas (KNN)')
    # plt.xlabel('Altura Predita (mm)')
    # plt.ylabel('Frequência')
    # plt.title('Distribuição das Alturas Preditas (Modelo Limpo)')
    # plt.legend()
    # plt.show()
    # plt.tight_layout()
    # plt.savefig(output, dpi=150)
    import seaborn as sns

    plt.figure(figsize=(10, 5))
    sns.kdeplot(height, label='Real', color='red')
    sns.kdeplot(pred_knn, label='KNN', color='orange')
    sns.kdeplot(pred_knn_dae, label='Denoised', color='blue')

    plt.xlabel('Altura (mm)')
    plt.ylabel('Densidade')
    plt.title('Distribuição das Alturas')
    plt.legend()
    plt.show()
    plt.savefig('kdeplot_' + output, dpi=150)

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axs[0].hist(height, bins=30, color='red')
    axs[0].set_title('Distribuição Real das Alturas de validação')

    axs[1].hist(pred_knn, bins=30, color='orange')
    axs[1].set_title('Preditas (KNN)')

    axs[2].hist(pred_knn_dae, bins=30, color='blue')
    axs[2].set_title('Preditas (Denoised)')

    for ax in axs:
        ax.set_ylabel('Frequência')

    axs[-1].set_xlabel('Altura (mm)')
    plt.tight_layout()
    plt.show()
    plt.legend()
    plt.savefig('histogram_' + output, dpi=150)


def load_experimental_data(fname):
    """Carrega dados experimentais e retorna lista de (altura, valores_16)."""
    file_path = Path(fname)
    data = []
    with h5py.File(file_path, 'r') as f:
        target = [None]
        def finder(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith('N_compound_maxenv'):
                target[0] = name
        f.visititems(finder)
        if target[0] is None:
            raise ValueError(f"Dataset não encontrado em {fname}")
        
        d = f[target[0]]
        axis_H = np.asanyarray(d.attrs.get('axis_H', np.arange(d.shape[0])))
        
        for hi, hval in enumerate(axis_H):
            if MIN_HEIGHT <= hval <= MAX_HEIGHT:
                for ki in range(d.shape[2]):  # Geralmente 1
                    vals = d[hi, 0, ki, 0, :]
                    vals_scaled = np.asanyarray(vals).ravel() / SCALE_FACTOR
                    data.append((float(hval), vals_scaled))
    return data

def main():
    parser = argparse.ArgumentParser(description='Validação do pipeline com denoising autoencoder e KNN.')
    parser.add_argument('--knn_model_path', type=str, required=True, help='Caminho para o modelo KNN (checkpoint joblib)')
    parser.add_argument('--dae_model_path', type=str, required=True, help='Caminho para o modelo do denoising autoencoder (.pt)')
    parser.add_argument('--experimental_file', type=str, required=True, help='Caminho para o arquivo experimental HDF5')
    parser.add_argument('--dae_checkpoint', type=str, required=True, help='Caminho para o checkpoint do KNN (contém scaler e modelo)')
    parser.add_argument('--art_file', type=str, default=None, help='Caminho para o arquivo artificial HDF5 (opcional)')

    args = parser.parse_args()

    # Carregar dados experimentais
    experimental_data = load_experimental_data(args.experimental_file)
    print(f"Dados experimentais carregados: {len(experimental_data)} amostras")

    # Preparar arrays
    heights = np.array([h for h, _ in experimental_data])
    features_scaled = np.array([v for _, v in experimental_data])
    print(f"Shapes: heights {heights.shape}, features {features_scaled.shape}")

    # Carregar denoising autoencoder
    # Extrair dimensões do checkpoint
    latent_dim, hidden_dim = extract_dims_from_checkpoint(args.dae_checkpoint)
    print(f"Dimensões extraídas do checkpoint: latent_dim={latent_dim}, hidden_dim={hidden_dim}")
    dae_model = DenoisingAutoencoder(input_dim=16, latent_dim=latent_dim, hidden_dim=hidden_dim)

    dae_model.load_state_dict(torch.load(args.dae_model_path, map_location='cpu'))
    dae_model.to('cpu')
    dae_model.eval()

 

    # Carregar scaler do DAE
    checkpoint = joblib.load(args.dae_checkpoint)
    scaler = checkpoint['scaler']
    knn_checkpoint = joblib.load(args.knn_model_path)
    knn_model = knn_checkpoint['knn']['KNN']
    knn_scaler = knn_checkpoint['scaler']

    print(f"Scaler carregado - mean (primeiras 4): {scaler.mean_[:4]}")

    # Normalizar os dados com o scaler antes de passar para o autoencoder
    features_scaled_normalized = scaler.transform(features_scaled)

    # Processar dados com autoencoder
    features_tensor = torch.tensor(features_scaled_normalized, dtype=torch.float32)
    with torch.no_grad():
        features_denoised_normalized = dae_model(features_tensor).numpy()

    # Desnormalizar as saídas do autoencoder
    features_denoised = scaler.inverse_transform(features_denoised_normalized)

    # Escalar saída do DAE com scaler do KNN (já desnormalizada, mas precisa normalizar novamente para KNN)
    features_denoised_scaled = knn_scaler.transform(features_denoised)

    # Predizer alturas com KNN
    predicted_heights = knn_model.predict(features_denoised_scaled)

    # Calcular métricas
    mae = mean_absolute_error(heights, predicted_heights)
    mse = mean_squared_error(heights, predicted_heights)

    print(f"Métricas para o ensaio:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

    print("\nMétricas para KNN direto nos dados de entrada")
    predicted_heights_direct = knn_model.predict(knn_scaler.transform(features_scaled))
    mae_direct = mean_absolute_error(heights, predicted_heights_direct)
    mse_direct = mean_squared_error(heights, predicted_heights_direct)
    print(f"MAE direto: {mae_direct:.4f}")
    print(f"MSE direto: {mse_direct:.4f}")
    plot_height_predictions(heights, heights, predicted_heights_direct, predicted_heights, output='experimental_KNN_only_height_predictions.png')
   
    hits_direct, misses_direct, df_direct = count_chunk_hits(heights, predicted_heights_direct, threshold=2.0)
    hits_dae, misses_dae, df_dae = count_chunk_hits(heights, predicted_heights, threshold=2.0)

    print('\nTabela de acertos (KNN direto) - chunk de 2:')
    print(df_direct.head())
    print(f'KNN direto hits <2: {hits_direct}, misses >=2: {misses_direct}, total={len(heights)}')
    print('\nTabela de acertos (KNN + DAE) - chunk de 2:')
    print(df_dae.head())
    print(f'KNN + DAE hits <2: {hits_dae}, misses >=2: {misses_dae}, total={len(heights)}')

    print("Dados para o KNN direto nos dados artificiais gerados (se fornecido):")
    if args.art_file:
        from denoising_autoencoder import load_paired_data

        X_noisy, X_clean, y_noisy, X_orig, y_orig, n_art = load_paired_data(Path(args.art_file))
         # Split e validação (artificial) usando split_by_original, conforme solicitado
        N_orig = 195
        n_art = 500
        test_frac = 0.3
        val_frac = 0.3
        train_idx, val_idx, test_idx, _ = split_by_original(N_orig=N_orig, n_art=n_art, test_frac=test_frac, val_frac=val_frac, random_state=42)

        print('\nSplit artificial realizado:')
        print(f'  N_orig={N_orig}, n_art={n_art}, test_frac={test_frac}, val_frac={val_frac}, random_state=42')
        print(f'  Índices: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

        art_heights = y_noisy[val_idx]
        art_features = X_noisy[val_idx]
        art_features_scaled = knn_scaler.transform(art_features)
        art_predicted_heights = knn_model.predict(art_features_scaled)
        art_mae = mean_absolute_error(art_heights, art_predicted_heights)
        art_mse = mean_squared_error(art_heights, art_predicted_heights)
        print(f"MAE (artificial): {art_mae:.4f}")
        print(f"MSE (artificial): {art_mse:.4f}")
        print()

        # Caso passasse pelo DAE
        art_features_scaled_normalized = scaler.transform(art_features)
        art_features_tensor = torch.tensor(art_features_scaled_normalized, dtype=torch.float32)
        with torch.no_grad():
            art_features_denoised_normalized = dae_model(art_features_tensor).numpy()
        art_features_denoised = scaler.inverse_transform(art_features_denoised_normalized)
        art_features_denoised_scaled = knn_scaler.transform(art_features_denoised)
        art_predicted_heights_dae = knn_model.predict(art_features_denoised_scaled)
        art_mae_dae = mean_absolute_error(art_heights, art_predicted_heights_dae)
        art_mse_dae = mean_squared_error(art_heights, art_predicted_heights_dae)
        print(f"MAE (artificial + DAE): {art_mae_dae:.4f}")
        print(f"MSE (artificial + DAE): {art_mse_dae:.4f}")

        # Avaliação no conjunto de validação artificial obtido via split_by_original
        art_val_heights = art_heights
        art_val_features = art_features

        art_val_pred_direct = knn_model.predict(knn_scaler.transform(art_val_features))
        art_val_features_scaled_normalized = scaler.transform(art_val_features)
        art_val_features_tensor = torch.tensor(art_val_features_scaled_normalized, dtype=torch.float32)
        with torch.no_grad():
            art_val_denoised_normalized = dae_model(art_val_features_tensor).numpy()
        art_val_denoised = scaler.inverse_transform(art_val_denoised_normalized)
        art_val_denoised_scaled = knn_scaler.transform(art_val_denoised)
        art_val_pred_dae = knn_model.predict(art_val_denoised_scaled)

        print('\nMétricas de validação (artificial val_idx):')
        print('KNN direto')
        print('  MAE', mean_absolute_error(art_val_heights, art_val_pred_direct))
        print('  MSE', mean_squared_error(art_val_heights, art_val_pred_direct))
        print('KNN + DAE')
        print('  MAE', mean_absolute_error(art_val_heights, art_val_pred_dae))
        print('  MSE', mean_squared_error(art_val_heights, art_val_pred_dae))

        plot_height_predictions(art_val_heights, art_val_heights, art_val_pred_direct, art_val_pred_dae,
                                output='artificial_validation_height_predictions.png')

        hits_direct, misses_direct, df_direct = count_chunk_hits(art_val_heights, art_val_pred_direct, threshold=2.0)
        hits_dae, misses_dae, df_dae = count_chunk_hits(art_val_heights, art_val_pred_dae, threshold=2.0)

        print('\nTabela de acertos (KNN direto) - chunk de 2:')
        print(df_direct.head())
        print(f'KNN direto hits <2: {hits_direct}, misses >=2: {misses_direct}, total={len(art_val_heights)}')

        print('\nTabela de acertos (KNN + DAE) - chunk de 2:')
        print(df_dae.head())
        print(f'KNN + DAE hits <2: {hits_dae}, misses >=2: {misses_dae}, total={len(art_val_heights)}')

if __name__ == '__main__':
    main()