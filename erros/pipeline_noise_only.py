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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from error_estimation_height_filter import MIN_HEIGHT, MAX_HEIGHT

GAIN_MEAN_MATRIX = np.array([
    [0.73799924, 1.28409673, 1.29267225, 0.96172604],
    [1.49915294, 0.83210729, 1.94994793, 0.73006332],
    [1.62797488, 2.59592461, 1.11130719, 1.42609107],
    [1.54227595, 1.68133359, 2.57544559, 1.31702773],
], dtype=np.float64)

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

from denoising_autoencoder import DenoisingAutoencoder

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
                    # for i in range(vals_scaled.size):
                    #     vals_scaled[i] /= GAIN_MEAN_MATRIX[i // 4, i % 4]

                    data.append((float(hval), vals_scaled))
    return data

def main():
    parser = argparse.ArgumentParser(description='Validação do pipeline com denoising autoencoder e KNN.')
    parser.add_argument('--knn_model_path', type=str, required=True, help='Caminho para o modelo KNN (checkpoint joblib)')
    parser.add_argument('--dae_model_path', type=str, required=True, help='Caminho para o modelo do denoising autoencoder (.pt)')
    parser.add_argument('--experimental_file', type=str, required=True, help='Caminho para o arquivo experimental HDF5')
    parser.add_argument('--dae_checkpoint', type=str, required=True, help='Caminho para o checkpoint do KNN (contém scaler e modelo)')

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

if __name__ == '__main__':
    main()