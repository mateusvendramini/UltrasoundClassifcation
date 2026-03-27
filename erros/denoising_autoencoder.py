"""
denoising_autoencoder.py

Treina um Denoising Autoencoder (DAE) em PyTorch para reconstruir medições
de ultrassom a partir de versões ruidosas geradas por perturbação multiplicativa.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Correspondência noisy ↔ clean (gerada por generate_artificial_dataset.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  generate_artificial_dataset.py itera sobre os N=606 pontos originais
  e gera N_ARTIFICIAL=5 cópias ruidosas para cada um:

      for i in range(N):          # índice original
          for j in range(N_ART):  # j = 0..4
              gains = N(1, 0.002) # vetor de 16 ganhos
              art_vals[i*N_ART + j] = original_vals[i] * gains

  Portanto, para o ponto artificial de índice k:
      noisy  = measurements[k]                      # shape (16,)
      clean  = original_measurements[k // N_ART]    # shape (16,)

  Esta correspondência é usada diretamente como par (entrada, alvo) no DAE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Arquitetura do DAE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Encoder: R^16 → R^32 → R^LATENT_DIM
  Decoder: R^LATENT_DIM → R^32 → R^16

  Saída = x̃ + decoder(encoder(x̃))   ← conexão residual
  O modelo aprende a correção do ruído, não o sinal completo.
  Isso é mais eficiente quando o ruído é pequeno (σ ≈ 0.002).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Divisão treino/val/teste
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A divisão é feita sobre os índices ORIGINAIS (606 pontos) para evitar
  vazamento de dados: um mesmo ponto original não pode ter cópias ruidosas
  simultaneamente em treino e teste.

  Depois de dividir os índices originais, expandimos para os índices ruidosos:
      orig_idx i → noisy_idx [i*N_ART, i*N_ART+1, ..., i*N_ART+N_ART-1]

Uso: python denoising_autoencoder.py
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

ART_FILE      = Path('artificial_20260326_235941.h5')
CHECKPOINT_OUT = Path('dae_checkpoint_matrizerror_bigger_N_100.joblib')
MODEL_OUT      = Path('dae_best_262_matrizerror_bigger_N_100.pt')

# Arquitetura
LATENT_DIM = 12

# Treino
BATCH_SIZE  = 64
EPOCHS      = 600 #300
LR          = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE    = 60        # early stopping

# Divisão (sobre índices originais)
TEST_FRAC = 0.15
VAL_FRAC  = 0.15

RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# 1. Carregamento dos dados e construção dos pares (noisy, clean)
# ---------------------------------------------------------------------------

def load_paired_data(art_file: Path):
    """
    Lê o arquivo HDF5 e retorna pares (X_noisy, X_clean) alinhados.

    Para cada índice k em X_noisy:
        X_noisy[k]  = measurements[k]
        X_clean[k]  = original_measurements[k // n_artificial_per_original]
    """
    with h5py.File(art_file, 'r') as f:
        X_noisy      = np.array(f['measurements'])           # (N*n_art, 16)
        y_noisy      = np.array(f['heights'])                # (N*n_art,)
        X_orig       = np.array(f['original_measurements'])  # (N, 16)
        y_orig       = np.array(f['original_heights'])       # (N,)
        n_art        = int(f.attrs['n_artificial_per_original'])
        gain_dist    = str(f.attrs['gain_distribution'])
        gain_std     = float(f.attrs['gain_std'])

    N_noisy = X_noisy.shape[0]
    N_orig  = X_orig.shape[0]

    # Correspondência: ponto ruidoso k → ponto original k // n_art
    orig_indices_for_noisy = np.arange(N_noisy) // n_art  # shape (N_noisy,)
    X_clean = X_orig[orig_indices_for_noisy]               # (N_noisy, 16)

    print(f'Arquivo: {art_file.name}')
    print(f'  Pontos originais   : {N_orig}')
    print(f'  Cópias ruidosas    : {N_noisy}  ({n_art} por original)')
    print(f'  Distribuição ruído : {gain_dist}  σ={gain_std}')
    print(f'  Razão média (art/orig): {(X_noisy / X_orig[orig_indices_for_noisy]).mean():.6f}')
    print()

    return X_noisy, X_clean, y_noisy, X_orig, y_orig, n_art


# ---------------------------------------------------------------------------
# 2. Divisão treino / val / teste (por índices originais)
# ---------------------------------------------------------------------------

def split_by_original(N_orig, n_art, test_frac, val_frac, random_state):
    """
    Divide os N_orig índices originais em treino/val/teste e
    expande para os índices ruidosos correspondentes.

    Garante que cópias do mesmo ponto original ficam sempre
    no mesmo subconjunto (sem data leakage).
    """
    orig_idx = np.arange(N_orig)

    train_orig, tmp_orig = train_test_split(
        orig_idx, test_size=test_frac + val_frac, random_state=random_state
    )
    val_frac_adj = val_frac / (test_frac + val_frac)
    val_orig, test_orig = train_test_split(
        tmp_orig, test_size=1 - val_frac_adj, random_state=random_state
    )

    def expand(idx):
        return np.array([i * n_art + j for i in idx for j in range(n_art)])

    train_noisy = expand(train_orig)
    val_noisy   = expand(val_orig)
    test_noisy  = expand(test_orig)

    print(f'Divisão por índices originais:')
    print(f'  treino : {len(train_orig)} orig  → {len(train_noisy)} ruidosos')
    print(f'  val    : {len(val_orig)}  orig  → {len(val_noisy)}  ruidosos')
    print(f'  teste  : {len(test_orig)}  orig  → {len(test_noisy)}  ruidosos')
    print()

    return train_noisy, val_noisy, test_noisy, train_orig


# ---------------------------------------------------------------------------
# 3. Normalização
# ---------------------------------------------------------------------------

def fit_scaler(X_clean_all, train_noisy_idx):
    """
    Ajusta o StandardScaler usando apenas os dados limpos de treino.
    Aplicado depois a todos os dados (noisy e clean).
    """
    # índices originais de treino = train_noisy_idx // n_art  (todos iguais dentro do grupo)
    # mas X_clean_all[train_noisy_idx] já tem os pares alinhados
    scaler = StandardScaler()
    scaler.fit(X_clean_all[train_noisy_idx])
    return scaler


# ---------------------------------------------------------------------------
# 4. Dataset PyTorch
# ---------------------------------------------------------------------------

class NoisyCleanDataset(Dataset):
    """
    Dataset de pares (medição ruidosa, medição limpa) já normalizados.
    """
    def __init__(self, X_noisy: np.ndarray, X_clean: np.ndarray):
        self.X_noisy = torch.tensor(X_noisy, dtype=torch.float32)
        self.X_clean = torch.tensor(X_clean, dtype=torch.float32)

    def __len__(self):
        return len(self.X_noisy)

    def __getitem__(self, idx):
        return self.X_noisy[idx], self.X_clean[idx]


# ---------------------------------------------------------------------------
# 5. Modelo: Denoising Autoencoder com conexão residual
# ---------------------------------------------------------------------------

class DenoisingAutoencoder(nn.Module):
    """
    Autoencoder denoising com conexão residual.

    Em vez de reconstruir o sinal completo, o modelo aprende a
    CORREÇÃO a aplicar sobre a entrada ruidosa:

        x̂ = x̃ + Decoder(Encoder(x̃))

    Isso é mais eficiente para ruídos pequenos (σ ≈ 0.002) porque
    o modelo só precisa estimar a perturbação, não o sinal inteiro.

    Parâmetros
    ----------
    input_dim  : dimensão das medições (16 canais)
    latent_dim : dimensão do espaço latente (espaço comprimido)
    """
    def __init__(self, input_dim: int = 16, latent_dim: int = 8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Linear(48, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Linear(48, input_dim),
        )

        # Inicializa os pesos do decoder próximos de zero para que
        # no início do treino a saída ≈ entrada (perturbação ≈ 0)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna x̃ + correção aprendida (conexão residual)."""
        correction = self.decode(self.encode(x))
        return x + correction


# ---------------------------------------------------------------------------
# 6. Loop de treino
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, epochs, lr, weight_decay, patience, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience // 3, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss  = float('inf')
    patience_count = 0
    history = {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        # --- treino ---
        model.train()
        total_loss = 0.0
        for x_noisy, x_clean in train_loader:
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)
            optimizer.zero_grad()
            x_hat = model(x_noisy)
            loss  = criterion(x_hat, x_clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_noisy.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # --- validação ---
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for x_noisy, x_clean in val_loader:
                x_noisy = x_noisy.to(device)
                x_clean = x_clean.to(device)
                x_hat = model(x_noisy)
                total_val += criterion(x_hat, x_clean).item() * x_noisy.size(0)
        val_loss = total_val / len(val_loader.dataset)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            patience_count += 1

        if epoch % 10 == 0 or patience_count == 0:
            print(f'Epoch {epoch:4d}/{epochs}  '
                  f'train={train_loss:.7f}  val={val_loss:.7f}'
                  + (' ← melhor' if patience_count == 0 else ''))

        if patience_count >= patience:
            print(f'\nEarly stopping na época {epoch} '
                  f'(melhor val_loss={best_val_loss:.7f})')
            break

    return history


# ---------------------------------------------------------------------------
# 7. Avaliação
# ---------------------------------------------------------------------------

def evaluate(model, X_noisy_sc, X_clean_sc, scaler, device):
    """
    Avalia o modelo no conjunto de teste.
    Retorna métricas no espaço original (antes de normalizar).
    """
    model.eval()
    with torch.no_grad():
        x_t   = torch.tensor(X_noisy_sc, dtype=torch.float32).to(device)
        x_hat = model(x_t).cpu().numpy()

    # De-normaliza para o espaço original
    X_noisy_orig   = scaler.inverse_transform(X_noisy_sc)
    X_clean_orig   = scaler.inverse_transform(X_clean_sc)
    X_denoised_orig = scaler.inverse_transform(x_hat)

    mae_noisy    = mean_absolute_error(X_clean_orig, X_noisy_orig)
    mae_denoised = mean_absolute_error(X_clean_orig, X_denoised_orig)
    mse_noisy    = mean_squared_error(X_clean_orig, X_noisy_orig)
    mse_denoised = mean_squared_error(X_clean_orig, X_denoised_orig)

    snr_noisy    = _snr(X_clean_orig, X_noisy_orig)
    snr_denoised = _snr(X_clean_orig, X_denoised_orig)

    print('=' * 55)
    print('Avaliação no conjunto de teste (espaço original)')
    print('=' * 55)
    print(f'{"Métrica":<20} {"Ruidoso":>12} {"Denoised":>12}')
    print('-' * 55)
    print(f'{"MAE":<20} {mae_noisy:>12.8f} {mae_denoised:>12.8f}')
    print(f'{"MSE":<20} {mse_noisy:>12.8f} {mse_denoised:>12.8f}')
    print(f'{"SNR (dB)":<20} {snr_noisy:>12.4f} {snr_denoised:>12.4f}')
    print('=' * 55)

    melhora_mae = (mae_noisy - mae_denoised) / mae_noisy * 100
    melhora_snr = snr_denoised - snr_noisy
    print(f'\nMelhora MAE : {melhora_mae:+.2f}%')
    print(f'Melhora SNR : {melhora_snr:+.4f} dB')

    return {
        'X_noisy_orig': X_noisy_orig,
        'X_clean_orig': X_clean_orig,
        'X_denoised_orig': X_denoised_orig,
        'mae_noisy': mae_noisy,
        'mae_denoised': mae_denoised,
        'mse_noisy': mse_noisy,
        'mse_denoised': mse_denoised,
        'snr_noisy': snr_noisy,
        'snr_denoised': snr_denoised,
    }


def _snr(signal, noisy_or_recon):
    """Signal-to-Noise Ratio em dB: 10 * log10(var(signal) / var(erro))"""
    erro = signal - noisy_or_recon
    var_s = np.var(signal)
    var_e = np.var(erro)
    if var_e == 0:
        return float('inf')
    return 10 * np.log10(var_s / var_e)


# ---------------------------------------------------------------------------
# 8. Visualizações
# ---------------------------------------------------------------------------

def plot_training_history(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history['train']) + 1)
    ax.semilogy(epochs, history['train'], label='treino', color='steelblue')
    ax.semilogy(epochs, history['val'],   label='validação', color='coral')
    ax.set_xlabel('Época')
    ax.set_ylabel('MSE Loss (log)')
    ax.set_title('Curva de Treinamento do DAE')
    ax.legend()
    plt.tight_layout()
    plt.savefig('dae_training_history.png', dpi=150)
    plt.show()
    print('Gráfico salvo: dae_training_history.png')


def plot_reconstruction(results, n_samples=6, channel=0):
    """Compara canal `channel` entre ruidoso, limpo e denoised."""
    X_n = results['X_noisy_orig'][:n_samples]
    X_c = results['X_clean_orig'][:n_samples]
    X_d = results['X_denoised_orig'][:n_samples]

    fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3), sharey=True)
    for ax, xn, xc, xd, idx in zip(axes, X_n, X_c, X_d, range(n_samples)):
        ax.bar(['ruidoso', 'limpo', 'denoised'],
               [xn[channel], xc[channel], xd[channel]],
               color=['coral', 'steelblue', 'mediumseagreen'])
        ax.set_title(f'Amostra {idx}')
        ax.set_ylabel(f'Canal {channel}' if idx == 0 else '')
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle(f'Reconstrução — Canal {channel}')
    plt.tight_layout()
    plt.savefig('dae_reconstruction_samples.png', dpi=150)
    plt.show()
    print('Gráfico salvo: dae_reconstruction_samples.png')


def plot_error_distribution(results):
    """Histograma do erro por canal: ruidoso vs denoised."""
    err_noisy    = results['X_noisy_orig']    - results['X_clean_orig']
    err_denoised = results['X_denoised_orig'] - results['X_clean_orig']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(err_noisy.ravel(),    bins=60, alpha=0.7, color='coral',
            label=f'ruidoso  MAE={results["mae_noisy"]:.6f}', density=True)
    ax.hist(err_denoised.ravel(), bins=60, alpha=0.7, color='mediumseagreen',
            label=f'denoised MAE={results["mae_denoised"]:.6f}', density=True)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Erro (reconstruído − limpo)')
    ax.set_ylabel('Densidade')
    ax.set_title('Distribuição do Erro por Elemento')
    ax.legend()

    ax = axes[1]
    mae_per_channel_noisy    = np.abs(err_noisy).mean(axis=0)
    mae_per_channel_denoised = np.abs(err_denoised).mean(axis=0)
    ch = np.arange(16)
    w = 0.35
    ax.bar(ch - w/2, mae_per_channel_noisy,    w, color='coral',         label='ruidoso',  alpha=0.8)
    ax.bar(ch + w/2, mae_per_channel_denoised, w, color='mediumseagreen', label='denoised', alpha=0.8)
    ax.set_xlabel('Canal')
    ax.set_ylabel('MAE')
    ax.set_title('MAE por Canal')
    ax.legend()
    ax.set_xticks(ch)

    plt.tight_layout()
    plt.savefig('dae_error_distribution.png', dpi=150)
    plt.show()
    print('Gráfico salvo: dae_error_distribution.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f'Dispositivo: {DEVICE}')
    print()

    # 1. Dados e pares de correspondência
    X_noisy, X_clean, y_noisy, X_orig, y_orig, n_art = load_paired_data(ART_FILE)

    # 2. Divisão por índices originais (sem data leakage)
    N_orig = X_orig.shape[0]
    train_idx, val_idx, test_idx, train_orig_idx = split_by_original(
        N_orig, n_art, TEST_FRAC, VAL_FRAC, RANDOM_STATE
    )

    # 3. Normalização: ajuste apenas nos dados limpos de treino
    scaler = StandardScaler()
    scaler.fit(X_clean[train_idx])

    X_noisy_sc = scaler.transform(X_noisy)
    X_clean_sc = scaler.transform(X_clean)

    # Subconjuntos
    X_noisy_train = X_noisy_sc[train_idx];  X_clean_train = X_clean_sc[train_idx]
    X_noisy_val   = X_noisy_sc[val_idx];    X_clean_val   = X_clean_sc[val_idx]
    X_noisy_test  = X_noisy_sc[test_idx];   X_clean_test  = X_clean_sc[test_idx]

    print(f'Shapes (normalizados):')
    print(f'  treino  : X_noisy={X_noisy_train.shape}  X_clean={X_clean_train.shape}')
    print(f'  val     : X_noisy={X_noisy_val.shape}    X_clean={X_clean_val.shape}')
    print(f'  teste   : X_noisy={X_noisy_test.shape}   X_clean={X_clean_test.shape}')
    print()

    # 4. DataLoaders
    train_loader = DataLoader(
        NoisyCleanDataset(X_noisy_train, X_clean_train),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        NoisyCleanDataset(X_noisy_val, X_clean_val),
        batch_size=BATCH_SIZE,
    )

    # 5. Modelo
    model = DenoisingAutoencoder(input_dim=16, latent_dim=LATENT_DIM)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Modelo: DenoisingAutoencoder')
    print(f'  input_dim  = 16')
    print(f'  latent_dim = {LATENT_DIM}')
    print(f'  parâmetros = {n_params}')
    print(f'  conexão residual = True  (saída = x̃ + Decoder(Encoder(x̃)))')
    print()

    # 6. Treino
    print('Iniciando treino...')
    history = train(
        model, train_loader, val_loader,
        EPOCHS, LR, WEIGHT_DECAY, PATIENCE, DEVICE
    )

    # 7. Carrega o melhor modelo e avalia no teste
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    model.to(DEVICE)

    print()
    results = evaluate(model, X_noisy_test, X_clean_test, scaler, DEVICE)

    # 8. Salva checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_dim': 16,
        'latent_dim': LATENT_DIM,
        'history': history,
        'art_file': str(ART_FILE),
        'n_art': n_art,
        'metrics': {
            'mae_noisy':    results['mae_noisy'],
            'mae_denoised': results['mae_denoised'],
            'mse_noisy':    results['mse_noisy'],
            'mse_denoised': results['mse_denoised'],
            'snr_noisy':    results['snr_noisy'],
            'snr_denoised': results['snr_denoised'],
        }
    }
    joblib.dump(checkpoint, CHECKPOINT_OUT)
    print(f'\nCheckpoint salvo: {CHECKPOINT_OUT}')

    # 9. Visualizações
    plot_training_history(history)
    plot_reconstruction(results)
    plot_error_distribution(results)


if __name__ == '__main__':
    main()
