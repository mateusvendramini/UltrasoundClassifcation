import h5py
from pathlib import Path
import numpy as np

DATA_DIR = Path('.')
file_path = DATA_DIR / 'model_compound_4_10_34_1_interface.h5'

print(f"Inspecting: {file_path.resolve()}")
if not file_path.exists():
    print('File not found:', file_path)
    raise SystemExit(1)

with h5py.File(file_path, 'r') as f:
    print('\nFile attributes:')
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")

    entries = []
    def visitor(name, obj):
        kind = 'Group' if isinstance(obj, h5py.Group) and not isinstance(obj, h5py.Dataset) else 'Dataset'
        if isinstance(obj, h5py.Dataset):
            shape = getattr(obj, 'shape', None)
            dtype = getattr(obj, 'dtype', None)
            entries.append((name, kind, shape, str(dtype), dict(obj.attrs)))
        else:
            entries.append((name, kind, None, None, dict(obj.attrs)))
    f.visititems(visitor)

    print('\nContents:')
    for name, kind, shape, dtype, attrs in entries:
        print(f"- {kind}: {name}")
        if shape is not None:
            print(f"    shape: {shape}, dtype: {dtype}")
        if attrs:
            print("    attrs:")
            for k, v in attrs.items():
                print(f"      {k}: {v}")

    # try to locate N_compound_maxenv
    target = None
    for name, kind, shape, dtype, attrs in entries:
        if name.endswith('N_compound_maxenv'):
            target = name
            break

    if target is None:
        print("\nDataset 'N_compound_maxenv' not found by suffix search. Listing top-level datasets for hints:")
        for name, kind, shape, dtype, attrs in entries[:40]:
            print(' ', name)
    else:
        print(f"\nFound target dataset: {target}")
        d = f[target]
        print('Shape:', d.shape)
        print('Dtype:', d.dtype)
        print('Dataset attributes:')
        for k, v in d.attrs.items():
            print(f"  {k}: {v}")

        # print a small sample: zeros index and shapes of slices
        try:
            data = d[()]
        except Exception as e:
            print('Could not read full dataset into memory:', e)
            # try to read small slices
            data = None

        if data is not None:
            arr = np.asanyarray(data)
            print('ndim:', arr.ndim)
            print('total elements:', arr.size)
            try:
                zeros_idx = tuple(0 for _ in range(arr.ndim))
                print('\nValue at all-zero indices:', arr[zeros_idx])
            except Exception as e:
                print('Could not index all zeros:', e)

            for axis in range(arr.ndim):
                slicer = [slice(None)] * arr.ndim
                slicer[axis] = 0
                vals = arr[tuple(slicer)]
                print(f"\nSlice axis={axis} index=0 shape={np.array(vals).shape}\nFirst 10 values:\n", np.array(vals).ravel()[:10])

        # --- Adicionado: ler as 16 medições por altura (e por axis_K) e imprimir ---
        try:
            axis_H = np.asanyarray(d.attrs.get('axis_H', np.arange(d.shape[0])))
        except Exception:
            axis_H = np.arange(d.shape[0])
        try:
            axis_K = np.asanyarray(d.attrs.get('axis_K', np.arange(d.shape[2])))
        except Exception:
            axis_K = np.arange(d.shape[2])
        axis_D = d.attrs.get('axis_D', None)
        axis_F = d.attrs.get('axis_F', None)

        print('\n--- Lendo medidas (16 valores) por altura e axis_K ---')
        print(f'axis_D={axis_D}, axis_F={axis_F}')
        print(f'Número de alturas: {len(axis_H)}, número de axis_K: {len(axis_K)}')

        # Itera por cada altura e por cada valor de axis_K e imprime os 16 valores
        for hi, hval in enumerate(axis_H):
            for ki, kval in enumerate(axis_K):
                try:
                    vals = d[hi, 0, ki, 0, :]
                except Exception as e:
                    print(f'Erro lendo slice H={hi} K={ki}: {e}')
                    continue
                vals_arr = np.asanyarray(vals)
                print(f"H_index={hi} height={float(hval)} | K_index={ki} K_val={float(kval)} -> measurements (len={vals_arr.size}):")
                # imprime os 16 valores em uma linha
                print(', '.join([f"{float(x):.6g}" for x in vals_arr]))

print('\nDone')
