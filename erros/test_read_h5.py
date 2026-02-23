from pathlib import Path
import h5py
import numpy as np

p = Path('.') / 'model_compound_4_10_34_1_interface.h5'
print('Arquivo alvo:', p.resolve())
if not p.exists():
    print('Arquivo não encontrado:', p)
    raise SystemExit(1)

with h5py.File(p, 'r') as f:
    found = {'v': False}
    def check(name, obj):
        if isinstance(obj, h5py.Dataset):
            shape = getattr(obj, 'shape', None)
            if shape and len(shape) >= 1 and shape[0] == 101:
                found['v'] = True
                data = obj[()]
                print('\nDataset:', name, 'shape:', shape, 'dtype:', obj.dtype)
                try:
                    if getattr(data, 'ndim', 0) == 1:
                        uniques = np.unique(data)
                    else:
                        uniques = np.unique(data, axis=0)
                except Exception:
                    uniques = np.unique(np.asanyarray(data).ravel())
                cnt = getattr(uniques, 'shape', [len(uniques)])[0]
                print('Valores distintos (count={}):'.format(cnt))
                print(uniques)
    f.visititems(check)
    if not found['v']:
        print('Nenhum dataset com primeira dimensão igual a 101 foi encontrado.')
