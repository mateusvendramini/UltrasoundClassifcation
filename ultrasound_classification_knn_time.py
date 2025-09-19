"""
Run a KNN classification pipeline on the Ultrasound_Interface_Time dataset.

This script:
 - Reads CSV/HDF files under large_dataset/Ultrasound_Interface_Time
 - Extracts height labels from filenames (60,63,...,240)
 - Concatenates all rows from `M` folders as the final test set
 - Uses remaining files for train/validation split (stratified by height)
 - Trains a StandardScaler + KNeighborsClassifier pipeline with GridSearchCV
 - Evaluates on validation and test sets and saves the trained model (joblib)

Usage:
    python ultrasound_classification_knn_time.py

"""

from pathlib import Path
import re
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Config
BASE_DIR = Path('large_dataset') / 'Ultrasound_Interface_Time'
HEIGHTS = list(range(60, 241, 3))
HEIGHTS_SET = set(map(str, HEIGHTS))
HEIGHT_RE = re.compile(r'_([0-9]{2,3})_water')


def extract_height_from_name(fname: str) -> str:
    m = HEIGHT_RE.search(fname)
    return m.group(1) if m else None


def read_timeseries_file(path: Path) -> np.ndarray:
    """Read a CSV or h5/hdf file and return its values as np.ndarray.

    Each row is treated as a sample.
    """
    try:
        df = pd.read_csv(path, header=None)
        return df.values
    except Exception:
        # try hdf
        try:
            df = pd.read_hdf(path)
            return df.values
        except Exception as e:
            logger.error('Failed to read %s: %s', path, e)
            raise


def build_dataset_from_files(file_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    for p in file_paths:
        height = extract_height_from_name(p.name)
        if height is None or height not in HEIGHTS_SET:
            logger.debug('Skipping file (no/invalid height): %s', p)
            continue
        arr = read_timeseries_file(p)
        if arr.size == 0:
            continue
        n = arr.shape[0]
        X_list.append(arr)
        y_list.append(np.full(n, int(height), dtype=int))
    if not X_list:
        return np.empty((0, 0)), np.empty((0,))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def collect_files(base_dir: Path) -> Tuple[List[Path], List[Path]]:
    all_files = []
    for speed_dir in sorted(base_dir.iterdir()):
        if not speed_dir.is_dir():
            continue
        for sub in sorted(speed_dir.iterdir()):
            if sub.is_dir():
                for f in sorted(sub.glob('*')):
                    if f.is_file():
                        all_files.append((speed_dir.name, sub.name, f))

    test_files = [f for speed, sub, f in all_files if sub == 'M']
    #trainval_files = [f for speed, sub, f in all_files if sub != 'M']
    return test_files#, trainval_files


def main():
    logger.info('Base dir: %s', BASE_DIR)
    if not BASE_DIR.exists():
        raise SystemExit(f'Base dir not found: {BASE_DIR}')

    test_files = collect_files(BASE_DIR)
    logger.info('Found total files: test=%d ', len(test_files))

    X_test, y_test = build_dataset_from_files(test_files)
    #X_rem, y_rem = build_dataset_from_files(trainval_files)

    logger.info('Test samples/features: %s', X_test.shape)
    #logger.info('Remaining samples/features: %s', X_rem.shape)
    logger.info('Test label counts: %s', Counter(y_test))
    #logger.info('Remaining label counts sample: %s', dict(Counter(y_rem).most_common(10)))

    if X_test.size == 0:
        raise SystemExit('No data found for test or train/val. Check paths and filename patterns.')

    # Align feature dims if needed

    # Split remaining into train/val stratified by label
    X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42, stratify=y_test)

    logger.info('Train/Val split: train=%s val=%s', X_train.shape, X_val.shape)

    # Encode labels
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    logger.info('Classes (original heights): %s', le.classes_)

    # Pipeline and grid
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': [1, 3, 5, 7],
        'knn__weights': ['uniform', 'distance']
    }

    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    logger.info('Starting GridSearchCV...')
    gs.fit(X_train, y_train_enc)
    logger.info('Best params: %s', gs.best_params_)
    logger.info('Best CV score: %s', gs.best_score_)

    best = gs.best_estimator_

    # Validation
    val_pred = best.predict(X_val)
    logger.info('Validation accuracy: %f', accuracy_score(y_val_enc, val_pred))
    logger.info('Validation classification report:\n%s', classification_report(y_val_enc, val_pred, target_names=[str(c) for c in le.classes_]))

    # Test
    test_pred = best.predict(X_test)
    logger.info('Test accuracy: %f', accuracy_score(y_test_enc, test_pred))
    logger.info('Test classification report:\n%s', classification_report(y_test_enc, test_pred, target_names=[str(c) for c in le.classes_]))

    out = Path('knn_time_model.joblib')
    joblib.dump({'model': best, 'label_encoder': le}, out)
    logger.info('Saved model to %s', out)


if __name__ == '__main__':
    main()
