"""Модуль извлечения, загрузки данных и стратифицированного разбиения без утечек."""

import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (
    ARCHIVE_NAMES,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DROP_COLS,
    N_SPLITS,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_COL,
    TARGET_MAPPING,
    TEST_SIZE,
)


def find_archive() -> Path:
    for name in ARCHIVE_NAMES:
        candidate = PROJECT_ROOT / name
        if candidate.is_file():
            return candidate

    for f in PROJECT_ROOT.glob("*.zip"):
        return f

    raise FileNotFoundError(
        f"Архив с данными не найден в {PROJECT_ROOT}. Проверены варианты: {ARCHIVE_NAMES}"
    )


def extract_archive(
    archive_path: Path | None = None,
    extract_to: Path = DATA_RAW_DIR,
    force: bool = False,
) -> list[Path]:
    if archive_path is None:
        archive_path = find_archive()

    extract_to.mkdir(parents=True, exist_ok=True)
    extracted_files = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.namelist():
            dest = extract_to / member
            if force or not dest.exists():
                zf.extract(member, extract_to)
            extracted_files.append(dest)

    return extracted_files


def load_raw_train_data(raw_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    train_path = raw_dir / "employee_attrition_train.csv"
    if not train_path.exists():
        extract_archive(extract_to=raw_dir)

    df = pd.read_csv(train_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Целевая колонка '{TARGET_COL}' отсутствует в файле {train_path}")

    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)

    return df


def load_raw_test_data(raw_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    test_path = raw_dir / "employee_attrition_test.csv"
    if not test_path.exists():
        extract_archive(extract_to=raw_dir)

    return pd.read_csv(test_path)


def create_leak_free_splits(
    df: pd.DataFrame | None = None,
    test_size: float = TEST_SIZE,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
    processed_dir: Path = DATA_PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df is None:
        df = load_raw_train_data()

    clean_drop = [c for c in DROP_COLS if c in df.columns]
    df_clean = df.drop(columns=clean_drop)

    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    dev_df = X_dev.copy()
    dev_df[TARGET_COL] = y_dev.values

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dev_df["fold"] = -1

    fold_stats = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_df, dev_df[TARGET_COL])):
        dev_df.iloc[val_idx, dev_df.columns.get_loc("fold")] = fold_idx

        val_subset = dev_df.iloc[val_idx]
        train_subset = dev_df.iloc[train_idx]

        val_pos = val_subset[TARGET_COL].sum()
        val_total = len(val_subset)
        train_pos = train_subset[TARGET_COL].sum()
        train_total = len(train_subset)

        fold_stats.append(
            {
                "fold": fold_idx,
                "train_samples": train_total,
                "train_pos_rate": train_pos / train_total,
                "val_samples": val_total,
                "val_pos_rate": val_pos / val_total,
            }
        )

    stats_df = pd.DataFrame(fold_stats)

    processed_dir.mkdir(parents=True, exist_ok=True)
    dev_path = processed_dir / "train_dev.parquet"
    test_path = processed_dir / "test_holdout.parquet"

    dev_df.to_parquet(dev_path, index=False)
    test_df.to_parquet(test_path, index=False)

    return dev_df, test_df, stats_df


if __name__ == "__main__":
    extracted = extract_archive()
    dev_df, test_df, stats_df = create_leak_free_splits()
    print(f"Размер обучающей выборки dev: {dev_df.shape}")
    print(f"Размер тестовой выборки holdout: {test_df.shape}")
    print(f"Баланс классов в тесте: {test_df[TARGET_COL].value_counts(normalize=True).to_dict()}")
    print("Баланс классов по фолдам:")
    print(stats_df.to_string(index=False))
