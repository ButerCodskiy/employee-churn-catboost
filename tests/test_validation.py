"""Модульные тесты проверки первого этапа: валидация без утечек данных."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    N_SPLITS,
    RAW_NUMERICAL_COLS,
    TARGET_COL,
)
from src.data_loader import (
    create_leak_free_splits,
    extract_archive,
    load_raw_train_data,
)


def test_archive_extraction():
    extracted = extract_archive(extract_to=DATA_RAW_DIR)
    assert len(extracted) > 0, "Файлы не извлечены из архива"
    train_file = DATA_RAW_DIR / "employee_attrition_train.csv"
    assert train_file.exists(), f"Файл {train_file} отсутствует после распаковки"


def test_holdout_isolation_and_no_overlap():
    dev_df, test_df, _ = create_leak_free_splits()

    total_samples = len(dev_df) + len(test_df)
    raw_df = load_raw_train_data()
    assert total_samples == len(raw_df), (
        f"Количество строк {total_samples} не совпадает с исходным {len(raw_df)}"
    )

    expected_test_len = round(len(raw_df) * 0.2)
    assert abs(len(test_df) - expected_test_len) <= 1, (
        f"Размер контрольной выборки {len(test_df)} отклоняется от целевого {expected_test_len}"
    )

    raw_pos_rate = raw_df[TARGET_COL].mean()
    test_pos_rate = test_df[TARGET_COL].mean()
    dev_pos_rate = dev_df[TARGET_COL].mean()

    assert abs(test_pos_rate - raw_pos_rate) < 0.01, (
        f"Доля оттока в тесте {test_pos_rate:.4f} отклоняется от исходной {raw_pos_rate:.4f}"
    )
    assert abs(dev_pos_rate - raw_pos_rate) < 0.01, (
        f"Доля оттока в dev {dev_pos_rate:.4f} отклоняется от исходной {raw_pos_rate:.4f}"
    )


def test_stratified_folds_balance():
    dev_df = pd.read_parquet(DATA_PROCESSED_DIR / "train_dev.parquet")

    assert "fold" in dev_df.columns, "Колонка индекса фолда отсутствует в train_dev.parquet"
    unique_folds = sorted(dev_df["fold"].unique())
    assert unique_folds == list(range(N_SPLITS)), (
        f"Ожидались фолды 0..{N_SPLITS - 1}, получены {unique_folds}"
    )

    dev_pos_rate = dev_df[TARGET_COL].mean()

    for f in unique_folds:
        val_fold = dev_df[dev_df["fold"] == f]
        val_rate = val_fold[TARGET_COL].mean()
        assert abs(val_rate - dev_pos_rate) < 0.015, (
            f"В фолде {f} доля {val_rate:.4f} отклоняется от общей {dev_pos_rate:.4f}"
        )


def test_leak_free_imputation_isolation():
    dev_df = pd.read_parquet(DATA_PROCESSED_DIR / "train_dev.parquet")

    f0_train = dev_df[dev_df["fold"] != 0]

    num_cols = [c for c in RAW_NUMERICAL_COLS if c in dev_df.columns]

    imputer = SimpleImputer(strategy="median")
    imputer.fit(f0_train[num_cols])

    train_medians = f0_train[num_cols].median().values
    assert np.allclose(imputer.statistics_, train_medians, equal_nan=True), (
        "Статистики импутации не совпадают строго с медианами обучающего фолда"
    )


if __name__ == "__main__":
    test_archive_extraction()
    print("[OK] test_archive_extraction")
    test_holdout_isolation_and_no_overlap()
    print("[OK] test_holdout_isolation_and_no_overlap")
    test_stratified_folds_balance()
    print("[OK] test_stratified_folds_balance")
    test_leak_free_imputation_isolation()
    print("[OK] test_leak_free_imputation_isolation")
    print("Все тесты валидации первого этапа успешно пройдены.")

