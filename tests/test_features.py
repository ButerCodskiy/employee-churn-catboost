"""Модульные тесты проверки третьего этапа: доменный Feature Engineering."""

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    RAW_CATEGORICAL_COLS,
    TARGET_COL,
)
from src.features import (
    FEATURE_DICTIONARY,
    HRFeatureTransformer,
    calculate_mutual_information,
)


def test_feature_transformer_fit_transform():
    dev_df = pd.read_parquet(DATA_PROCESSED_DIR / "train_dev.parquet")
    X = dev_df.drop(columns=[TARGET_COL, "fold"])

    transformer = HRFeatureTransformer()
    transformer.fit(X)
    X_transformed = transformer.transform(X)

    for feature_name in FEATURE_DICTIONARY:
        assert feature_name in X_transformed.columns, f"Признак {feature_name} не сгенерирован"
        assert not X_transformed[feature_name].isna().all(), (
            f"Признак {feature_name} содержит сплошные пропуски"
        )

    assert len(X_transformed) == len(X), "Количество строк изменилось после трансформации"


def test_feature_transformer_leak_freeness():
    train_data = pd.DataFrame(
        {
            "MonthlyIncome": [5000.0, 10000.0, 3000.0, 8000.0],
            "JobRole": ["Developer", "Developer", "Sales", "Sales"],
            "Department": ["IT", "IT", "Sales", "Sales"],
            "YearsSinceLastPromotion": [1, 2, 0, 3],
            "YearsAtCompany": [5, 10, 2, 6],
            "YearsWithCurrManager": [2, 4, 1, 3],
            "YearsInCurrentRole": [3, 5, 1, 4],
            "EnvironmentSatisfaction": [3, 4, 2, 4],
            "JobSatisfaction": [3, 3, 4, 2],
            "RelationshipSatisfaction": [4, 3, 2, 3],
            "BusinessTravel": ["Travel_Rarely", "Non-Travel", "Travel_Frequently", "Travel_Rarely"],
            "OverTime": ["Yes", "No", "Yes", "No"],
            "WorkLifeBalance": [3, 4, 2, 3],
            "TotalWorkingYears": [8, 15, 4, 10],
            "Age": [28, 40, 24, 35],
            "NumCompaniesWorked": [2, 3, 1, 4],
        }
    )

    val_data = pd.DataFrame(
        {
            "MonthlyIncome": [6000.0, 12000.0],
            "JobRole": ["Developer", "NewUnseenRole"],
            "Department": ["IT", "NewUnseenDept"],
            "YearsSinceLastPromotion": [0, 4],
            "YearsAtCompany": [3, 8],
            "YearsWithCurrManager": [1, 5],
            "YearsInCurrentRole": [2, 6],
            "EnvironmentSatisfaction": [4, 2],
            "JobSatisfaction": [3, 4],
            "RelationshipSatisfaction": [3, 2],
            "BusinessTravel": ["Non-Travel", "Travel_Frequently"],
            "OverTime": ["No", "Yes"],
            "WorkLifeBalance": [3, 2],
            "TotalWorkingYears": [6, 12],
            "Age": [30, 42],
            "NumCompaniesWorked": [1, 2],
        }
    )

    transformer = HRFeatureTransformer()
    transformer.fit(train_data)
    val_transformed = transformer.transform(val_data)

    expected_dev_median = 7500.0
    actual_val_ratio_0 = val_transformed.loc[0, "Relative_Income_to_Role"]
    expected_ratio_0 = 6000.0 / expected_dev_median

    assert np.isclose(actual_val_ratio_0, expected_ratio_0), (
        "Медиана обучающей выборки не была изолирована"
    )
    assert not val_transformed.isna().any().any(), (
        "Обнаружены необработанные пропуски при трансформации валидации"
    )


def test_mutual_information_computation():
    dev_df = pd.read_parquet(DATA_PROCESSED_DIR / "train_dev.parquet")
    X = dev_df.drop(columns=[TARGET_COL, "fold"])
    y = dev_df[TARGET_COL]

    transformer = HRFeatureTransformer()
    X_enriched = transformer.fit_transform(X)

    mi_df = calculate_mutual_information(X_enriched, y, categorical_cols=RAW_CATEGORICAL_COLS)

    assert len(mi_df) == len(X_enriched.columns), "Таблица взаимной информации неполна"
    assert (mi_df["mutual_information"] >= 0.0).all(), "Отрицательные значения взаимной информации"

    top_features = mi_df.head(5)["feature"].tolist()
    assert len(top_features) == 5, "Топ-5 признаков не определены"


def test_feature_dictionary_completeness():
    assert len(FEATURE_DICTIONARY) == 9, (
        f"Ожидалось 9 доменных признаков, найдено {len(FEATURE_DICTIONARY)}"
    )
    for key, desc in FEATURE_DICTIONARY.items():
        assert len(desc) > 10, f"Описание признака {key} слишком короткое"


if __name__ == "__main__":
    test_feature_transformer_fit_transform()
    print("[OK] test_feature_transformer_fit_transform")
    test_feature_transformer_leak_freeness()
    print("[OK] test_feature_transformer_leak_freeness")
    test_mutual_information_computation()
    print("[OK] test_mutual_information_computation")
    test_feature_dictionary_completeness()
    print("[OK] test_feature_dictionary_completeness")
    print("Все тесты модуля Feature Engineering успешно пройдены.")
