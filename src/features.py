"""Модуль генерации доменных признаков для прогнозирования оттока сотрудников."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


class HRFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-Learn совместимый трансформер генерации доменных HR-признаков без утечек данных."""

    def __init__(self) -> None:
        self.role_medians_: dict[str, float] = {}
        self.dept_medians_: dict[str, float] = {}
        self.global_income_median_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HRFeatureTransformer":
        self.global_income_median_ = float(X["MonthlyIncome"].median())
        self.role_medians_ = X.groupby("JobRole")["MonthlyIncome"].median().to_dict()
        self.dept_medians_ = X.groupby("Department")["MonthlyIncome"].median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_out = X.copy()

        df_out["Promotion_Stagnation_Ratio"] = (df_out["YearsSinceLastPromotion"] + 1.0) / (
            df_out["YearsAtCompany"] + 1.0
        )

        df_out["Manager_Stability"] = (df_out["YearsWithCurrManager"] + 1.0) / (
            df_out["YearsInCurrentRole"] + 1.0
        )

        role_medians = df_out["JobRole"].map(self.role_medians_).fillna(self.global_income_median_)
        df_out["Relative_Income_to_Role"] = df_out["MonthlyIncome"] / role_medians

        dept_medians = (
            df_out["Department"].map(self.dept_medians_).fillna(self.global_income_median_)
        )
        df_out["Relative_Income_to_Dept"] = df_out["MonthlyIncome"] / dept_medians

        df_out["Overall_Satisfaction"] = (
            df_out["EnvironmentSatisfaction"]
            + df_out["JobSatisfaction"]
            + df_out["RelationshipSatisfaction"]
        )

        travel_weights = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
        travel_score = df_out["BusinessTravel"].map(travel_weights).fillna(1)

        overtime_flags = {"Yes": 1, "No": 0, 1: 1, 0: 0}
        overtime_score = df_out["OverTime"].map(overtime_flags).fillna(0)

        df_out["Burnout_Index"] = (
            overtime_score * (5.0 - df_out["WorkLifeBalance"]) * (travel_score + 1.0)
        )

        df_out["Tenure_to_TotalWorkingYears"] = (df_out["YearsAtCompany"] + 1.0) / (
            df_out["TotalWorkingYears"] + 1.0
        )

        safe_age = np.maximum(df_out["Age"].values, 18.0)
        df_out["Tenure_to_Age"] = df_out["YearsAtCompany"] / safe_age

        safe_working_years = np.maximum(df_out["TotalWorkingYears"].values, 1.0)
        df_out["Company_Hop_Frequency"] = df_out["NumCompaniesWorked"] / safe_working_years

        return df_out


def calculate_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Вычисляет взаимную информацию признаков с целевой переменной оттока."""
    X_encoded = X.copy()
    discrete_features = []

    for idx, col in enumerate(X_encoded.columns):
        if col in categorical_cols or X_encoded[col].dtype == "object":
            X_encoded[col] = X_encoded[col].astype("category").cat.codes
            discrete_features.append(idx)
        elif pd.api.types.is_integer_dtype(X_encoded[col]):
            discrete_features.append(idx)

    X_filled = X_encoded.fillna(X_encoded.median(numeric_only=True))
    mi_scores = mutual_info_classif(
        X_filled,
        y,
        discrete_features=discrete_features,
        random_state=random_state,
    )

    df_mi = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "mutual_information": mi_scores,
            }
        )
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )

    return df_mi


FEATURE_DICTIONARY: dict[str, str] = {
    "Promotion_Stagnation_Ratio": "Отношение стажа без повышения к общему стажу в компании: индикатор застоя в карьере.",
    "Manager_Stability": "Отношение срока работы с текущим руководителем к сроку в текущей роли: маркер конфликта со свежим руководством.",
    "Relative_Income_to_Role": "Отношение оклада сотрудника к медианному окладу коллег той же должности.",
    "Relative_Income_to_Dept": "Отношение оклада к медианному уровню вознаграждения всего департамента.",
    "Overall_Satisfaction": "Суммарный индекс удовлетворенности (среда, работа, взаимоотношения в коллективе).",
    "Burnout_Index": "Фактор выгорания: мультипликация сверхурочных часов, частоты поездок и дефицита баланса личной жизни.",
    "Tenure_to_TotalWorkingYears": "Доля карьеры, проведенная в текущей компании: мера организационной лояльности.",
    "Tenure_to_Age": "Отношение стажа в компании к возрасту сотрудника.",
    "Company_Hop_Frequency": "Средняя частота смены мест работы: склонность к частым переходам.",
}
