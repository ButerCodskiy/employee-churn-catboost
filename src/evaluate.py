"""Модуль финальной верификации на отложенном тесте и аналитической интерпретации SHAP."""

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool

from src.config import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    N_SPLITS,
    RAW_CATEGORICAL_COLS,
    REPORTS_DIR,
    TARGET_COL,
)
from src.features import HRFeatureTransformer
from src.metrics import (
    evaluate_threshold_metrics,
    find_optimal_threshold,
    plot_confusion_matrix_heatmap,
    plot_precision_recall_curve,
)


def load_trained_ensemble() -> tuple[list[CatBoostClassifier], Any, HRFeatureTransformer]:
    """Загружает сохраненные модели фолдов, калибратор и трансформер признаков."""
    models: list[CatBoostClassifier] = []

    for fold_idx in range(N_SPLITS):
        model_path = MODELS_DIR / f"catboost_fold_{fold_idx}.cbm"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Файл модели {model_path} не найден. Сначала выполните train.py."
            )
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        models.append(model)

    calibrator_path = MODELS_DIR / "calibrator.pkl"
    transformer_path = MODELS_DIR / "feature_transformer.pkl"

    if not calibrator_path.exists() or not transformer_path.exists():
        raise FileNotFoundError("Калибратор или трансформер признаков не найдены в models/.")

    calibrator = joblib.load(calibrator_path)
    transformer = joblib.load(transformer_path)

    return models, calibrator, transformer


def prepare_holdout_test_data(
    transformer: HRFeatureTransformer,
) -> tuple[pd.DataFrame, np.ndarray, Pool]:
    """Загружает и преобразует изолированную тестовую выборку."""
    test_path = DATA_PROCESSED_DIR / "test_holdout.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Файл {test_path} не найден.")

    df_test = pd.read_parquet(test_path)
    y_test = df_test[TARGET_COL].values
    X_test = df_test.drop(columns=[TARGET_COL])

    X_test_trans = transformer.transform(X_test)
    cat_features = [c for c in RAW_CATEGORICAL_COLS if c in X_test_trans.columns]

    for c in cat_features:
        X_test_trans[c] = X_test_trans[c].astype(str).fillna("Missing")

    test_pool = Pool(X_test_trans, y_test, cat_features=cat_features)
    return X_test_trans, y_test, test_pool


def calculate_ensemble_predictions(
    models: list[CatBoostClassifier],
    test_pool: Pool,
    calibrator: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Вычисляет усредненные предсказания ансамбля и калиброванные вероятности."""
    fold_probs = [m.predict_proba(test_pool)[:, 1] for m in models]
    raw_ensemble_probs = np.mean(fold_probs, axis=0)
    calibrated_probs = calibrator.predict(raw_ensemble_probs)
    return raw_ensemble_probs, calibrated_probs


def calculate_ensemble_shap_values(
    models: list[CatBoostClassifier],
    test_pool: Pool,
) -> np.ndarray:
    """Вычисляет усредненные значения Шепли (TreeSHAP) по ансамблю моделей фолдов."""
    shap_list = [m.get_feature_importance(test_pool, type="ShapValues")[:, :-1] for m in models]
    ensemble_shap = np.mean(shap_list, axis=0)
    return ensemble_shap


def generate_shap_visualizations(
    shap_matrix: np.ndarray,
    X_features: pd.DataFrame,
) -> list[Path]:
    """Генерирует комплекс аналитических графиков SHAP для отчета."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated_plots: list[Path] = []

    summary_path = FIGURES_DIR / "shap_summary.png"
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_matrix, X_features, show=False)
    plt.title("Глобальное влияние факторов на риск увольнения (SHAP Beeswarm)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()
    generated_plots.append(summary_path)

    bar_path = FIGURES_DIR / "shap_importance_bar.png"
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_matrix, X_features, plot_type="bar", show=False)
    plt.title("Рейтинг важности признаков модели (Mean |SHAP Value|)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    generated_plots.append(bar_path)

    dep_overtime_path = FIGURES_DIR / "shap_dependence_overtime.png"
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        "OverTime",
        shap_matrix,
        X_features,
        interaction_index="TotalWorkingYears",
        show=False,
    )
    plt.title("SHAP зависимость: Переработки vs Общий стаж", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(dep_overtime_path, dpi=300)
    plt.close()
    generated_plots.append(dep_overtime_path)

    dep_stagnation_path = FIGURES_DIR / "shap_dependence_stagnation.png"
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        "Promotion_Stagnation_Ratio",
        shap_matrix,
        X_features,
        interaction_index="YearsAtCompany",
        show=False,
    )
    plt.title("SHAP зависимость: Застой в карьере vs Стаж в компании", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(dep_stagnation_path, dpi=300)
    plt.close()
    generated_plots.append(dep_stagnation_path)

    dep_income_path = FIGURES_DIR / "shap_dependence_income.png"
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        "MonthlyIncome",
        shap_matrix,
        X_features,
        interaction_index="JobLevel",
        show=False,
    )
    plt.title("SHAP зависимость: Ежемесячный доход vs Грейд", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(dep_income_path, dpi=300)
    plt.close()
    generated_plots.append(dep_income_path)

    return generated_plots


def run_full_evaluation() -> dict[str, Any]:
    """Проводит полную оценку качества на тестовом сете и формирует отчеты."""
    models, calibrator, transformer = load_trained_ensemble()
    X_test_trans, y_test, test_pool = prepare_holdout_test_data(transformer)

    oof_path = DATA_PROCESSED_DIR / "oof_predictions.parquet"
    df_oof = pd.read_parquet(oof_path)
    calibrated_oof = df_oof["calibrated_probability"].values
    y_oof = df_oof[TARGET_COL].values

    optimal_threshold, _ = find_optimal_threshold(y_oof, calibrated_oof)

    _, calibrated_test_probs = calculate_ensemble_predictions(models, test_pool, calibrator)
    test_metrics = evaluate_threshold_metrics(
        y_true=y_test,
        y_prob=calibrated_test_probs,
        threshold=optimal_threshold,
    )

    ensemble_shap = calculate_ensemble_shap_values(models, test_pool)
    generate_shap_visualizations(ensemble_shap, X_test_trans)

    test_cm_path = FIGURES_DIR / "test_confusion_matrix.png"
    plot_confusion_matrix_heatmap(
        y_true=y_test,
        y_prob=calibrated_test_probs,
        threshold=optimal_threshold,
        save_path=test_cm_path,
    )

    test_pr_path = FIGURES_DIR / "test_pr_curve.png"
    plot_precision_recall_curve(
        y_true=y_test,
        y_prob=calibrated_test_probs,
        optimal_threshold=optimal_threshold,
        save_path=test_pr_path,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "final_test_evaluation.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Итоговая оценка модели на изолированной тестовой выборке (Hold-out Test)\n\n")
        f.write(f"- Объем тестовой выборки: **{len(y_test)}** сотрудников\n")
        f.write(
            f"- Число уволившихся в тесте: **{int(np.sum(y_test == 1))}** ({np.mean(y_test):.1%})\n"
        )
        f.write(f"- Рабочий порог классификации (tau*): **{optimal_threshold:.2f}**\n\n")
        f.write("## Метрики ранжирования и классификации:\n\n")
        f.write("| Метрика | Значение |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **PR-AUC (Average Precision)** | **{test_metrics['pr_auc']:.4f}** |\n")
        f.write(f"| **ROC-AUC** | **{test_metrics['roc_auc']:.4f}** |\n")
        f.write(f"| **Полнота (Recall)** | **{test_metrics['recall']:.4f}** |\n")
        f.write(f"| **Точность (Precision)** | **{test_metrics['precision']:.4f}** |\n")
        f.write(f"| **F1-Score** | **{test_metrics['f1']:.4f}** |\n")
        f.write(f"| **F2-Score (приоритет Recall)** | **{test_metrics['f2']:.4f}** |\n")
        f.write(
            f"| **Доля правильных ответов (Accuracy)** | **{test_metrics['accuracy']:.4f}** |\n\n"
        )
        f.write("## Бизнес-показатели (HR Cost Matrix):\n\n")
        f.write(
            f"- Выявлено увольнений (TP): **{test_metrics['tp']}** из {int(np.sum(y_test == 1))}\n"
        )
        f.write(f"- Пропущено увольнений (FN): **{test_metrics['fn']}**\n")
        f.write(f"- Ложных тревог (FP): **{test_metrics['fp']}**\n")
        f.write(f"- Суммарные затраты компании: **{test_metrics['total_cost']:,.2f} $**\n")
        f.write(f"- Предотвращенный ущерб: **{test_metrics['cost_reduction']:,.2f} $**\n")

    return {
        "optimal_threshold": optimal_threshold,
        "test_metrics": test_metrics,
        "shap_matrix": ensemble_shap,
        "report_path": report_path,
    }


if __name__ == "__main__":
    res = run_full_evaluation()
    m = res["test_metrics"]
    print("Финальная оценка на Hold-out тесте успешно завершена.")
    print(f"Порог tau*: {res['optimal_threshold']:.2f}")
    print(f"Test PR-AUC: {m['pr_auc']:.4f}")
    print(f"Test ROC-AUC: {m['roc_auc']:.4f}")
    print(f"Test Recall: {m['recall']:.4f}")
    print(f"Test Precision: {m['precision']:.4f}")
    print(f"Test F1: {m['f1']:.4f}")
    print(f"Предотвращенный ущерб: {m['cost_reduction']:,.2f} $")
