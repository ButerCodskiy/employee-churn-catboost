"""Модульные тесты для модуля финальной оценки, инференса и SHAP-анализа."""

import numpy as np

from src.config import FIGURES_DIR, REPORTS_DIR, TARGET_COL
from src.evaluate import (
    calculate_ensemble_predictions,
    calculate_ensemble_shap_values,
    load_trained_ensemble,
    prepare_holdout_test_data,
)


def test_ensemble_loading() -> None:
    """Проверяет корректность загрузки всех 5 моделей ансамбля и сопутствующих артефактов."""
    models, calibrator, transformer = load_trained_ensemble()
    assert len(models) == 5, f"Ожидалось 5 моделей в ансамбле, получено {len(models)}"
    assert hasattr(calibrator, "predict"), "Калибратор не реализует метод predict"
    assert hasattr(transformer, "transform"), "Трансформер не реализует метод transform"
    print("[OK] test_ensemble_loading пройден.")


def test_holdout_data_preparation() -> None:
    """Проверяет загрузку и трансформацию отложенного тестового датасета."""
    _, _, transformer = load_trained_ensemble()
    X_test_trans, y_test, test_pool = prepare_holdout_test_data(transformer)

    assert len(X_test_trans) == 206, f"Ожидалось 206 строк в тесте, получено {len(X_test_trans)}"
    assert len(y_test) == 206, f"Ожидалось 206 меток в тесте, получено {len(y_test)}"
    assert TARGET_COL not in X_test_trans.columns, (
        "Целевая колонка не должна присутствовать в матрице признаков"
    )
    assert test_pool.num_row() == 206, "Неверное число строк в объекте Pool"
    print("[OK] test_holdout_data_preparation пройден.")


def test_ensemble_predictions_bounded() -> None:
    """Проверяет, что ансамблевые и калиброванные вероятности строго ограничены диапазоном [0, 1]."""
    models, calibrator, transformer = load_trained_ensemble()
    _, _, test_pool = prepare_holdout_test_data(transformer)

    raw_probs, cal_probs = calculate_ensemble_predictions(models, test_pool, calibrator)
    assert len(raw_probs) == 206
    assert len(cal_probs) == 206
    assert np.all((raw_probs >= 0.0) & (raw_probs <= 1.0)), (
        "Сырые вероятности выходят за пределы [0, 1]"
    )
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0)), (
        "Калиброванные вероятности выходят за пределы [0, 1]"
    )
    print("[OK] test_ensemble_predictions_bounded пройден.")


def test_shap_values_computation() -> None:
    """Проверяет вычисление матриц TreeSHAP по ансамблю моделей."""
    models, _, transformer = load_trained_ensemble()
    X_test_trans, _, test_pool = prepare_holdout_test_data(transformer)

    shap_matrix = calculate_ensemble_shap_values(models, test_pool)
    assert shap_matrix.shape[0] == 206, (
        f"Число строк SHAP ({shap_matrix.shape[0]}) не совпадает с объемом теста"
    )
    assert shap_matrix.shape[1] == X_test_trans.shape[1], (
        "Число колонок SHAP не совпадает с признаками"
    )
    assert not np.isnan(shap_matrix).any(), "Матрица SHAP содержит пропущенные значения NaN"
    print("[OK] test_shap_values_computation пройден.")


def test_evaluation_artifacts_exist() -> None:
    """Проверяет генерацию и непустоту отчетов и графиков финального этапа."""
    report_file = REPORTS_DIR / "final_test_evaluation.md"
    recommendations_file = REPORTS_DIR / "hr_recommendations.md"

    expected_figures = [
        FIGURES_DIR / "shap_summary.png",
        FIGURES_DIR / "shap_importance_bar.png",
        FIGURES_DIR / "shap_dependence_overtime.png",
        FIGURES_DIR / "shap_dependence_stagnation.png",
        FIGURES_DIR / "shap_dependence_income.png",
        FIGURES_DIR / "test_confusion_matrix.png",
        FIGURES_DIR / "test_pr_curve.png",
    ]

    assert report_file.exists() and report_file.stat().st_size > 100, (
        "Отчет final_test_evaluation.md не создан"
    )
    assert recommendations_file.exists() and recommendations_file.stat().st_size > 100, (
        "Памятка hr_recommendations.md не создана"
    )

    for fig in expected_figures:
        assert fig.exists(), f"График {fig.name} не найден"
        assert fig.stat().st_size > 500, f"График {fig.name} пуст или поврежден"

    print("[OK] test_evaluation_artifacts_exist пройден.")


def run_all_tests() -> None:
    """Запускает полный комплект тестов модуля evaluate."""
    test_ensemble_loading()
    test_holdout_data_preparation()
    test_ensemble_predictions_bounded()
    test_shap_values_computation()
    test_evaluation_artifacts_exist()
    print("Все тесты модуля evaluate успешно пройдены!")


if __name__ == "__main__":
    run_all_tests()
