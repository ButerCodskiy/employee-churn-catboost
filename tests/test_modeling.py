"""Модульные тесты проверки четвертого этапа: обучение CatBoost и калибровка вероятностей."""

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from src.calibrate import run_probability_calibration
from src.config import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    N_SPLITS,
)
from src.train import run_cross_validation_training


def test_cv_training_and_artifacts():
    results = run_cross_validation_training(iterations=100, early_stopping_rounds=20)

    models = results["models"]
    assert len(models) == N_SPLITS, f"Ожидалось {N_SPLITS} моделей, получено {len(models)}"

    for fold_idx in range(N_SPLITS):
        model_file = MODELS_DIR / f"catboost_fold_{fold_idx}.cbm"
        assert model_file.exists(), f"Файл модели {model_file} не найден на диске"

        loaded_model = CatBoostClassifier()
        loaded_model.load_model(str(model_file))
        tree_depth = loaded_model.get_params().get("depth", 6)
        assert tree_depth <= 6, f"Глубина дерева {tree_depth} превышает лимит 6"

    oof_file = DATA_PROCESSED_DIR / "oof_predictions.parquet"
    assert oof_file.exists(), "Файл OOF-предсказаний отсутствует"

    df_oof = pd.read_parquet(oof_file)
    assert "oof_probability" in df_oof.columns, "Колонка oof_probability отсутствует"
    assert (df_oof["oof_probability"] >= 0.0).all() and (df_oof["oof_probability"] <= 1.0).all(), (
        "Вероятности OOF выходят за пределы диапазона [0, 1]"
    )

    learning_curve_plot = FIGURES_DIR / "learning_curves.png"
    assert learning_curve_plot.exists() and learning_curve_plot.stat().st_size > 0, (
        "График кривых обучения не сгенерирован"
    )


def test_probability_calibration():
    cal_results = run_probability_calibration()

    calibrator_file = MODELS_DIR / "calibrator.pkl"
    assert calibrator_file.exists(), "Файл калибратора отсутсвует"

    calibrator = joblib.load(calibrator_file)
    assert hasattr(calibrator, "predict"), "Калибратор не имеет метода predict"

    assert cal_results["brier_after"] <= cal_results["brier_before"], (
        "Калибровка ухудшила показатель Brier Score"
    )
    assert cal_results["brier_after"] < 0.15, (
        f"Итоговый Brier Score {cal_results['brier_after']:.4f} слишком высок"
    )

    cal_plot = FIGURES_DIR / "calibration_curve.png"
    assert cal_plot.exists() and cal_plot.stat().st_size > 0, (
        "График диаграммы надежности не сформирован"
    )

    df_oof = pd.read_parquet(DATA_PROCESSED_DIR / "oof_predictions.parquet")
    assert "calibrated_probability" in df_oof.columns, "Колонка calibrated_probability отсутствует"


if __name__ == "__main__":
    test_cv_training_and_artifacts()
    print("[OK] test_cv_training_and_artifacts")
    test_probability_calibration()
    print("[OK] test_probability_calibration")
    print("Все тесты четвертого этапа успешно пройдены.")
