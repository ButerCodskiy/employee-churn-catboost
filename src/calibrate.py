"""Модуль калибровки вероятностей предсказаний для выравнивания риск-скоров."""

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.config import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    TARGET_COL,
)
from src.metrics import evaluate_threshold_metrics, find_optimal_threshold


def plot_reliability_diagram(
    y_true: np.ndarray,
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    brier_before: float,
    brier_after: float,
    save_path: Path | None = None,
) -> Path:
    """Строит диаграмму надежности (Reliability Curve) до и после калибровки."""
    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "calibration_curve.png"

    prob_true_raw, prob_pred_raw = calibration_curve(
        y_true, uncalibrated_probs, n_bins=8, strategy="quantile"
    )
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true, calibrated_probs, n_bins=8, strategy="quantile"
    )

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Идеальная калибровка")
    plt.plot(
        prob_pred_raw,
        prob_true_raw,
        marker="s",
        color="#d62728",
        lw=2,
        label=f"До калибровки (Brier = {brier_before:.4f})",
    )
    plt.plot(
        prob_pred_cal,
        prob_true_cal,
        marker="o",
        color="#2ca02c",
        lw=2,
        label=f"После калибровки (Brier = {brier_after:.4f})",
    )

    plt.title("Диаграмма надежности вероятностей (Reliability Diagram)", fontsize=14, pad=12)
    plt.xlabel("Средняя предсказанная вероятность", fontsize=12)
    plt.ylabel("Фактическая доля уволившихся", fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def run_probability_calibration() -> dict[str, Any]:
    """Обучает калибратор вероятностей IsotonicRegression на OOF-предсказаниях."""
    oof_path = DATA_PROCESSED_DIR / "oof_predictions.parquet"
    if not oof_path.exists():
        raise FileNotFoundError(f"Файл {oof_path} не найден. Требуется сначала выполнить обучение.")

    df_oof = pd.read_parquet(oof_path)
    y_true = df_oof[TARGET_COL].values
    uncalibrated_probs = df_oof["oof_probability"].values

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(uncalibrated_probs, y_true)

    calibrated_probs = calibrator.predict(uncalibrated_probs)

    brier_before = float(brier_score_loss(y_true, uncalibrated_probs))
    brier_after = float(brier_score_loss(y_true, calibrated_probs))
    logloss_before = float(log_loss(y_true, uncalibrated_probs))
    logloss_after = float(log_loss(y_true, calibrated_probs))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    calibrator_path = MODELS_DIR / "calibrator.pkl"
    joblib.dump(calibrator, calibrator_path)

    plot_reliability_diagram(
        y_true=y_true,
        uncalibrated_probs=uncalibrated_probs,
        calibrated_probs=calibrated_probs,
        brier_before=brier_before,
        brier_after=brier_after,
    )

    optimal_cal_thr, _ = find_optimal_threshold(y_true, calibrated_probs)
    cal_metrics = evaluate_threshold_metrics(y_true, calibrated_probs, threshold=optimal_cal_thr)

    df_oof["calibrated_probability"] = calibrated_probs
    df_oof["calibrated_prediction"] = (calibrated_probs >= optimal_cal_thr).astype(int)
    df_oof.to_parquet(oof_path, index=False)

    return {
        "brier_before": brier_before,
        "brier_after": brier_after,
        "logloss_before": logloss_before,
        "logloss_after": logloss_after,
        "optimal_calibrated_threshold": optimal_cal_thr,
        "calibrated_metrics": cal_metrics,
        "calibrator_path": calibrator_path,
    }


if __name__ == "__main__":
    results = run_probability_calibration()
    print("Калибровка вероятностей успешно завершена.")
    print(f"Brier Score: {results['brier_before']:.4f} -> {results['brier_after']:.4f}")
    print(f"LogLoss: {results['logloss_before']:.4f} -> {results['logloss_after']:.4f}")
    print(f"Оптимальный порог после калибровки: {results['optimal_calibrated_threshold']:.2f}")
    print(f"PR-AUC: {results['calibrated_metrics']['pr_auc']:.4f}")
    print(f"ROC-AUC: {results['calibrated_metrics']['roc_auc']:.4f}")
    print(f"Суммарный ущерб: {results['calibrated_metrics']['total_cost']:,.2f} $")
