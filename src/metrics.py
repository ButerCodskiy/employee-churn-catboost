"""Модуль расчета бизнес-метрик, матрицы затрат и оптимизации порога отсечения."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    COST_FN,
    COST_FP,
    COST_TN,
    COST_TP,
    FIGURES_DIR,
)


def calculate_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    cost_tp: float = COST_TP,
    cost_tn: float = COST_TN,
) -> float:
    """Вычисляет суммарные финансовые потери компании при заданном пороге классификации."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn * cost_fn + fp * cost_fp + tp * cost_tp + tn * cost_tn)


def evaluate_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    cost_tp: float = COST_TP,
    cost_tn: float = COST_TN,
) -> dict[str, float]:
    """Рассчитывает полный спектр метрик качества строго при едином фиксированном пороге."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total_cost = float(fn * cost_fn + fp * cost_fp + tp * cost_tp + tn * cost_tn)
    naive_loss = float(np.sum(y_true == 1) * cost_fn)
    cost_reduction = float(naive_loss - total_cost)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": total_cost,
        "cost_reduction": cost_reduction,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    cost_tp: float = COST_TP,
    cost_tn: float = COST_TN,
    step: float = 0.01,
) -> tuple[float, pd.DataFrame]:
    """Находит порог, минимизирующий суммарный ущерб бизнеса по сетке вероятностей."""
    thresholds = np.arange(0.01, 0.99 + step, step)
    records: list[dict[str, float]] = []

    for thr in thresholds:
        metrics = evaluate_threshold_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=thr,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            cost_tp=cost_tp,
            cost_tn=cost_tn,
        )
        records.append(metrics)

    df_results = pd.DataFrame(records)
    best_row = df_results.loc[df_results["total_cost"].idxmin()]
    optimal_threshold = float(best_row["threshold"])

    return optimal_threshold, df_results


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_threshold: float | None = None,
    save_path: Path | None = None,
) -> Path:
    """Строит кривую Precision-Recall с отображением базовой линии и оптимального порога."""
    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "pr_curve.png"

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline = np.mean(y_true)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#1f77b4", lw=2, label=f"CatBoost (PR-AUC = {pr_auc:.4f})")
    plt.axhline(
        y=baseline, color="gray", linestyle="--", lw=1.5, label=f"Базовый уровень ({baseline:.4f})"
    )

    if optimal_threshold is not None and len(thresholds) > 0:
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        opt_prec = precision[idx]
        opt_rec = recall[idx]
        plt.scatter(
            [opt_rec],
            [opt_prec],
            color="#d62728",
            s=100,
            zorder=5,
            label=f"Порог tau* = {optimal_threshold:.2f}",
        )

    plt.title("Кривая точности и полноты (Precision-Recall Curve)", fontsize=14, pad=12)
    plt.xlabel("Полнота (Recall)", fontsize=12)
    plt.ylabel("Точность (Precision)", fontsize=12)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def plot_cost_curve(
    df_thresholds: pd.DataFrame,
    optimal_threshold: float,
    save_path: Path | None = None,
) -> Path:
    """Отображает зависимость суммарного бизнес-ущерба от порога классификации."""
    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "cost_curve.png"

    min_cost = df_thresholds.loc[
        df_thresholds["threshold"] == optimal_threshold, "total_cost"
    ].values[0]

    plt.figure(figsize=(9, 6))
    plt.plot(
        df_thresholds["threshold"],
        df_thresholds["total_cost"] / 1000.0,
        color="#2ca02c",
        lw=2.5,
        label="Финансовые потери",
    )
    plt.axvline(
        x=optimal_threshold,
        color="#d62728",
        linestyle="--",
        lw=1.5,
        label=f"Оптимум tau* = {optimal_threshold:.2f}",
    )
    plt.scatter([optimal_threshold], [min_cost / 1000.0], color="#d62728", s=100, zorder=5)

    plt.title("Оптимизация порога по матрице затрат (HR Cost Function)", fontsize=14, pad=12)
    plt.xlabel("Порог классификации (Threshold)", fontsize=12)
    plt.ylabel("Суммарный ущерб бизнеса (тыс. $)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    save_path: Path | None = None,
) -> Path:
    """Визуализирует матрицу ошибок с абсолютными и процентными показателями."""
    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "confusion_matrix.png"

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    annot = np.array(
        [
            [
                f"TN: {cm[0, 0]}\n({cm[0, 0] / cm[0].sum():.1%})",
                f"FP: {cm[0, 1]}\n({cm[0, 1] / cm[0].sum():.1%})",
            ],
            [
                f"FN: {cm[1, 0]}\n({cm[1, 0] / cm[1].sum():.1%})",
                f"TP: {cm[1, 1]}\n({cm[1, 1] / cm[1].sum():.1%})",
            ],
        ]
    )

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues", cbar=False, annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"Матрица ошибок (порог = {threshold:.2f})", fontsize=14, pad=12)
    plt.xlabel("Предсказанный класс", fontsize=12)
    plt.ylabel("Истинный класс", fontsize=12)
    plt.xticks([0.5, 1.5], ["Лоялен (0)", "Уволился (1)"], fontsize=11)
    plt.yticks([0.5, 1.5], ["Лоялен (0)", "Уволился (1)"], fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path
