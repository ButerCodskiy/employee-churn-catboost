"""Модульные тесты проверки второго этапа: расчет бизнес-метрик и оптимизация порога."""

from pathlib import Path

import numpy as np

from src.metrics import (
    calculate_cost,
    evaluate_threshold_metrics,
    find_optimal_threshold,
    plot_confusion_matrix_heatmap,
    plot_cost_curve,
    plot_precision_recall_curve,
)


def test_cost_calculation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    cost_fn = 1000.0
    cost_fp = 100.0
    cost_tp = 200.0
    cost_tn = 0.0

    cost_at_half = calculate_cost(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        cost_tp=cost_tp,
        cost_tn=cost_tn,
    )
    assert cost_at_half == 400.0, f"Ожидалась стоимость 400.0, получено {cost_at_half}"

    cost_at_one = calculate_cost(
        y_true=y_true,
        y_prob=y_prob,
        threshold=1.0,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        cost_tp=cost_tp,
        cost_tn=cost_tn,
    )
    assert cost_at_one == 2000.0, f"Ожидалась стоимость 2000.0, получено {cost_at_one}"


def test_metrics_consistency():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.05, 0.12, 0.35, 0.45, 0.55, 0.65, 0.75, 0.82, 0.91, 0.22])

    threshold = 0.5
    metrics = evaluate_threshold_metrics(y_true, y_prob, threshold=threshold)

    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    expected_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    assert np.isclose(metrics["precision"], expected_prec), "Несогласованность точности"
    assert np.isclose(metrics["recall"], expected_rec), "Несогласованность полноты"
    assert metrics["tp"] == tp, "Несовпадение истинноположительных исходов"
    assert metrics["fp"] == fp, "Несовпадение ложноположительных исходов"
    assert metrics["fn"] == fn, "Несовпадение ложноотрицательных исходов"
    assert metrics["tn"] == tn, "Несовпадение истинноотрицательных исходов"


def test_optimal_threshold_finding():
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.2, 500)
    y_prob = np.clip(y_true * 0.5 + np.random.normal(0.2, 0.15, 500), 0.01, 0.99)

    opt_thr, df_thr = find_optimal_threshold(y_true, y_prob)

    assert 0.01 <= opt_thr <= 0.99, f"Порог вышел за допустимый диапазон: {opt_thr}"
    min_cost = df_thr.loc[df_thr["threshold"] == opt_thr, "total_cost"].values[0]
    naive_loss = np.sum(y_true == 1) * 39000.0

    assert min_cost < naive_loss, "Оптимальная модель не превзошла наивный базовый убыток"


def test_visualization_generation(tmp_path: Path):
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.05, 0.12, 0.35, 0.45, 0.55, 0.65, 0.75, 0.82, 0.91, 0.22])

    opt_thr, df_thr = find_optimal_threshold(y_true, y_prob)

    pr_path = tmp_path / "pr_curve.png"
    cost_path = tmp_path / "cost_curve.png"
    cm_path = tmp_path / "cm.png"

    plot_precision_recall_curve(y_true, y_prob, optimal_threshold=opt_thr, save_path=pr_path)
    plot_cost_curve(df_thr, optimal_threshold=opt_thr, save_path=cost_path)
    plot_confusion_matrix_heatmap(y_true, y_prob, threshold=opt_thr, save_path=cm_path)

    assert pr_path.exists() and pr_path.stat().st_size > 0, "Файл кривой PR не сформирован"
    assert cost_path.exists() and cost_path.stat().st_size > 0, "Файл кривой затрат не сформирован"
    assert cm_path.exists() and cm_path.stat().st_size > 0, "Файл матрицы ошибок не сформирован"


if __name__ == "__main__":
    test_cost_calculation()
    print("[OK] test_cost_calculation")
    test_metrics_consistency()
    print("[OK] test_metrics_consistency")
    test_optimal_threshold_finding()
    print("[OK] test_optimal_threshold_finding")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_visualization_generation(Path(td))
    print("[OK] test_visualization_generation")
    print("Все тесты модуля бизнес-метрик успешно пройдены.")
