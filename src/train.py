"""Модуль обучения моделей CatBoost на стратифицированной кросс-валидации."""

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RAW_CATEGORICAL_COLS,
    TARGET_COL,
)
from src.features import HRFeatureTransformer
from src.metrics import (
    evaluate_threshold_metrics,
    find_optimal_threshold,
    plot_confusion_matrix_heatmap,
    plot_cost_curve,
    plot_precision_recall_curve,
)


def train_fold_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_features: list[str],
    fold_idx: int,
    depth: int = 4,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 5.0,
    iterations: int = 600,
    early_stopping_rounds: int = 50,
) -> tuple[CatBoostClassifier, np.ndarray, dict[str, list[float]]]:
    """Обучает классификатор CatBoost на конкретном обучающем фолде с валидацией."""
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        auto_class_weights="Balanced",
        eval_metric="PRAUC",
        random_seed=RANDOM_SEED + fold_idx,
        verbose=False,
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    val_probs = model.predict_proba(val_pool)[:, 1]
    evals_result = model.get_evals_result()

    return model, val_probs, evals_result


def plot_cv_learning_curves(
    evals_history: list[dict[str, list[float]]],
    save_path: Path | None = None,
) -> Path:
    """Строит кривые обучения PR-AUC по фолдам кросс-валидации."""
    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "learning_curves.png"

    plt.figure(figsize=(9, 6))

    for idx, evals in enumerate(evals_history):
        learn_prauc = evals.get("learn", {}).get("PRAUC", [])
        val_prauc = evals.get("validation", {}).get("PRAUC", [])

        if val_prauc:
            plt.plot(val_prauc, lw=1.5, alpha=0.85, label=f"Фолд {idx} (Val)")
        if learn_prauc:
            plt.plot(learn_prauc, lw=1.0, linestyle="--", alpha=0.4)

    plt.title("Кривые обучения CatBoost на кросс-валидации (Метрика: PR-AUC)", fontsize=14, pad=12)
    plt.xlabel("Номер итерации", fontsize=12)
    plt.ylabel("PR-AUC", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def run_cross_validation_training(
    depth: int = 4,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 5.0,
    iterations: int = 600,
    early_stopping_rounds: int = 50,
) -> dict[str, Any]:
    """Выполняет полный цикл обучения ансамбля фолдов CatBoost и расчет OOF-метрик."""
    dev_path = DATA_PROCESSED_DIR / "train_dev.parquet"
    if not dev_path.exists():
        raise FileNotFoundError(f"Файл {dev_path} не найден. Требуется выполнить этап 1.")

    df_dev = pd.read_parquet(dev_path)
    n_folds = int(df_dev["fold"].nunique())

    oof_probs = np.zeros(len(df_dev))
    y_true = df_dev[TARGET_COL].values

    models: list[CatBoostClassifier] = []
    transformers: list[HRFeatureTransformer] = []
    evals_history: list[dict[str, list[float]]] = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for fold_idx in range(n_folds):
        train_mask = df_dev["fold"] != fold_idx
        val_mask = df_dev["fold"] == fold_idx

        train_data = df_dev[train_mask].copy()
        val_data = df_dev[val_mask].copy()

        y_train = train_data[TARGET_COL].values
        y_val = val_data[TARGET_COL].values

        X_train = train_data.drop(columns=[TARGET_COL, "fold"])
        X_val = val_data.drop(columns=[TARGET_COL, "fold"])

        transformer = HRFeatureTransformer()
        transformer.fit(X_train)
        X_train_trans = transformer.transform(X_train)
        X_val_trans = transformer.transform(X_val)

        cat_features = [c for c in RAW_CATEGORICAL_COLS if c in X_train_trans.columns]
        for c in cat_features:
            X_train_trans[c] = X_train_trans[c].astype(str).fillna("Missing")
            X_val_trans[c] = X_val_trans[c].astype(str).fillna("Missing")

        model, val_probs, evals_result = train_fold_model(
            X_train=X_train_trans,
            y_train=y_train,
            X_val=X_val_trans,
            y_val=y_val,
            cat_features=cat_features,
            fold_idx=fold_idx,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
        )

        oof_probs[val_mask] = val_probs
        models.append(model)
        transformers.append(transformer)
        evals_history.append(evals_result)

        model_path = MODELS_DIR / f"catboost_fold_{fold_idx}.cbm"
        model.save_model(str(model_path))

    transformer_path = MODELS_DIR / "feature_transformer.pkl"
    joblib.dump(transformers[0], transformer_path)

    plot_cv_learning_curves(evals_history)

    optimal_thr, df_thr = find_optimal_threshold(y_true, oof_probs)
    final_metrics = evaluate_threshold_metrics(y_true, oof_probs, threshold=optimal_thr)

    plot_precision_recall_curve(y_true, oof_probs, optimal_threshold=optimal_thr)
    plot_cost_curve(df_thr, optimal_threshold=optimal_thr)
    plot_confusion_matrix_heatmap(y_true, oof_probs, threshold=optimal_thr)

    df_oof = df_dev[["fold", TARGET_COL]].copy()
    df_oof["oof_probability"] = oof_probs
    df_oof["oof_prediction"] = (oof_probs >= optimal_thr).astype(int)
    df_oof.to_parquet(DATA_PROCESSED_DIR / "oof_predictions.parquet", index=False)

    return {
        "models": models,
        "optimal_threshold": optimal_thr,
        "metrics": final_metrics,
        "oof_probabilities": oof_probs,
    }


if __name__ == "__main__":
    results = run_cross_validation_training()
    metrics = results["metrics"]
    print("Обучение кросс-валидации успешно завершено.")
    print(f"Оптимальный порог: {results['optimal_threshold']:.2f}")
    print(f"OOF PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"OOF ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"OOF F1-Score: {metrics['f1']:.4f}")
    print(f"OOF Recall: {metrics['recall']:.4f}")
    print(f"OOF Precision: {metrics['precision']:.4f}")
    print(f"Суммарный ущерб бизнеса: {metrics['total_cost']:,.2f} $")
