"""Скрипт проведения сравнительного исследования полезности признаков (Ablation Study)."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import (
    DATA_PROCESSED_DIR,
    RANDOM_SEED,
    RAW_CATEGORICAL_COLS,
    REPORTS_DIR,
    TARGET_COL,
)
from src.features import HRFeatureTransformer
from src.metrics import evaluate_threshold_metrics, find_optimal_threshold


def evaluate_cv_pipeline(
    df_dev: pd.DataFrame,
    use_feature_engineering: bool = False,
) -> dict[str, float]:
    """Оценивает модель на кросс-валидации с изоляцией трансформаций внутри каждого фолда."""
    n_folds = int(df_dev["fold"].nunique())
    oof_probs = np.zeros(len(df_dev))
    y_true = df_dev[TARGET_COL].values

    for fold_idx in range(n_folds):
        train_mask = df_dev["fold"] != fold_idx
        val_mask = df_dev["fold"] == fold_idx

        train_data = df_dev[train_mask].copy()
        val_data = df_dev[val_mask].copy()

        y_train = train_data[TARGET_COL].values
        y_val = val_data[TARGET_COL].values

        X_train = train_data.drop(columns=[TARGET_COL, "fold"])
        X_val = val_data.drop(columns=[TARGET_COL, "fold"])

        if use_feature_engineering:
            transformer = HRFeatureTransformer()
            transformer.fit(X_train)
            X_train = transformer.transform(X_train)
            X_val = transformer.transform(X_val)

        cat_features = [c for c in RAW_CATEGORICAL_COLS if c in X_train.columns]
        for c in cat_features:
            X_train[c] = X_train[c].astype(str).fillna("Missing")
            X_val[c] = X_val[c].astype(str).fillna("Missing")

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5.0,
            auto_class_weights="Balanced",
            random_seed=RANDOM_SEED + fold_idx,
            verbose=False,
        )

        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=40, verbose=False)
        val_preds_prob = model.predict_proba(val_pool)[:, 1]
        oof_probs[val_mask] = val_preds_prob

    opt_threshold, _ = find_optimal_threshold(y_true, oof_probs)
    final_metrics = evaluate_threshold_metrics(y_true, oof_probs, threshold=opt_threshold)

    return {
        "pr_auc": final_metrics["pr_auc"],
        "roc_auc": final_metrics["roc_auc"],
        "optimal_threshold": opt_threshold,
        "f1": final_metrics["f1"],
        "f2": final_metrics["f2"],
        "total_cost": final_metrics["total_cost"],
        "cost_reduction": final_metrics["cost_reduction"],
    }


def run_ablation_study() -> pd.DataFrame:
    """Запускает сравнение базового и обогащенного набора признаков на 5 фолдах."""
    dev_path = DATA_PROCESSED_DIR / "train_dev.parquet"
    df_dev = pd.read_parquet(dev_path)

    baseline_metrics = evaluate_cv_pipeline(df_dev, use_feature_engineering=False)
    enriched_metrics = evaluate_cv_pipeline(df_dev, use_feature_engineering=True)

    results = [
        {"Модель": "Базовая (сырые признаки, 28 фичей)", **baseline_metrics},
        {"Модель": "Обогащенная (+ 9 HR-признаков, 37 фичей)", **enriched_metrics},
    ]

    df_results = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "ablation_study.md"

    header = "| " + " | ".join(df_results.columns) + " |\n"
    separator = "| " + " | ".join(["---"] * len(df_results.columns)) + " |\n"
    rows = []
    for _, row in df_results.iterrows():
        formatted_row = []
        for col in df_results.columns:
            val = row[col]
            if isinstance(val, float):
                formatted_row.append(f"{val:.4f}")
            else:
                formatted_row.append(str(val))
        rows.append("| " + " | ".join(formatted_row) + " |\n")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(
            "# Результаты сравнительного исследования полезности признаков (Ablation Study)\n\n"
        )
        f.write(header + separator + "".join(rows))
        f.write("\n\n## Выводы:\n")
        pr_delta = enriched_metrics["pr_auc"] - baseline_metrics["pr_auc"]
        cost_savings = baseline_metrics["total_cost"] - enriched_metrics["total_cost"]
        f.write(f"- Прирост PR-AUC: {pr_delta:+.4f}\n")
        f.write(f"- Финансовая экономия для компании: {cost_savings:,.2f} $\n")

    return df_results


if __name__ == "__main__":
    df_res = run_ablation_study()
    print("Ablation Study успешно завершено:")
    print(df_res.to_string(index=False))
