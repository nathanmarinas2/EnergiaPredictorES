"""Evalúa baselines en modo day-ahead estricto.

El objetivo del script es medir un escenario compatible con producción:
para predecir el día t solo se usan calendario y demanda observada hasta
t-1. Las variables de mercado, precios y generación del propio día quedan
fuera aunque estén presentes en el CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    add_lag_features,
    add_temporal_features,
    regularize_time_index,
)
from src.models.baseline import (
    evaluate_predictions,
    naive_forecast,
    prepare_data,
    seasonal_naive_forecast,
    select_feature_columns,
    train_lightgbm,
    train_xgboost,
)


def load_daily_market_csv(path: Path) -> pd.DataFrame:
    """Carga el CSV largo de mercado y crea el dataset diario."""
    df_long = pd.read_csv(path, parse_dates=["datetime"])
    df = df_long.pivot_table(
        index="datetime",
        columns="name",
        values="value",
        aggfunc="first",
    )
    df.columns.name = None

    rename_map = {
        "Demanda real": "total load actual",
        "Demanda programada PBF total": "total load forecast",
    }
    for old_name, new_name in list(rename_map.items()):
        for column in df.columns:
            if old_name.lower() in column.lower():
                rename_map[column] = new_name
                break
    df = df.rename(columns=rename_map)

    df.index = pd.to_datetime(df.index).normalize()
    df = df.groupby(level=0).first().sort_index()
    df = regularize_time_index(df, frequency="D")

    fill_columns = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"total load actual", "total load forecast"}
    ]
    df[fill_columns] = df[fill_columns].ffill()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/production_results.csv"),
    )
    args = parser.parse_args()

    df = load_daily_market_csv(args.input)
    df = add_temporal_features(df)
    df = add_lag_features(df, frequency="D")

    lag_columns = [
        c for c in df.columns
        if c.startswith("load_lag_") or c.startswith("load_rolling_")
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["total load actual", *lag_columns])

    target = "total load actual"
    train, val, test = prepare_data(df, target)
    y_test = test[target].to_numpy()
    results = [
        evaluate_predictions(
            y_test,
            naive_forecast(train, test, target),
            "Naive (día anterior)",
        ),
        evaluate_predictions(
            y_test,
            seasonal_naive_forecast(train, test, target),
            "Seasonal Naive (semana anterior)",
        ),
    ]

    xgb_model, xgb_features = train_xgboost(train, val, target)
    results.append(
        evaluate_predictions(
            y_test,
            xgb_model.predict(test[xgb_features]),
            "XGBoost",
        )
    )

    lgb_model, lgb_features = train_lightgbm(train, val, target)
    results.append(
        evaluate_predictions(
            y_test,
            lgb_model.predict(test[lgb_features]),
            "LightGBM",
        )
    )

    result_df = pd.DataFrame(results).sort_values("mape")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)

    manifest = args.output.with_name("production_feature_manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "target": target,
                "train_start": str(train.index.min()),
                "train_end": str(train.index.max()),
                "validation_start": str(val.index.min()),
                "validation_end": str(val.index.max()),
                "test_start": str(test.index.min()),
                "test_end": str(test.index.max()),
                "test_rows": len(test),
                "features": select_feature_columns(train, target),
                "excluded_same_day_sources": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(result_df.to_string(index=False))
    print(f"\nResultados: {args.output}")
    print(f"Features: {len(select_feature_columns(train, target))}")


if __name__ == "__main__":
    main()
