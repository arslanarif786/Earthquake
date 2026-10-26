"""Part 2, Task D: train, evaluate, and save an earthquake model."""
import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split

from . import config
from .evaluate import (
    best_threshold,
    classification_report_dict,
    cross_validate_model,
    regression_report,
)
from .models import make_classifier, make_regressor


def load_data(db_path=None):
    """Load the model-ready quakes table created by ``quakes.etl``."""
    path = Path(db_path) if db_path is not None else config.PROCESSED_DIR / "quakes.db"
    if not path.exists():
        raise FileNotFoundError(
            f"Earthquake database not found at {path}. Run `python -m quakes.etl --mode batch` first."
        )
    with sqlite3.connect(path) as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='quakes'", connection
        )
        if tables.empty:
            raise ValueError(f"No 'quakes' table found in {path}. Run the ETL command first.")
        df = pd.read_sql_query('SELECT * FROM "quakes"', connection)
    if df.empty:
        raise ValueError(f"The 'quakes' table in {path} is empty")
    required = {"mag", config.TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"The quakes table is missing required columns: {sorted(missing)}")
    return df


def _positive_probability(pipe, X):
    probabilities = pipe.predict_proba(X)
    classes = list(pipe.named_steps["model"].classes_)
    return probabilities[:, classes.index(1)] if 1 in classes else probabilities[:, 0] * 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an earthquake prediction model")
    parser.add_argument("--task", choices=("regression", "classification"), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--balanced", action="store_true", help="Use balanced class weights for logistic regression")
    parser.add_argument("--db", default=None, help="Optional SQLite database path")
    args = parser.parse_args()

    df = load_data(args.db)
    if args.task == "regression":
        model_kind = args.model or "linear"
        target = "mag"
        X = df.drop(columns=["mag", config.TARGET], errors="ignore")
        y = df[target]
        scoring = "r2"
        model = make_regressor(model_kind, degree=args.degree, alpha=args.alpha)
        baseline = make_regressor("dummy")
    else:
        model_kind = args.model or "logistic"
        target = config.TARGET
        X = df.drop(columns=["mag", config.TARGET], errors="ignore")
        y = df[target].astype(int)
        scoring = "f1"
        weight = "balanced" if args.balanced else None
        model = make_classifier(model_kind, class_weight=weight)
        baseline = make_classifier("dummy")

    stratify = y if args.task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=stratify
    )

    baseline.fit(X_train, y_train)
    model.fit(X_train, y_train)
    report = {
        "task": args.task,
        "model": model_kind,
        "target": target,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": int(config.SEED),
    }

    if args.task == "regression":
        report["baseline_train"] = regression_report(y_train, baseline.predict(X_train))
        report["model_train"] = regression_report(y_train, model.predict(X_train))
        report["baseline_test"] = regression_report(y_test, baseline.predict(X_test))
        report["model_test"] = regression_report(y_test, model.predict(X_test))
        report["cross_validation"] = cross_validate_model(
            make_regressor(model_kind, degree=args.degree, alpha=args.alpha),
            X_train, y_train, scoring=scoring, cv=5,
        )
    else:
        baseline_proba = _positive_probability(baseline, X_test)
        report["baseline_test"] = classification_report_dict(
            y_test, baseline.predict(X_test), baseline_proba
        )
        oof_proba = cross_val_predict(
            make_classifier(model_kind, class_weight=weight),
            X_train, y_train, cv=5, method="predict_proba",
        )
        # Determine which probability column represents class 1 from the training labels.
        threshold = best_threshold(y_train, oof_proba[:, list(sorted(y_train.unique())).index(1)])
        model_proba = _positive_probability(model, X_test)
        model_predictions = (model_proba >= threshold).astype(int)
        report["threshold_from_training_oof"] = float(threshold)
        report["model_test"] = classification_report_dict(y_test, model_predictions, model_proba)
        report["cross_validation"] = cross_validate_model(
            make_classifier(model_kind, class_weight=weight),
            X_train, y_train, scoring=scoring, cv=5,
        )

    config.ROOT.joinpath("models").mkdir(parents=True, exist_ok=True)
    config.ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    model_path = config.ROOT / "models" / f"{args.task}_{model_kind}.joblib"
    report_path = config.ROOT / "reports" / f"{args.task}_{model_kind}_report.json"
    joblib.dump(model, model_path)
    report["model_path"] = str(model_path.relative_to(config.ROOT))
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
