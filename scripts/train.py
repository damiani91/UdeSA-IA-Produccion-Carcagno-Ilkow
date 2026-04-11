"""Uso: python -m scripts.train --date 2024-12-01"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--raw-data", default="data/raw/produccion.csv")
    args = parser.parse_args()

    # 1. Feature Pipeline
    from scripts.populate_feature_store import (
        prepare_offline_store, apply_feast, populate_online_store,
    )
    print(f"=== Feature Pipeline (hasta {args.date}) ===")
    prepare_offline_store(up_to_date=args.date)
    apply_feast()
    populate_online_store()

    # 2. Training Pipeline
    from src.training_pipeline.train import train_model
    print("=== Training Pipeline ===")
    run_id = train_model(training_date=args.date)
    print(f"=== Completo. MLflow Run ID: {run_id} ===")


if __name__ == "__main__":
    main()
