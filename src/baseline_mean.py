import argparse
from pathlib import Path

import pandas as pd


POLLUTANT_COLUMNS = [
    "valeur_NO2",
    "valeur_CO",
    "valeur_O3",
    "valeur_PM10",
    "valeur_PM25",
]


def generate_baseline_mean_submission(
    train_csv_path: Path, test_csv_path: Path, output_csv_path: Path
) -> None:
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Compute global mean per pollutant over the entire training set (ignore NaNs)
    means = {
        col: float(train_df[col].mean(skipna=True)) for col in POLLUTANT_COLUMNS
    }

    # Build submission by repeating the global means for each timestamp in test
    submission_df = pd.DataFrame({"id": test_df["id"]})
    for col in POLLUTANT_COLUMNS:
        submission_df[col] = means[col]

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate baseline submission using global mean per pollutant."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Directory containing train.csv and test.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "submissions"
            / "submission_baseline_mean.csv"
        ),
        help="Output CSV path for the submission",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_csv_path = data_dir / "train.csv"
    test_csv_path = data_dir / "test.csv"
    output_csv_path = Path(args.out)

    generate_baseline_mean_submission(train_csv_path, test_csv_path, output_csv_path)

    print(f"Saved baseline submission to: {output_csv_path}")


if __name__ == "__main__":
    main()


