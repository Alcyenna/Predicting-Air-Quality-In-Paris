from pathlib import Path

import pandas as pd


def main() -> None:
    p = Path(__file__).resolve().parents[1] / "data" / "external" / "referentiel-comptages-routiers.parquet"
    df = pd.read_parquet(p)
    print("PATH:", p)
    print("COLUMNS:", list(df.columns))
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()


