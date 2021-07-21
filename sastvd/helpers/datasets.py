import pandas as pd
import sastvd as svd
import sastvd.helpers.git as svdg


def bigvul():
    """Read BigVul Data."""
    df = pd.read_csv(svd.external_dir() / "bigvul2020.csv.gzip", compression="gzip")
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"
    svdg.mp_code2diff(df)
    return df
