import os
import time
from glob import glob
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
from ray.tune import Analysis


def main(config, df):
    """Get test results."""
    main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_new")
    df_gtype = df[
        (df["config/gtype"] == config["config/gtype"])
        & (df["config/splits"] == config["config/splits"])
        & (df["config/embtype"] == config["config/embtype"])
    ]

    skipall = True
    for row in df_gtype.itertuples():
        chkpt_list = glob(row.logdir + "/checkpoint_*")
        chkpt_list = [i + "/checkpoint" for i in chkpt_list]
        for chkpt in chkpt_list:
            chkpt_info = Path(chkpt).parent.name
            chkpt_res_path = main_savedir / f"{row.trial_id}_{chkpt_info}.csv"
            if not os.path.exists(chkpt_res_path):
                skipall = False
                break
    if skipall:
        return

    hparam_cols = df_gtype.columns[df_gtype.columns.str.contains("config")].tolist()
    hparam_cols += ["experiment_id", "logdir"]
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=1024,
        nsampling_hops=2,
        gtype=config["config/gtype"],
        splits=config["config/splits"],
        feat=config["config/embtype"],
    )
    for row in df_gtype.itertuples():
        chkpt_list = glob(row.logdir + "/checkpoint_*")
        chkpt_list = [i + "/checkpoint" for i in chkpt_list]
        try:
            for chkpt in chkpt_list:
                chkpt_info = Path(chkpt).parent.name
                chkpt_res_path = main_savedir / f"{row.trial_id}_{chkpt_info}.csv"
                if os.path.exists(chkpt_res_path):
                    continue
                # Load model and test
                model = lvd.LitGNN()
                model = lvd.LitGNN.load_from_checkpoint(chkpt, strict=False)
                trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")
                trainer.test(model, data)
                res = [
                    row.trial_id,
                    chkpt_info,
                    model.res1vo,
                    model.res2mt,
                    model.res2f,
                    model.res3vo,
                    model.res2,
                    model.lr,
                ]
                # Save DF
                mets = lvd.get_relevant_metrics(res)
                hparams = df[df.trial_id == res[0]][hparam_cols].to_dict("records")[0]
                res_df = pd.DataFrame.from_records([{**mets, **hparams}])
                res_df.to_csv(chkpt_res_path, index=0)
        except Exception as E:
            print(E)


if __name__ == "__main__":

    while True:
        try:
            # Get analysis directories in storage/processed
            raytune_dirs = glob(str(svd.processed_dir() / "raytune_*_-1"))
            tune_dirs = [i for j in [glob(f"{rd}/*") for rd in raytune_dirs] for i in j]

            # Load full dataframe
            df_list = []
            for d in tune_dirs:
                df_list.append(Analysis(d).dataframe())
            df = pd.concat(df_list)

            # Get configurations
            if "config/splits" not in df.columns:
                df["config/splits"] = "default"
            if "config/embtype" not in df.columns:
                df["config/embtype"] = "codebert"
            configs = df[["config/gtype", "config/splits", "config/embtype"]]
            configs = configs.drop_duplicates().to_dict("records")

            # Start testing
            for config in configs:
                main(config, df)
        except Exception as E:
            print(E)
            pass

        # Sleep
        print("Sleeping...")
        time.sleep(60)
