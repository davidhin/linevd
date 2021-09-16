from glob import glob

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.dclass as svddc
import sastvd.helpers.hljs as hljs
import sastvd.helpers.rank_eval as svdhr
import sastvd.linevd as lvd
import torch as th
from ray.tune import Analysis
from tqdm import tqdm


def preds(model, datapartition, vid):
    """Get predictions given model and ID and data."""
    id2idx = {v: k for k, v in datapartition.idx2id.items()}
    idx = id2idx[vid]
    g = datapartition[idx]
    ret_logits = model(g, test=True)
    line_ranks = th.nn.functional.softmax(ret_logits[0], dim=1)[:, 1]
    line_ranks = [i ** 3 for i in line_ranks]
    ret = list(zip(line_ranks, g.ndata["_LINE"], g.ndata["_VULN"]))
    ret = [[i[0].item(), i[1].item(), i[2].item()] for i in ret]
    ret = sorted(ret, key=lambda x: x[0], reverse=True)
    return ret


def save_html_preds(vid, model, data):
    """Save HTML visualised preds."""
    line_preds = preds(model, data, vid)
    vulns = [i[1] for i in line_preds if i[2] == 1]

    norm_vulns = []
    for idx, i in enumerate(line_preds[:5]):
        norm_vulns.append([0.7 - (0.15 * (idx)), i[1], i[2]])

    line_preds = {i[1] - 1: i[0] for i in norm_vulns}
    hljs.linevd_to_html(svddc.BigVulDataset.itempath(vid), line_preds, vulns)


if __name__ == "__main__":

    # Get analysis directories in storage/processed
    raytune_dirs = glob(str(svd.processed_dir() / "raytune_*_-1"))
    tune_dirs = [i for j in [glob(f"{rd}/*") for rd in raytune_dirs] for i in j]

    # Load full dataframe
    df_list = []
    for d in tune_dirs:
        df_list.append(Analysis(d).dataframe())
    df = pd.concat(df_list)
    df = df[df["config/splits"] == "default"]

    # Load results df
    results = glob(str(svd.outputs_dir() / "rq_results/*.csv"))
    res_df = pd.concat([pd.read_csv(i) for i in results])

    # Merge DFs and load best model
    mdf = df.merge(res_df[["trial_id", "checkpoint", "stmt_f1"]], on="trial_id")
    best = mdf.sort_values("stmt_f1", ascending=0).iloc[0]
    best_path = f"{best['logdir']}/{best['checkpoint']}/checkpoint"

    # Load modules
    model = lvd.LitGNN()
    datamodule_args = {
        "batch_size": 1024,
        "nsampling_hops": 2,
        "gtype": best["config/gtype"],
        "splits": best["config/splits"],
    }
    data = lvd.BigVulDatasetLineVDDataModule(**datamodule_args)
    trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")
    model = lvd.LitGNN.load_from_checkpoint(best_path, strict=False)
    trainer.test(model, data)

    # Check statement metrics
    print(model.res2mt)

    # %% Finding suitable examples
    datapartition = data.train
    cve_dict = svdd.bigvul_cve()
    stats = []
    for datapartition in [data.train, data.val, data.test]:

        temp_df = datapartition.df[datapartition.df.vul == 1]
        temp_df = temp_df[
            (temp_df.before.str.len() > 300) & (temp_df.before.str.len() < 1000)
        ]
        model.eval()
        for i in tqdm(range(len(temp_df))):
            sample = temp_df.iloc[i]
            p = preds(model, datapartition, sample.id)
            sorted_pred = [i[2] for i in p]
            try:
                prec5 = svdhr.precision_at_k(sorted_pred, 5)
                if prec5 > 0.5:
                    save_html_preds(sample.id, model, datapartition)
                    stats.append(
                        {
                            "vid": sample.id,
                            "cve": cve_dict[sample.id],
                            "p@5": prec5,
                            "gt_vul": sum(sorted_pred),
                            "len": len(sorted_pred),
                            "vul_ratio": sum(sorted_pred) / len(sorted_pred),
                        }
                    )
            except Exception as E:
                print(E)
                continue
    pd.DataFrame.from_records(stats).to_csv(
        svd.outputs_dir() / "visualise.csv", index=0
    )
