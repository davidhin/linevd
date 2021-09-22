import pickle as pkl
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.dclass as svddc
import sastvd.helpers.hljs as hljs
import sastvd.linevd as lvd
import torch as th
from ray.tune import Analysis


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
    print("RESRANK1")
    print(model.res1vo)
    print("RES2MT")
    print(model.res2mt)
    print("RESF")
    print(model.res2f)
    print("RESRANK")
    print(model.res3vo)
    print("RESLINE")
    print(model.res2)

    # Get first-rank lines
    vulns = [i for i in model.all_funcs if max(i[1]) == 1]
    vulns = [i for i in vulns if i[2].max() == 1]

    def get_fr(v):
        """Get first vuln pred helper."""
        zipped = list(zip([i[1] for i in v[0]], v[1]))
        zipped.sort(reverse=True, key=lambda x: x[0])
        for rank, i in enumerate(zipped):
            if i[1] == 1:
                return rank + 1

    histogram_data = [get_fr(v) for v in vulns]
    num_bins = 90
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    plt.hist(histogram_data, bins=num_bins)
    savedir = svd.get_dir(svd.outputs_dir() / "mfrplots")
    plt.savefig(savedir / "mfrhist.pdf", bbox_inches="tight")
    with open(savedir / "histdata.pkl", "wb") as f:
        pkl.dump(histogram_data, f)
    pkl.load(open(savedir / "histdata.pkl", "rb"))
