from importlib import reload

import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.dclass as svddc
import sastvd.helpers.hljs as hljs
import sastvd.helpers.rank_eval as svdhr
import sastvd.linevd as lvd
import torch as th
from tqdm import tqdm

# PDG+RAW, Gat1Layer
checkpoint = "raytune_-1/202109031655_f87dcf9_add_perfect_test/tune_linevd/train_linevd_2a3f5_00013_13_gatdropout=0.2,gnntype=gat,gtype=pdg+raw,hdropout=0.3,modeltype=gat2layer,stmtweight=10_2021-09-04_07-55-21/checkpoint_epoch=129-step=63310/checkpoint"

# Load modules
model = lvd.LitGNN()
datamodule_args = {"batch_size": 1024, "nsampling_hops": 2, "gtype": "pdg+raw"}
data = lvd.BigVulDatasetLineVDDataModule(**datamodule_args)
trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")

# %% Load best model
best_model = svd.processed_dir() / checkpoint
model = lvd.LitGNN.load_from_checkpoint(best_model, strict=False)
trainer.test(model, data)

# %% Print results
print(model.res1vo)
print(model.res1)
print(model.res2)
print(model.res2mt)
print(model.res2f)
print(model.res3)
print(model.res3vo)
print(model.res4)
model.plot_pr_curve()

# %% Find sample


def preds(model, data, vid):
    """Get predictions given model and ID and data."""
    id2idx = {v: k for k, v in data.test.idx2id.items()}
    idx = id2idx[vid]
    g = data.test[idx]
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
    line_preds = line_preds
    line_preds = {i[1] - 1: i[0] for i in line_preds}
    hljs.linevd_to_html(svddc.BigVulDataset.itempath(vid), line_preds, vulns)


# Finding suitable examples
temp_df = data.test.df[data.test.df.vul == 1]
temp_df = temp_df[(temp_df.before.str.len() > 500) & (temp_df.before.str.len() < 800)]
model.eval()
reload(hljs)
for i in tqdm(range(len(temp_df))):
    sample = temp_df.iloc[i]
    sorted_pred = [i[2] for i in preds(model, data, sample.id)]
    try:
        prec5 = svdhr.precision_at_k(sorted_pred, 5)
        if prec5 > 0.7:
            save_html_preds(sample.id, model, data)
    except Exception as E:
        print(E)
        continue
