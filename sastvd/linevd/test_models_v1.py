from glob import glob
from importlib import reload

import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd

# %% TESTING
reload(lvd)
result_dir = "minibatch_tests_-1"
ckpt_str = "lightning_logs/version_0/checkpoints/*.ckpt"

# Codebert single-task (line)
run_id = "202108271123_8dd6708_update_joern_test_ids"

# Codebert + GAT single-task (method)
run_id = "202108300848_107b176_whitespace_update"

# Codebert + GAT single-task (line)
run_id = "202108301010_107b176_whitespace_update"

# Codebert + GAT (multitask) Weighted 1:30
run_id = "202108301139_8a89360_update_multitask_model"

# Codebert + GAT (multitask) Weighted 1:1
run_id = "202108271658_2ac4767_update_default_codebert"

# Codebert + GAT (multitask) Weighted 1:5
run_id = "202108301341_59b6c98_final_update_run_script"
run_id = "202108311702_bfb2547_explicit_set_num_workers_getgraphs"

# Codebert + GAT (multitask) Weighted 1:10
run_id = "202108301341_59b6c98_final_update_run_script"

# Codebert + GAT (multitask) Weighted 1:20
run_id = "202108301536_40f42a0_update_run_layout"

# Codebert (multitask) Weighted 1:20
run_id = "202108310923_bfb2547_explicit_set_num_workers_getgraphs"

# Codebert + GAT (multitask) Weighted 1:30
run_id = "202108311203_bfb2547_explicit_set_num_workers_getgraphs"

# Codebert + GAT8Head (multitask) Weighted 1:30
run_id = "202108311552_bfb2547_explicit_set_num_workers_getgraphs"

# Codebert + GAT (multitask) Weighted 1:5
run_id = "202109011035_ea3f849_update_raytune_settings"
result_dir = "raytune_-1"
ckpt_str = "lightning_logs/version_2/checkpoints/*.ckpt"

# Mystery (RayTune) Weighted 1:40
run_id = "202109011703_74795f7_add_method_level_running"
result_dir = "raytune_-1"
ckpt_str = "lightning_logs/version_4/checkpoints/*.ckpt"

# Mystery (RayTune) REPORTED IN PAPER (have to copy in first)
run_id = "202109011703_74795f7_add_method_level_running"
result_dir = "raytune_-1"
ckpt_str = "lightning_logs/version_12/checkpoints/*.ckpt"

# GCN (mystery)
best_model = (
    svd.processed_dir()
    / "raytune_-1/202109011703_74795f7_add_method_level_running/tune_linevd/train_linevd_e8ecf_00015_15_gatdropout=0.2,gnntype=gcn,hdropout=0.25,hfeat=512,modeltype=gat1layer,stmtweight=1_2021-09-02_03-43-07/checkpoint_epoch=129-step=62270/checkpoint"
)

# GCN 2 layer - Best Res2MT and best Res2F, but low Res3VO and Res1VO
best_model = (
    svd.processed_dir()
    / "raytune_-1/202109030833_a7bc051_add_method_level_testing/tune_linevd/train_linevd_090b7_00000_0_gatdropout=0.2,gnntype=gcn,hdropout=0.25,hfeat=512,modeltype=gat2layer,stmtweight=10_2021-09-03_08-33-50/checkpoint_epoch=66-step=32093/checkpoint"
)

# GCN 2 layer
best_model = (
    svd.processed_dir()
    / "raytune_-1/202109030833_a7bc051_add_method_level_testing/tune_linevd/train_linevd_090b7_00001_1_gatdropout=0.15,gnntype=gcn,hdropout=0.25,hfeat=512,modeltype=gat2layer,stmtweight=15_2021-09-03_08-33-50/checkpoint_epoch=62-step=30177/checkpoint"
)

# GAT Raytune (same performance as in paper)
best_model = (
    svd.processed_dir()
    / "raytune_-1/202109030833_a7bc051_add_method_level_testing/tune_linevd/train_linevd_090b7_00002_2_gatdropout=0.2,gnntype=gat,hdropout=0.25,hfeat=512,modeltype=gat1layer,stmtweight=15_2021-09-03_09-13-53/checkpoint_epoch=149-step=71850/checkpoint"
)

# GAT 1 layer PDG (could potentially be trained longer)
best_model = (
    svd.processed_dir()
    / "raytune_-1/202109031120_0de0e64_update_raytune_hparam_gtype/tune_linevd/train_linevd_495df_00000_0_gatdropout=0.2,gnntype=gat,gtype=pdg,hdropout=0.25,hfeat=512,loss=ce,modeltype=gat1layer,scea=0.5,stmtw_2021-09-03_11-20-16/checkpoint_epoch=112-step=55031/checkpoint"
)

# GCN 1 layer cfgcdg + raw
best_model = (
    svd.processed_dir()
    / "raytune_-1/202109031120_0de0e64_update_raytune_hparam_gtype/tune_linevd/train_linevd_495df_00001_1_gatdropout=0.15,gnntype=gcn,gtype=cfgcdg+raw,hdropout=0.3,hfeat=512,loss=ce,modeltype=gat1layer,scea=0._2021-09-03_11-20-16/checkpoint_epoch=98-step=47421/checkpoint"
)


best_model = (
    svd.processed_dir()
    / "raytune_-1/202109031120_0de0e64_update_raytune_hparam_gtype/tune_linevd/train_linevd_495df_00003_3_gatdropout=0.2,gnntype=gcn,gtype=pdg,hdropout=0.3,hfeat=512,loss=ce,modeltype=gat2layer,scea=0.5,stmtwe_2021-09-03_14-26-53/checkpoint_epoch=20-step=10227/checkpoint"
)

# Load modules
model = lvd.LitGNN(model="gat2layer", gnntype="gcn")
data = lvd.BigVulDatasetLineVDDataModule(
    batch_size=1024, nsampling_hops=2, gtype="cfgcdg"
)
trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")

# %% Load best model
best_model = glob(str(svd.processed_dir() / result_dir / run_id / ckpt_str))[0]
model = lvd.LitGNN.load_from_checkpoint(best_model, strict=False)
trainer.test(model, data)
model.hparams


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

import torch as th


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


# Finding suitable examples
temp_df = data.test.df[data.test.df.vul == 1]
temp_df = temp_df[(temp_df.before.str.len() > 700) & (temp_df.before.str.len() < 800)]

sample = temp_df.iloc[4]
preds(model, data, sample.id)

print(sample.before)
