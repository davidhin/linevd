from glob import glob
from importlib import reload

import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd

# %% TESTING
reload(lvd)

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

model = lvd.LitGNN()
data = lvd.BigVulDatasetLineVDDataModule(batch_size=1024, nsampling_hops=2)
trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")

result_dir = "minibatch_tests_-1"
ckpt_str = "lightning_logs/version_0/checkpoints/*.ckpt"
best_model = glob(str(svd.processed_dir() / result_dir / run_id / ckpt_str))[0]
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
