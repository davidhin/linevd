from glob import glob
from importlib import reload

import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd

run_id = svd.get_run_id()
samplesz = -1
savepath = svd.get_dir(svd.processed_dir() / f"minibatch_tests_{samplesz}" / run_id)
model = lvd.LitGNN(
    methodlevel=False,
    nsampling=True,
    model="gat2layer",
    loss="ce",
    hdropout=0.2,
    multitask="linemethod",
    stmtweight=5,
)

# Load data
data = lvd.BigVulDatasetLineVDDataModule(
    batch_size=1024,
    sample=samplesz,
    methodlevel=False,
    nsampling=True,
    nsampling_hops=2,
)

# # Train model
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
trainer = pl.Trainer(
    gpus=1,
    auto_lr_find=True,
    default_root_dir=savepath,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback],
    max_epochs=20000,
)
tuned = trainer.tune(model, data)
trainer.fit(model, data)

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

best_model = glob(
    str(
        svd.processed_dir()
        / f"minibatch_tests_{samplesz}"
        / run_id
        / "lightning_logs/version_0/checkpoints/*.ckpt"
    )
)[0]
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
