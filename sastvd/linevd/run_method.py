import os

import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd

os.environ["SLURM_JOB_NAME"] = "bash"

run_id = svd.get_run_id()
samplesz = -1
savepath = svd.get_dir(svd.processed_dir() / f"methodlevel_{samplesz}" / run_id)

# Load model
model = lvd.LitGNN(
    methodlevel=True,
    nsampling=False,
    model="gat2layer",
    embtype="codebert",
    loss="ce",
    hdropout=0.3,
    gatdropout=0.2,
    mlpdropout=0.2,
    num_heads=4,
    multitask="line",
    stmtweight=1,
    gnntype="gat",
    lr=1e-4,
)

# Load data
data = lvd.BigVulDatasetLineVDDataModule(
    batch_size=32,
    sample=samplesz,
    methodlevel=True,
    nsampling=False,
    nsampling_hops=2,
    gtype="pdg+raw",
    splits="default",
)

# Train model
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
metrics = ["train_loss", "val_loss", "val_auroc"]
trainer = pl.Trainer(
    gpus=1,
    auto_lr_find=True,
    default_root_dir=savepath,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback],
    max_epochs=1000,
)
trainer.tune(model, data)
trainer.fit(model, data)
