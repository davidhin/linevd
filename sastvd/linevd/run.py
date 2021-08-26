from glob import glob

import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd

run_id = svd.get_run_id()
samplesz = -1
savepath = svd.get_dir(svd.processed_dir() / f"minibatch_tests_{samplesz}" / run_id)
model = lvd.LitGNN(methodlevel=False, nsampling=True)

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
)
tuned = trainer.tune(model, data)
trainer.fit(model, data)

# %% TESTING
run_id = "202108251237_e14fc5f_codebert_only"
run_id = "202108251105_5a6e846_update_train_code"
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

model.res1vo
model.res1
model.res2
model.res3
model.res3vo
model.res4
