from glob import glob

import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd

run_id = svd.get_run_id()
samplesz = -1
savepath = svd.get_dir(svd.processed_dir() / f"minibatch_tests_{samplesz}" / run_id)
model = lvd.LitGNN(
    methodlevel=False, nsampling=True, model="gat2layer", loss="ce", hdropout=0.2
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
# Uses old codebert embeddings (no attention mask)
run_id = "202108251105_5a6e846_update_train_code"  # gat2layer
run_id = "202108261224_1bd41cb_remove_print_line"  # mlp-only
run_id = "202108261332_d1af738_update_codebert_encoding"  # gat2layer v2
run_id = "202108261657_c42a446_load_everything_from_hparams"  # gat2layer 0.25 hdropout
# Uses new codebert embeddings (has attention mask)
run_id = "202108271123_8dd6708_update_joern_test_ids"  # mlp-only 0.2 hdropout
run_id = "202108271328_8c84ea4_fix_torch_cat_func_embedding"  # mlponly+femb

run_id = "202108271622_8c84ea4_fix_torch_cat_func_embedding"

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

print(model.res1vo)
print(model.res1)
print(model.res2)
print(model.res3)
print(model.res3vo)
print(model.res4)

# %% One run
# from importlib import reload

# reload(lvd)
# model = lvd.LitGNN(methodlevel=False, nsampling=True, model="mlponly+femb")
# # sample = next(iter(data.train_dataloader()))
# model(sample)
# # trainer.fit(model, data)

# # %%
# # Comments
# # I really appreciate your presentation as I am always annoyed at not being able to run research code. Not a question, but just wanted to mention a few other things I've found useful. VSCode is actually quite powerful - I handle all source control, linting, and testing (pytest) all within the ide. For people who work HPC, sometimes conda environments are not sufficient, and Docker is unavailable due to security concerns. Singularity is a great alternative.
# th.cat([th.Tensor([1, 2, 3]), th.Tensor([1, 2, 3])])
