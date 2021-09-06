import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
from ray import tune
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)


def train_linevd(config, samplesz=-1, max_epochs=130, num_gpus=1, checkpoint_dir=None):
    """Wrap Pytorch Lightning to pass to RayTune."""
    model = lvd.LitGNN(
        hfeat=config["hfeat"],
        methodlevel=False,
        nsampling=True,
        model=config["modeltype"],
        loss=config["loss"],
        hdropout=config["hdropout"],
        gatdropout=config["gatdropout"],
        num_heads=4,
        multitask="linemethod",
        stmtweight=config["stmtweight"],
        gnntype=config["gnntype"],
        scea=config["scea"],
    )

    # Load data
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=config["batch_size"],
        sample=samplesz,
        methodlevel=False,
        nsampling=True,
        nsampling_hops=2,
        gtype=config["gtype"],
    )

    # # Train model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    metrics = ["train_loss", "val_loss", "val_auroc"]
    raytune_callback = TuneReportCallback(metrics, on="validation_end")
    rtckpt_callback = TuneReportCheckpointCallback("val_loss", on="validation_end")
    trainer = pl.Trainer(
        gpus=num_gpus,
        auto_lr_find=True,
        default_root_dir=savepath,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, raytune_callback, rtckpt_callback],
        max_epochs=max_epochs,
    )
    trainer.tune(model, data)
    trainer.fit(model, data)


# Hyperparameters
config = {
    "hfeat": tune.choice([256, 512]),
    "stmtweight": tune.choice([2, 5, 8, 10, 12, 15, 18, 20]),
    "hdropout": tune.choice([0.15, 0.2, 0.25, 0.3]),
    "gatdropout": tune.choice([0.1, 0.15, 0.2]),
    "modeltype": tune.choice(["gat1layer", "gat2layer", "mlponly"]),
    "gnntype": tune.choice(["gat", "gcn"]),
    "loss": tune.choice(["ce", "sce"]),
    "scea": tune.choice([0.4, 0.5, 0.6]),
    "gtype": tune.choice(["cfgcdg", "cfgcdg+raw", "pdg", "pdg+raw"]),
    "batch_size": tune.choice([256, 512, 1024]),
}

samplesz = -1
trainable = tune.with_parameters(train_linevd, samplesz=samplesz)
run_id = svd.get_run_id()
savepath = svd.get_dir(svd.processed_dir() / f"raytune_{samplesz}" / run_id)

analysis = tune.run(
    trainable,
    resources_per_trial={"cpu": 1, "gpu": 1},
    metric="val_loss",
    mode="min",
    config=config,
    num_samples=100,
    name="tune_linevd",
    local_dir=savepath,
    keep_checkpoints_num=2,
    checkpoint_score_attr="min-val_loss",
)
