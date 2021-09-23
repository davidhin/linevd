import os

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
from ray import tune
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)

os.environ["SLURM_JOB_NAME"] = "bash"


def train_ml(
    config, savepath, samplesz=-1, max_epochs=130, num_gpus=1, checkpoint_dir=None
):
    """Wrap Pytorch Lightning to pass to RayTune."""
    model = lvd.LitGNN(
        methodlevel=True,
        nsampling=False,
        model=config["modeltype"],
        embtype="glove",
        loss="ce",
        hdropout=config["hdropout"],
        gatdropout=config["gatdropout"],
        num_heads=4,
        multitask="line",
        stmtweight=1,
        gnntype=config["gnntype"],
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
        feat="glove",
    )

    # # Train model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    metrics = ["train_loss", "val_loss", "val_auroc"]
    raytune_callback = TuneReportCallback(metrics, on="validation_end")
    rtckpt_callback = TuneReportCheckpointCallback(metrics, on="validation_end")
    trainer = pl.Trainer(
        gpus=1,
        auto_lr_find=False,
        default_root_dir=savepath,
        num_sanity_val_steps=3,
        callbacks=[checkpoint_callback, raytune_callback, rtckpt_callback],
        max_epochs=max_epochs,
    )
    trainer.fit(model, data)

    # Save test results
    main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_methodonly")
    trainer.test(model, data, ckpt_path="best")
    res = [
        "methodonly",
        "methodonly",
        model.res1vo,
        model.res2mt,
        model.res2f,
        model.res3vo,
        model.res2,
        model.lr,
    ]
    mets = lvd.get_relevant_metrics(res)
    res_df = pd.DataFrame.from_records([mets])
    res_df.to_csv(str(main_savedir / svd.get_run_id()) + ".csv", index=0)

    # Save best
    # trainer.test(model, data, ckpt_path="best")
    # res = [
    #     "methodonly",
    #     "methodonly",
    #     model.res1vo,
    #     model.res2mt,
    #     model.res2f,
    #     model.res3vo,
    #     model.res2,
    #     model.lr,
    # ]
    # mets = lvd.get_relevant_metrics(res)
    # res_df = pd.DataFrame.from_records([mets])
    # res_df.to_csv(str(main_savedir / svd.get_run_id()) + ".best.csv", index=0)


config = {
    "gnntype": tune.choice(["gat", "gcn"]),
    "hdropout": tune.choice([0.1, 0.15, 0.2, 0.25]),
    "gatdropout": tune.choice([0.15, 0.2]),
    "modeltype": tune.choice(["gat1layer", "gat2layer"]),
}
samplesz = -1
run_id = svd.get_run_id()
sp = svd.get_dir(svd.processed_dir() / f"raytune_methodlevel_{samplesz}" / run_id)
trainable = tune.with_parameters(train_ml, samplesz=samplesz, savepath=sp)

analysis = tune.run(
    trainable,
    resources_per_trial={"cpu": 1, "gpu": 1},
    metric="val_loss",
    mode="min",
    config=config,
    num_samples=1000,
    name="tune_linevd",
    local_dir=sp,
    keep_checkpoints_num=2,
    checkpoint_score_attr="min-val_loss",
)
