import argparse
import logging
from pathlib import Path
import shutil
import sys
from datetime import datetime
import traceback
import numpy as np
import tqdm

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
    DeviceStatsMonitor,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
from code_gnn.models.flow_gnn.gin import FlowGNNModule
from sastvd.linevd import BigVulDatasetLineVDDataModule

from code_gnn.globals import all_datasets, all_models, project_root_dir, seed_all
from code_gnn.models import model_class_dict
from code_gnn.models.base_module import BaseModule
from code_gnn.models.periodic_checkpoint import PeriodicModelCheckpoint

logger = logging.getLogger()


def get_data_module(config):
    print("config =", config)
    data = BigVulDatasetLineVDDataModule(
        batch_size=config["batch_size"],
        # sample=100,
        methodlevel=False,
        # nsampling=True,
        # nsampling_hops=2,
        # gtype="pdg+raw",
        gtype="cfg",
        # splits="default",
        # feat="all",
        feat=config["feat"],
        # load_code=config["dataset_only"],
        cache_all=config["cache_all"],
        undersample=not config["no_undersample_graphs"],
        filter_cwe=config["filter_cwe"],
        sample_mode=config["sample_mode"],
        use_cache=not config["disable_cache"],
        train_workers=config["train_workers"],
        split=config["split"],
        seed=config["seed"],
    )
    # config["steps_per_epoch"] = int(len(data.train_dataloader()))

    # check is ok
    print("graph", data.train[0])
    print("graph 2nd time", data.train[0])
    print("graph data", data.train[0].ndata)
    try:
        if config["feat"].startswith("_ABS_DATAFLOW"):
            featname = "_ABS_DATAFLOW"
        else:
            featname = config["feat"]
        config["input_dim"] = data.train[0].ndata[featname].shape[1]
        print("shape", data.train[0].ndata[featname].shape)
        print("sum", data.train[0].ndata[featname].sum())
        print("sum no 1st dim", data.train[0].ndata[featname][:, 1:].sum())
    except Exception:
        print("error logging first example")
        traceback.print_exc()

    return data


def checkout_dataset(data, config):
    if config["feat"].startswith("_ABS_DATAFLOW"):
        featname = "_ABS_DATAFLOW"
    else:
        featname = config["feat"]
    for ds in (data.train, data.val, data.test):
        print(ds.partition, "examine")
        sums = []
        num_known = []
        num_unknown = []
        lens = []
        printed = 0
        for d in tqdm.tqdm(
                ds,
                total=len(ds),
                desc=ds.partition
            ):
            if d is None:
                continue
            if printed < 5:
                print(printed, d)
                print(d.ndata)
                printed += 1
            feats = d.ndata[featname]
            sums.append(feats.sum())
            num_known.append(feats[:, 1:].sum())
            num_unknown.append(feats[:, 0].sum())
            lens.append(feats.shape[0])

        num_unknown = np.array(num_unknown)
        lens = np.array(lens)
        print(np.average(lens), "average length")
        print(np.average(sums / lens), "average percentage CFG")
        print(np.average(num_known / lens), "average percentage known")
        print(np.average(num_unknown / lens), "average percentage unknown")
        print(sum(num_unknown > 0), "out of", len(num_unknown), "have any unknown nodes")


def train_single_model(data, args):
    """
    Train a single model
    """

    trainer = get_trainer(args)
    config = vars(args)

    model = config["model_class"](**config)
    if not config["skip_train"]:
        if config["test_every"]:
            trainer.fit(model, train_dataloaders=data.train_dataloader(), val_dataloaders=[data.val_dataloader(), data.test_dataloader()])
        else:
            trainer.fit(model, datamodule=data)
    if config["evaluation"]:
        if config["take_checkpoint"] == "best":
            ckpt = trainer.checkpoint_callback.best_model_path
            logger.info("loading checkpoint %s", ckpt)
            trainer.test(model=model, datamodule=data, ckpt_path=ckpt)
        elif config["resume_from_checkpoint"]:
            logger.info("loading checkpoint %s", config["resume_from_checkpoint"])
            if Path(config["resume_from_checkpoint"]).is_file():
                trainer.test(
                    model=model, datamodule=data, ckpt_path=config["resume_from_checkpoint"]
                )
            if Path(config["resume_from_checkpoint"]).is_dir():
                ckpts = list(Path(config["resume_from_checkpoint"]).glob("checkpoints/*.ckpt"))
                periodical_ckpts = []
                performance_ckpts = []
                other_ckpts = []
                for c in ckpts:
                    if c.name.startswith("periodical"):
                        periodical_ckpts.append(c)
                    elif c.name.startswith("performance"):
                        performance_ckpts.append(c)
                    else:
                        other_ckpts.append(c)
                
                periodical_ckpts = sorted(periodical_ckpts, key=lambda fp: int(str(fp.name).split("-")[1]))
                performance_ckpts = sorted(performance_ckpts, reverse=True, key=lambda fp: float(str(fp.name).split("-")[3].split("=")[1].split(".")[0]))
                
                ckpts = periodical_ckpts + performance_ckpts + other_ckpts
                logger.info("ckpts: %s", [c.name for c in ckpts])
                for ckpt in ckpts:
                    logger.info("loading checkpoint %s", ckpt)
                    trainer.test(model=model, datamodule=data, ckpt_path=ckpt)


def get_callbacks(config):
    callbacks = []

    ckpt_dir = base_dir / "checkpoints"
    logger.info(f"Checkpointing to {ckpt_dir}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="performance-{epoch:02d}-{step:02d}-{"
        + config["target_metric"]
        + ":02f}",
        monitor=config["target_metric"],
        mode="min",
        save_last=True,
        save_top_k=5,
        # verbose=True,
    )
    callbacks.append(checkpoint_callback)
    checkpoint_callback = PeriodicModelCheckpoint(
        dirpath=str(ckpt_dir),
        every=25,
    )
    callbacks.append(checkpoint_callback)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    try:
        gpu_stats = DeviceStatsMonitor()
        callbacks.append(gpu_stats)
    except pl.utilities.exceptions.MisconfigurationException:
        traceback.print_exc()
        pass

    if config["patience"] is not None:
        early_stopping_callback = EarlyStopping(
            monitor=config["target_metric"],
            mode="max",
            patience=config["patience"],
        )
        callbacks.append(early_stopping_callback)

    if config["profile"]:
        callbacks.append(DeviceStatsMonitor())

    return callbacks

def get_trainer(args):
    callbacks = get_callbacks(args)

    trainer = pl.Trainer(
        gpus=1 if config["cuda"] else 0,
        default_root_dir="storage/ptl",
        num_sanity_val_steps=0 if config["tune"] else 2,
        overfit_batches=1 if config["debug_overfit"] else 0,
        limit_train_batches=config["debug_train_batches"]
        if config["debug_train_batches"]
        else 1.0,
        # # https://forums.pytorchlightning.ai/t/validation-sanity-check/174/6
        # detect_anomaly=True,
        # callbacks=callbacks,
        # max_epochs=config["max_epochs"],
        # # default_root_dir=base_dir,  # Use checkpoint callback instead
        # # deterministic=True,  # RuntimeError: scatter_add_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
        # enable_checkpointing=True,
        # # profiler=profiler,
        # resume_from_checkpoint=config["resume_from_checkpoint"],
        # # track_grad_norm=2,
        # gradient_clip_val=config["gradient_clip_val"],
        # accumulate_grad_batches=config["accumulate_grad_batches"],
    )
    return pl.Trainer.from_argparse_args(args, callbacks=callbacks)


def parse_args():
    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # model
    # parser.add_argument(
    #     "--model",
    #     choices=all_models,
    #     help="short ID for the model type to train",
    #     required=True,
    # )
    # dataset
    # logging and reproducibility
    # parser.add_argument("--seed", type=int, default=0, help="random seed")
    # parser.add_argument(
    #     "--log_suffix",
    #     type=str,
    #     default="",
    #     help="suffix to append after log directory",
    # )
    # parser.add_argument(
    #     "-h", "--help", action="store_true", help="print this help message"
    # )
    # parser.add_argument(
    #     "--version", type=str, default=None, help="version ID to use for logging"
    # )
    # different run modes
    # parser.add_argument(
    #     "--dataset_only", action="store_true", help="only load the dataset, then exit"
    # )
    # parser.add_argument("--check_mode", action='store_true', help='check the dataset, then exit')
    # parser.add_argument(
    #     "--profile",
    #     action="store_true",
    #     help="run training under the profiler and report results at the end",
    # )
    # tuning options
    # parser.add_argument("--tune", action="store_true", help="tune hyperparameters")
    # parser.add_argument(
    #     "--resume", action="store_true", help="resume previous tune progress"
    # )
    # parser.add_argument(
    #     "--n_trials", type=int, default=50, help="how many trials to tune"
    # )
    # parser.add_argument(
    #     "--tune_timeout", type=int, default=60 * 60 * 24, help="time limit for tuning"
    # )
    # training options
    # parser.add_argument("--skip_train", action="store_true", help="skip training")
    # parser.add_argument(
    #     "--evaluation", action="store_true", help="do evaluation on test set"
    # )
    # parser.add_argument(
    #     "--no_undersample_graphs",
    #     action="store_true",
    #     help="undersample graphs as in LineVD",
    # )
    # parser.add_argument(
    #     "--debug_overfit", action="store_true", help="debug mode - overfit one batch"
    # )
    # parser.add_argument("--clean", action="store_true", help="clean old outputs")
    # parser.add_argument("--filter_cwe", nargs="+", help="CWE to filter examples")
    # parser.add_argument(
    #     "--target_metric", type=str, default="val_loss", help="metric to optimize for"
    # )
    # parser.add_argument(
    #     "--take_checkpoint",
    #     type=str,
    #     default="best",
    #     help="how to select checkpoint for evaluation",
    # )
    # parser.add_argument(
    #     "--resume_from_checkpoint", type=str, help="checkpoint file to resume from"
    # )
    # parser.add_argument(
    #     "--n_folds",
    #     type=int,
    #     default=1,
    #     help="number of cross-validation folds to run.",
    # )
    # parser.add_argument(
    #     "--max_epochs", type=int, default=250, help="max number of epochs to run."
    # )
    # parser.add_argument(
    #     "--patience",
    #     type=int,
    #     help="patience value to use for early stopping. Omit to disable early stopping.",
    # )
    # parser.add_argument(
    #     "--test_every", action="store_true", help="run test dataloader every epoch"
    # )
    # parser.add_argument("--roc_every", type=int, help="print ROC curve every n epochs.")
    # parser.add_argument(
    #     "--gradient_clip_val",
    #     type=float,
    #     default=None,
    #     help="Value to clip gradient norm",
    # )
    # parser.add_argument(
    #     "--use_lr_scheduler", type=str, help="use a learning rate scheduler"
    # )
    # parser.add_argument(
    #     "--accumulate_grad_batches",
    #     type=int,
    #     default=1,
    #     help="how many batches to accumulate gradients",
    # )
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    BaseModule.add_model_specific_args(parser)
    FlowGNNModule.add_model_specific_args(parser)

    # args, _ = parser.parse_known_args()

    # BaseModule.add_model_specific_args(parser)
    # args.model_class = model_class_dict[args.model]
    # args.model_class.add_model_specific_args(parser)

    args = parser.parse_args()
    # args.model_class = model_class_dict[args.model]
    args.cuda = torch.cuda.is_available()
    args.unique_id = get_unique_id(args)

    logger.info(f"args={args}")
    return args

def get_unique_id(args):
    unique_id = "_".join(
        map(
            str,
            (
                args.model,
                args.dataset,
                args.label_style,
                args.feat,
                args.node_limit,
                args.graph_limit,
                "noundersample" if args.no_undersample_graphs else "undersample",
                args.undersample_factor,
                args.filter,
                f"{args.learning_rate:f}".rstrip("0").rstrip("."),
                f"{args.weight_decay:f}".rstrip("0").rstrip("."),
                args.batch_size,
                args.seed,
                args.gradient_clip_val,
                args.use_lr_scheduler,
                args.accumulate_grad_batches,
            ),
        )
    )
    if args.filter_cwe:
        unique_id += "_filter_" + "_".join(args.filter_cwe)
    if args.model == "devign":
        unique_id += "_" + "_".join(
            map(str, (args.window_size, args.graph_embed_size, args.num_layers))
        )
    else:
        unique_id += "_" + "_".join(
            map(
                str,
                (
                    args.num_layers,
                    args.num_mlp_layers,
                    args.hidden_dim,
                    args.learn_eps,
                    args.final_dropout,
                    args.graph_pooling_type,
                    args.neighbor_pooling_type,
                ),
            )
        )
    if args.debug_overfit:
        unique_id += "_debug_overfit"
    return unique_id


def cli_main():
    args = parse_args()
    seed_all(args.seed)

    logger.info(f"gpus={torch.cuda.is_available()}, {torch.cuda.device_count()}")

    data = get_data_module()

    config = vars(args)
    if args.dataset_only:
        checkout_dataset(data, config)
    else:
        train_single_model(data, config)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    cli_main()
