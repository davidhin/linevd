import argparse
import logging
import shutil
import sys
from datetime import datetime
import traceback
import pandas as pd
import tqdm

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sastvd.linevd import BigVulDatasetLineVDDataModule

from code_gnn.globals import all_datasets, all_models, project_root_dir, seed_all
from code_gnn.models import model_class_dict
from code_gnn.models.base_module import BaseModule
from code_gnn.models.periodic_checkpoint import PeriodicModelCheckpoint

logger = logging.getLogger()


def train_single_model(config):
    """
    Train a single model
    """

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
    )

    if config["dataset_only"]:
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
        return

    trainer = get_trainer(config)
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

    # if config["check_mode"]:
    #     blacklist = []
    #     it = iter(data.train)
    #     for i in tqdm.tqdm(range(len(data.train)), desc="check train"):
    #         try:
    #             next(it)
    #         except Exception:
    #             traceback.print_exc()
    #             print("blacklist", i, data.train.idx2id[i])
    #             blacklist.append(data.train.idx2id[i])
    #     it = iter(data.val)
    #     for i in tqdm.tqdm(range(len(data.val)), desc="check val"):
    #         try:
    #             next(it)
    #         except Exception:
    #             traceback.print_exc()
    #             print("blacklist", i, data.val.idx2id[i])
    #             blacklist.append(data.val.idx2id[i])
    #     it = iter(data.test)
    #     for i in tqdm.tqdm(range(len(data.test)), desc="check test"):
    #         try:
    #             next(it)
    #         except Exception:
    #             traceback.print_exc()
    #             print("blacklist", i, data.test.idx2id[i])
    #             blacklist.append(data.test.idx2id[i])
    #     with open("blacklist.txt", "w") as f:
    #         f.write("\n".join(blacklist))
    #     return

    model = config["model_class"](**config)
    if not config["skip_train"]:
        trainer.fit(model, datamodule=data)
    if config["evaluation"]:
        if config["resume_from_checkpoint"]:
            logger.info("loading checkpoint %s", config["resume_from_checkpoint"])
            trainer.test(
                model=model, datamodule=data, ckpt_path=config["resume_from_checkpoint"]
            )
        elif config["take_checkpoint"] == "best":
            ckpt = trainer.checkpoint_callback.best_model_path
            logger.info("loading checkpoint %s", ckpt)
            trainer.test(model=model, datamodule=data, ckpt_path=ckpt)
        else:
            ckpts = list(config["base_dir"].glob("checkpoints/periodical-*.ckpt"))
            logger.info(
                "unsorted: %s", str([int(str(fp.name).split("-")[1]) for fp in ckpts])
            )
            ckpts = sorted(ckpts, key=lambda fp: int(str(fp.name).split("-")[1]))
            logger.info(
                "sorted: %s", str([int(str(fp.name).split("-")[1]) for fp in ckpts])
            )
            for ckpt in ckpts:
                logger.info("loading checkpoint %s", ckpt)
                trainer.test(model=model, datamodule=data, ckpt_path=ckpt)


def get_trainer(config):
    callbacks = []

    # Fixed by following example https://gist.github.com/Crissman/9cea7f22939a8816081f31afb1c8ab03
    if config["tune"]:
        base_dir = (
            project_root_dir
            / "logs_tune"
            / (config["unique_id"] + config["log_suffix"])
            / f'trial_{config["tune_trial"].number}'
        )
    elif config["n_folds"] > 1:
        base_dir = (
            project_root_dir
            / "logs_crossval"
            / (config["unique_id"] + config["log_suffix"])
            / f'fold_{config["fold_idx"]}'
        )
    elif config["debug_overfit"]:
        base_dir = (
            project_root_dir
            / "logs"
            / (config["unique_id"] + config["log_suffix"])
            / "overfit_batch"
        )
    # elif config["evaluation"]:
    #     base_dir = (
    #             project_root_dir
    #             / "logs"
    #             / (config["unique_id"] + config["log_suffix"])
    #             / 'evaluation'
    #     )
    #     assert config["resume_from_checkpoint"] is not None
    else:
        base_dir = (
            project_root_dir
            / "logs"
            / (config["unique_id"] + config["log_suffix"])
            / "default"
        )
    config["base_dir"] = base_dir

    if base_dir.exists():
        if config["clean"]:
            if config["resume_from_checkpoint"] is not None or config["evaluation"]:
                logger.warning(
                    f"Told to clean {base_dir}, but also to load. Skipping --clean."
                )
            else:
                logger.info(f"Cleaning old results from {base_dir}...")
                shutil.rmtree(base_dir)
        elif config["resume_from_checkpoint"] is None:
            raise NotImplementedError(f"Please clear old results from {base_dir}")

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
        # verbose=True,
    )
    callbacks.append(checkpoint_callback)
    checkpoint_callback = PeriodicModelCheckpoint(
        dirpath=str(ckpt_dir),
        every=25,
    )
    callbacks.append(checkpoint_callback)

    tb_logger = TensorBoardLogger(str(base_dir), version="", name="")

    if config["patience"] is not None:
        early_stopping_callback = EarlyStopping(
            monitor=config["target_metric"],
            mode="max",
            patience=config["patience"],
        )
        callbacks.append(early_stopping_callback)

    if config["profile"]:
        callbacks.append(DeviceStatsMonitor())
    # if "tune_trial" in config:
    #     callbacks.append(
    #         PyTorchLightningPruningCallback(
    #             config["tune_trial"], monitor=config["target_metric"]
    #         )
    #     )

    # profiler = pl.profiler.AdvancedProfiler(filename="profile.txt")

    trainer = pl.Trainer(
        gpus=1 if config["cuda"] else 0,
        num_sanity_val_steps=0 if config["tune"] else 2,
        overfit_batches=1 if config["debug_overfit"] else 0,
        limit_train_batches=config["debug_train_batches"]
        if config["debug_train_batches"]
        else 1.0,
        # https://forums.pytorchlightning.ai/t/validation-sanity-check/174/6
        detect_anomaly=True,
        callbacks=callbacks,
        logger=tb_logger,
        max_epochs=config["max_epochs"],
        # default_root_dir=base_dir,  # Use checkpoint callback instead
        # deterministic=True,  # RuntimeError: scatter_add_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
        enable_checkpointing=True,
        # profiler=profiler,
        resume_from_checkpoint=config["resume_from_checkpoint"],
    )
    return trainer


def main(config):
    logger.info(f"config={config}")
    seed_all(config["seed"])

    config["cuda"] = torch.cuda.is_available()
    logger.info(f"gpus={torch.cuda.is_available()}, {torch.cuda.device_count()}")

    seed_everything(config["seed"], workers=True)
    if config["tune"]:
        pass
    else:
        train_single_model(config)


def log_results(study):
    logger.info(f"Number of finished trials: {len(study.trials)}")
    trial = study.best_trial
    logger.info(f"Best trial: {trial.number}")
    logger.info(f"\tTarget: {trial.value}")
    logger.info(f"\tHyperparameters:")
    for key, value in trial.params.items():
        logger.info("\t{}: {}".format(key, value))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # model
    parser.add_argument(
        "--model",
        choices=all_models,
        help="short ID for the model type to train",
        required=True,
    )
    # dataset
    parser.add_argument(
        "--dataset",
        choices=all_datasets,
        required=True,
        help="short ID for the dataset to train on",
    )
    parser.add_argument("--feat", required=True, help="node features to use")
    parser.add_argument(
        "--node_limit", type=int, help="upper limit to the number of nodes in a graph"
    )
    parser.add_argument(
        "--graph_limit", type=int, help="upper limit to the number of graphs to parse"
    )
    parser.add_argument(
        "--filter", type=str, help="filter data to a certain persuasion", default=""
    )
    parser.add_argument(
        "--label_style", type=str, help="use node or graph labels", default="graph"
    )
    parser.add_argument(
        "--debug_train_batches", type=int, help="debug mode - train with n batches"
    )
    parser.add_argument(
        "--undersample_factor", type=float, help="factor to undersample majority class"
    )
    parser.add_argument(
        "--cache_all", action="store_true", help="cache all items in memory"
    )
    parser.add_argument(
        "--disable_cache", action="store_true", help="use cached files for dataset"
    )
    parser.add_argument(
        "--sample_mode", action="store_true", help="load only sample of dataset"
    )
    # logging and reproducibility
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--log_suffix",
        type=str,
        default="",
        help="suffix to append after log directory",
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="print this help message"
    )
    parser.add_argument(
        "--version", type=str, default=None, help="version ID to use for logging"
    )
    # different run modes
    parser.add_argument(
        "--dataset_only", action="store_true", help="only load the dataset, then exit"
    )
    # parser.add_argument("--check_mode", action='store_true', help='check the dataset, then exit')
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run training under the profiler and report results at the end",
    )
    # tuning options
    parser.add_argument("--tune", action="store_true", help="tune hyperparameters")
    parser.add_argument(
        "--resume", action="store_true", help="resume previous tune progress"
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="how many trials to tune"
    )
    parser.add_argument(
        "--tune_timeout", type=int, default=60 * 60 * 24, help="time limit for tuning"
    )
    # training options
    parser.add_argument("--skip_train", action="store_true", help="skip training")
    parser.add_argument(
        "--evaluation", action="store_true", help="do evaluation on test set"
    )
    parser.add_argument(
        "--no_undersample_graphs",
        action="store_true",
        help="undersample graphs as in LineVD",
    )
    parser.add_argument(
        "--debug_overfit", action="store_true", help="debug mode - overfit one batch"
    )
    parser.add_argument("--clean", action="store_true", help="clean old outputs")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="number of items to load in a batch"
    )
    parser.add_argument("--filter_cwe", nargs="+", help="CWE to filter examples")
    parser.add_argument(
        "--target_metric", type=str, default="valid/loss", help="metric to optimize for"
    )
    parser.add_argument(
        "--take_checkpoint",
        type=str,
        default="best",
        help="how to select checkpoint for evaluation",
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, help="checkpoint file to resume from"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=1,
        help="number of cross-validation folds to run.",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=250, help="max number of epochs to run."
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="patience value to use for early stopping. Omit to disable early stopping.",
    )
    parser.add_argument("--roc_every", type=int, help="print ROC curve every n epochs.")

    args, _ = parser.parse_known_args()

    BaseModule.add_model_specific_args(parser)
    args.model_class = model_class_dict[args.model]
    args.model_class.add_model_specific_args(parser)

    args = parser.parse_args()
    args.model_class = model_class_dict[args.model]

    # args.unique_id = '_'.join(map(str, (args.model, args.dataset, args.label_style, args.feat, args.node_limit, args.graph_limit, args.no_undersample_graphs, args.undersample_factor, args.filter, f"{args.learning_rate:f}".rstrip("0").rstrip("."), f"{args.weight_decay:f}".rstrip("0").rstrip("."), args.batch_size)))
    args.unique_id = "_".join(
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
            ),
        )
    )
    if args.filter_cwe:
        args.unique_id += "_filter_" + "_".join(args.filter_cwe)
    if args.model == "devign":
        args.unique_id += "_" + "_".join(
            map(str, (args.window_size, args.graph_embed_size, args.num_layers))
        )
    else:
        args.unique_id += "_" + "_".join(
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
        args.unique_id += "_debug_overfit"

    with open("ran_main.py_log.txt", "a") as f:
        f.write(f'{datetime.now()} {" ".join(sys.argv)}\n')

    if args.help:
        parser.print_help()
    else:
        main(vars(args))
