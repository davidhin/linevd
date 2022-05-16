import argparse
import copy
import functools
import json
import logging
import pickle
import shutil
import sys
from datetime import datetime

import dgl
import gensim
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# import optuna
# from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (DeviceStatsMonitor, EarlyStopping,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from sastvd.linevd import BigVulDatasetLineVDDataModule

from code_gnn.dataset import MyDGLDataset, split
from code_gnn.globals import (all_datasets, all_models, project_root_dir,
                              seed_all)
from code_gnn.models import model_class_dict
from code_gnn.models.base_module import BaseModule
from code_gnn.periodic_checkpoint import PeriodicModelCheckpoint

logger = logging.getLogger()


def visualize_example(g, b=None, a=None, _id=None):
    for cn, c in (("before", b), ("after", a)):
        c = c.splitlines()
        # visualize
        plt.figure(figsize=(15, 12))
        # color_map = [("white" if h.sum().item() == 0 else "red") for (n, attr), h in zip(graph.nodes(data=True), node_embeddings)]
        color_map = None
        int2label = {}
        dataflow_embeddings = g.ndata["_DATAFLOW"]
        node_label = g.ndata["_VULN"]
        line = g.ndata["_LINE"]
        for i in enumerate(g.nodes()):
            s = str(node_label[i].item())
            if c is not None:
                s += ": " + c[line[i]]
            if any(dataflow_embeddings[i]):
                for j, j_feat in enumerate(dataflow_embeddings[i]):
                    if j < len(dataflow_embeddings.shape[1]) / 2:
                        s += "\ngen "
                    else:
                        s += "\nkill "
                    if j_feat != 0:
                        # breakpoint()
                        s += j
            int2label[i] = s
        graph = dgl.to_networkx(g)
        graph_rl = nx.relabel_nodes(graph, int2label)
        # pos = nx.spring_layout(graph)
        pos = None
        nx.draw(graph_rl, pos=pos, node_color=color_map, with_labels=True)
        plt.savefig("images/" + _id + "_" + cn + ".png")


def train_optuna(trial, config):
    """
    Train a single model for hyperparameter tuning
    """

    assert config["n_folds"] == 1, 'Do not use cross-validation while running hyperparameter tuning!'

    config = copy.deepcopy(config)
    config["tune_trial"] = trial
    if config["model"] == 'flow_gnn' or config["model"] == 'flow_gnn_only':
        config["num_layers"] = trial.suggest_int("num_layers", 1, 8)
        config["num_mlp_layers"] = trial.suggest_int("num_mlp_layers", 1, 8)
        config["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    elif config["model"] == 'devign':
        # window size fixed at 100
        config["num_layers"] = trial.suggest_int("num_layers", 1, 8)
        config["num_mlp_layers"] = trial.suggest_categorical("graph_embed_size", [32, 64, 128, 256])

    return train_single_model(config)


def train_single_model(config):
    """
    Train a single model
    """

    # # Expects dataset to be shuffled BEFORE this point.
    # # How much data to reserve for test and validation sets
    # n_folds = config["n_folds"]
    # if n_folds == 1:
    #     test_val_portion = 0.1
    # else:
    #     test_val_portion = 1 / n_folds
    # dataset_splits = [1 - (2 * test_val_portion), test_val_portion, test_val_portion]
    # roll_n = len(dataset) // n_folds  # How many indices to roll the dataset for each fold

    # logger.info(f'{n_folds=} {dataset_splits=} {roll_n=}')
    # test_performances = []
    # for fold_idx in range(n_folds):
    #     if config["n_folds"] > 1:
    #         config["fold_idx"] = fold_idx
    #         logger.info(f'{fold_idx=}')
    #     if fold_idx > 0:
    #         dataset.roll(roll_n)  # TODO: What to do for odd values of roll_n?
    #     train_dataloader, valid_dataloader, test_dataloader = split(config["filter"], config["batch_size"], dataset, dataset_splits)
    #     logger.info(f'{dataset=} splits: {len(train_dataloader)=} {len(valid_dataloader)=} {len(test_dataloader)=} {sum(len(b.batch_num_nodes()) for b in train_dataloader)=} {sum(len(b.batch_num_nodes()) for b in valid_dataloader)=} {sum(len(b.batch_num_nodes()) for b in test_dataloader)=}')

        # trainer = get_trainer(config)

        # slim_config = copy.deepcopy(config)
        # if config["tune"]:
        #     del slim_config["tune_trial"]
        # model = config["model_class"](**slim_config)
        # trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
        # if trainer.interrupted:
        #     raise KeyboardInterrupt()

        # if config["debug_overfit"]:
        #     return

        # if config["take_checkpoint"] == 'last':
        #     test_performance = trainer.test(model=model, dataloaders=test_dataloader)
        # else:
        #     test_performance = trainer.test(model=model, ckpt_path=config["take_checkpoint"],
        #                                     dataloaders=test_dataloader)

        # logger.info(f'Got {len(test_performance)} test performances')
        # logger.info(json.dumps(test_performance, indent=2))

        # chosen_test_performance = test_performance[0]
        # test_performances.append(chosen_test_performance)

        # if config["tune"]:
        #     return chosen_test_performance["test_f1"]

    # agg_performance = {k: [] for k in test_performances[0].keys()}
    # for perf in test_performances:
    #     for metric in perf:
    #         agg_performance[metric].append(perf[metric])
    # for metric in agg_performance:
    #     logger.info(f'Average {metric} of {len(agg_performance[metric])} runs: {np.average(agg_performance[metric])}')
    
    print("config =", config)
    data = BigVulDatasetLineVDDataModule(
        batch_size=config["batch_size"],
        # sample=100,
        methodlevel=False,
        # nsampling=True,
        # nsampling_hops=2,
        # gtype="pdg+raw",
        gtype="cfg",
        splits="default",
        # feat="all",
        feat=config["feat"],
        #load_code=config["dataset_only"],
    )
    #if config["dataset_only"]:
    #    for i in range(10):
    #        visualize_example(data.train[i], data.train.df.loc[data.train.idx2id[i]]["before"], data.train.df.loc[data.train.idx2id[i]]["after"])
    #    return
    trainer = get_trainer(config)
    print("graph", data.train[0])
    print("graph data", data.train[0].ndata)

    # config["input_dim"] = data.max_df_dim
    config["input_dim"] = data.train[0].ndata["_ABS_DATAFLOW"].shape[1]
    
    model = config["model_class"](**config)
    if config["evaluation"]:
        test_performance = trainer.test(model=model, datamodule=data, ckpt_path=config["resume_from_checkpoint"])
        logger.info(test_performance)
    else:
        trainer.fit(model, datamodule=data)


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
                / 'overfit_batch'
        )
    elif config["evaluation"]:
        base_dir = (
                project_root_dir
                / "logs"
                / (config["unique_id"] + config["log_suffix"])
                / 'evaluation'
        )
        assert config["resume_from_checkpoint"] is not None
    else:
        base_dir = (
                project_root_dir
                / "logs"
                / (config["unique_id"] + config["log_suffix"])
                / 'default'
        )

    if base_dir.exists():
        if config["clean"]:
            if config["resume_from_checkpoint"] is not None:
                logger.warning(f'Told to clean {base_dir}, but also to load. Skipping --clean.')
            else:
                logger.info(f'Cleaning old results from {base_dir}...')
                shutil.rmtree(base_dir)
        elif config["resume_from_checkpoint"] is None:
            raise NotImplementedError(f'Please clear old results from {base_dir}')

    ckpt_dir = base_dir / 'checkpoints'
    logger.info(f'Checkpointing to {ckpt_dir}')
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename='performance-{epoch:02d}-{step:02d}-{' + config["target_metric"] + ':02f}',
        monitor=config["target_metric"],
        mode='max',
        save_last=True,
        # verbose=True,
    )
    callbacks.append(checkpoint_callback)
    checkpoint_callback = PeriodicModelCheckpoint(
        dirpath=str(ckpt_dir),
        every=25,
    )
    callbacks.append(checkpoint_callback)

    tb_logger = TensorBoardLogger(str(base_dir), version='', name='')

    if config["patience"] is not None:
        early_stopping_callback = EarlyStopping(
            monitor=config["target_metric"],
            mode="max",
            patience=config["patience"],
        )
        callbacks.append(early_stopping_callback)

    if config["profile"]:
        callbacks.append(DeviceStatsMonitor())
    if "tune_trial" in config:
        callbacks.append(PyTorchLightningPruningCallback(config["tune_trial"], monitor=config["target_metric"]))

    trainer = pl.Trainer(
        gpus=1 if config["cuda"] else 0,
        num_sanity_val_steps=0 if config["tune"] else 2,
        overfit_batches=1 if config["debug_overfit"] else 0,
        limit_train_batches=config["debug_train_batches"] if config["debug_train_batches"] else 1.0,
        # https://forums.pytorchlightning.ai/t/validation-sanity-check/174/6
        detect_anomaly=True,
        callbacks=callbacks,
        logger=tb_logger,
        max_epochs=config["max_epochs"],
        # default_root_dir=base_dir,  # Use checkpoint callback instead
        # deterministic=True,  # RuntimeError: scatter_add_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
        enable_checkpointing=True,
        profiler="simple" if config["profile"] else None,
        resume_from_checkpoint=config["resume_from_checkpoint"],
    )
    return trainer


def main(config):
    logger.info(f'config={config}')
    seed_all(config["seed"])

    config["cuda"] = torch.cuda.is_available()
    logger.info(f'gpus={torch.cuda.is_available()}, {torch.cuda.device_count()}')

    # dataset = MyDGLDataset(config, verbose=True)
    # logger.debug(f'{dataset[0]=}')
    # logger.debug(f'{dataset[0].ndata=}')

    # config["input_dim"] = dataset.input_dim
    # if config["dataset_only"]:
    #     # logger.debug('Quitting early after dataset load...')
    #     # logger.debug(f'{torch.sum(torch.eq(dataset.labels, 0))=}')
    #     # logger.debug(f'{torch.sum(torch.eq(dataset.labels, 1))=}')
    #     # logger.debug(f'{len(dataset.labels)=}')
    #     n = 0
    #     n_pos = 0
    #     n_target_1 = 0
    #     n_target_1_nopos = 0
    #     graph_sizes = []
    #     for i in range(len(dataset)):
    #         graph = dataset[i]
    #         this_n = graph.number_of_nodes()
    #         n += this_n
    #         this_n_pos = graph.ndata["node_label"].sum().item()
    #         n_pos += this_n_pos
    #         if config["dataset"] != "SARD":
    #             if dataset.df.loc[dataset.idx_to_row_name[i]]["target"] == 1:
    #                 n_target_1 += 1
    #                 if this_n_pos == 0:
    #                     n_target_1_nopos += 1
    #         graph_sizes.append(this_n)
    #         # logger.debug(f'{i=} {len(dataset)=}')
    #     logger.debug(f'{n=} {n_pos=} {n_target_1=} {n_target_1_nopos=} average_graph_size={np.average(graph_sizes)} positive_percent={(n_pos / n) * 100:.2f}%')
    #     return

    # config["input_dim"] = dataset.input_dim

    seed_everything(config["seed"], workers=True)
    if config["tune"]:
        pass
        # objective_fn = functools.partial(train_optuna, config=config)
        # study_savefile = project_root_dir / 'saved_studies' / f'{args.unique_id}.pkl'
        # study_savefile.parent.mkdir(parents=True, exist_ok=True)
        # if args.resume:
        #     with open(study_savefile, 'rb') as f:
        #         study = pickle.load(f)
        #     logger.info('Intermediate study results:')
        #     log_results(study)
        # else:
        #     study = optuna.create_study(direction="maximize", study_name=args.unique_id,
        #                                 sampler=optuna.samplers.TPESampler(),
        #                                 pruner=optuna.pruners.NopPruner(),
        #                                 load_if_exists=args.resume)
        # try:
        #     study.optimize(objective_fn, n_trials=config["n_trials"], timeout=config["tune_timeout"])
        # except KeyboardInterrupt:
        #     logger.warning('Detected keyboard interrupt.')

        # with open(study_savefile, 'wb') as f:
        #     pickle.dump(study, f)
        # log_results(study)
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


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model
    parser.add_argument("--model", choices=all_models,
                        help='short ID for the model type to train',
                        required=True)
    # dataset
    parser.add_argument("--dataset", choices=all_datasets,
                        required=True,
                        help='short ID for the dataset to train on')
    parser.add_argument("--feat", required=True, help='node features to use')
    parser.add_argument("--node_limit", type=int, help='upper limit to the number of nodes in a graph')
    parser.add_argument("--graph_limit", type=int, help='upper limit to the number of graphs to parse')
    parser.add_argument("--filter", type=str, help='filter data to a certain persuasion', default='')
    parser.add_argument("--label_style", type=str, help='use node or graph labels', default='graph')
    parser.add_argument("--debug_train_batches", type=int, help='debug mode - train with n batches')
    parser.add_argument("--undersample_factor", type=float, help='factor to undersample majority class')
    # logging and reproducibility
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--log_suffix", type=str, default='', help='suffix to append after log directory')
    parser.add_argument("-h", '--help', action='store_true', help='print this help message')
    parser.add_argument("--version", type=str, default=None, help='version ID to use for logging')
    # different run modes
    parser.add_argument("--dataset_only", action='store_true', help='only load the dataset, then exit')
    parser.add_argument("--profile", action='store_true',
                        help='run training under the profiler and report results at the end')
    # tuning options
    parser.add_argument("--tune", action='store_true', help='tune hyperparameters')
    parser.add_argument("--resume", action='store_true', help='resume previous tune progress')
    parser.add_argument("--n_trials", type=int, default=50, help='how many trials to tune')
    parser.add_argument("--tune_timeout", type=int, default=60 * 60 * 24, help='time limit for tuning')
    # training options
    parser.add_argument("--evaluation", action='store_true', help='only do evaluation on test set')
    parser.add_argument("--debug_overfit", action='store_true', help='debug mode - overfit one batch')
    parser.add_argument("--clean", action='store_true', help='clean old outputs')
    parser.add_argument("--batch_size", type=int, default=64, help='number of items to load in a batch')
    parser.add_argument("--target_metric", type=str, default='valid/loss', help='metric to optimize for')
    parser.add_argument("--take_checkpoint", type=str, default='best', help='how to select checkpoint for evaluation')
    parser.add_argument("--resume_from_checkpoint", type=str, help='checkpoint file to resume from')
    parser.add_argument("--n_folds", type=int, default=1, help='number of cross-validation folds to run.')
    parser.add_argument("--max_epochs", type=int, default=250, help='max number of epochs to run.')
    parser.add_argument("--patience", type=int, help='patience value to use for early stopping. Omit to disable early stopping.')
    parser.add_argument("--roc_every", type=int, help='print ROC curve every n epochs.')

    args, _ = parser.parse_known_args()

    BaseModule.add_model_specific_args(parser)
    args.model_class = model_class_dict[args.model]
    args.model_class.add_model_specific_args(parser)

    args = parser.parse_args()
    args.model_class = model_class_dict[args.model]

    args.unique_id = '_'.join(map(str, (args.model, args.dataset, args.feat, args.node_limit, args.graph_limit, args.undersample_factor, args.filter, f"{args.learning_rate:f}".rstrip("0").rstrip("."), f"{args.weight_decay:f}".rstrip("0").rstrip("."), args.batch_size)))
    if args.model == 'devign':
        args.unique_id += '_' + '_'.join(map(str, (args.window_size, args.graph_embed_size, args.num_layers)))
    else:
        args.unique_id += '_' + '_'.join(
            map(str, (args.num_layers, args.num_mlp_layers, args.hidden_dim, args.learn_eps,
                      args.final_dropout, args.graph_pooling_type, args.neighbor_pooling_type)))

    # if args.label_style is None:
    #     args.label_style = 'node' if args.dataset in ('SARD', 'MSR') else 'graph'

    with open('ran_main.py_log.txt', 'a') as f:
        f.write(f'{datetime.now()} {" ".join(sys.argv)}\n')

    if args.help:
        parser.print_help()
    else:
        main(vars(args))
