import argparse
import copy
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import jsonlines
import networkx as nx
import numpy as np
import tqdm

from code_gnn.analysis.cpg import get_cpg
from code_gnn.globals import ml_data_dir, get_cache_filename
from code_gnn.models import model_class_dict
from code_gnn.models.embedding_ids import node_type_map
from code_gnn.models.embeddings import DataflowEmbeddingGetter, DevignEmbeddingGetter, NoFeaturesEmbeddingGetter

logger = logging.getLogger()


def slugify(data, fields):
    return '_'.join(
        map(str, [data[field] for field in fields]))


def parse_all_cpgs(parsed_dir, raw_datas, label_style, node_limit=None, graph_limit=None):
    """Parse all CPGs with base dir parsed_dir"""
    failures = defaultdict(int)
    labels = []
    cpgs = []
    desc = 'Loading graphs...'
    parsed = 0
    i = 0
    with tqdm.tqdm(raw_datas, desc=desc) as pbar:
        for i, raw_data in enumerate(pbar, start=-1):

            file = raw_data["slug_filename"]
            pbar.set_description(f'{desc} ({file})')
            try:
                cpg = get_cpg(parsed_dir, file)
                cpg.graph["filepath"] = str(file)
            except AssertionError as e:
                failures[f"cpgparse_assertionerror"] += 1
                pbar.write(f'AssertionError parsing {file}: {e}')
                continue
            except Exception as e:
                failures[f"cpgparse_genericerror"] += 1
                pbar.write(f'Error parsing {file}: {e}')
                logger.exception(e)
                continue

            # Remove graphs which are obviously incorrect
            if not any(cpg.nodes[node]["type"] == node_type_map['Function'] for node in cpg.nodes):
                failures[f"cpgparse_nofunction"] += 1
                continue

            # Remove graphs with more than 500 nodes for computational efficiency, as in the paper
            if node_limit is not None and len(cpg.nodes) > node_limit:
                failures[f"cpgparse_toobig"] += 1
                continue

            if label_style == 'graph':
                cpgs.append(cpg)
                labels.append(raw_data["target"])
                parsed += 1
            elif label_style == 'node':
                # Assign nodes to nodes in the functions
                my_labels = set()
                for d in cpg.nodes():
                    for label in raw_data["labels"]:
                        if label["tag"] in ("flaw", "mixed"):
                            if cpg.nodes[d]["line"] == label["line"]:
                                my_labels.add(d)
                if len(my_labels) == 0:
                    failures[f"cpgparse_nolabels"] += 1
                else:
                    # Extract every function with a labeled node
                    for d in cpg.nodes():
                        if cpg.nodes[d]["type"] == node_type_map["FunctionDef"]:
                            func_cpg = cpg.subgraph(nx.descendants(cpg, d)).copy()
                            node_labels = {n: 1 if n in my_labels else 0 for n in func_cpg.nodes()}
                            if sum(node_labels.values()) > 0:
                                nx.set_node_attributes(func_cpg, node_labels, name='label')
                                cpgs.append(func_cpg)
                                # labels.append(list(sorted(matching_labels)))
                    # Make the graph interprocedural
                    # funcs = {cpg.nodes[d]["code"]: d for d in nx.descendants(cpg) if cpg.nodes[d]["type"] == node_type_map["FunctionDef"]}
                    # for d in nx.descendants(cpg):
                    #     if cpg.nodes[d]["type"] == node_type_map["Callee"]:
                    #         cpg.add_edge(d, funcs[cpg.nodes[d]["code"]])
                    labels.append(-1)
                    parsed += 1
            else:
                raise NotImplementedError(label_style)

            if graph_limit is not None and parsed >= graph_limit:
                num_files_skipped = len(raw_datas) - parsed
                logger.info(f'Stopping at {graph_limit} and skipping {num_files_skipped} files')
                failures[f"cpgparse_skipped"] += num_files_skipped
                break
    return cpgs, labels, failures, i


def get_raw_data(name, root_dir):
    """
    Get un-parsed raw data: code and labels.

    Should have fields on each example after output:
    - func: the source code of the function
    - file_name: the filename which should be a flat, valid filename which is unique throughout the dataset
    - target: the target (1 vulnerable, 0 non-vulnerable)
    """
    if name == 'devign':
        with open(root_dir / 'function.json') as f:
            raw_datas = json.load(f)
        # Generate names for files and sort them by name
        raw_datas_len = len(raw_datas)
        num_digits = len(str(raw_datas_len))
        for i, d in enumerate(raw_datas):
            # NOTE: Adding this attribute for convenience, but it will not be persisted
            # suffix = '_'.join(d["func"][:d["func"].find('(')].split()[:5]).replace('/', '__')
            d["slug_filename"] = f'{str(i).rjust(num_digits, "0")}_' \
                                 + slugify(d, ["project", "commit_id", "target"])
        raw_datas = list(sorted(raw_datas, key=lambda x: x["slug_filename"]))
    elif name == 'combined_new':
        raw_datas = []
        with open(root_dir.parent / 'D2A' / 'D2A_slim.jsonl') as f:
            for line in f.readlines():
                d = json.loads(line)
                raw_datas.append(d)
        with open(root_dir.parent / 'MSR' / 'MSR_slim.jsonl') as f:
            for line in f.readlines():
                d = json.loads(line)
                d["slug_filename"] = d["id"]
                raw_datas.append(d)
        with open(root_dir.parent / 'CVEFixes' / 'CVEFixes_slim.jsonl') as f:
            for line in f.readlines():
                d = json.loads(line)
                d["slug_filename"] = d["id"]
                raw_datas.append(d)
    elif name == 'D2A':
        with open(root_dir / 'D2A_slim.jsonl') as f:
            raw_datas = [json.loads(line) for line in f.readlines()]
    elif name == 'MSR':
        with open(root_dir / 'MSR_slim.jsonl') as f:
            raw_datas = [json.loads(line) for line in f.readlines()]
        for d in raw_datas:
            d["slug_filename"] = d["id"]
    elif name == 'CVEFixes':
        with open(root_dir / 'CVEFixes_slim.jsonl') as f:
            raw_datas = [json.loads(line) for line in f.readlines()]
        for d in raw_datas:
            d["slug_filename"] = d["id"]
    elif name == 'combined':
        with open(root_dir / 'MSR_D2A_CVEFixes_processed.json') as f:
            raw_datas = json.load(f)
        raw_datas_len = len(raw_datas)
        num_digits = len(str(raw_datas_len))
        for i, d in enumerate(raw_datas):
            d["slug_filename"] = f'{str(i).rjust(num_digits, "0")}_' \
                                 + d["slug_filename"] + '.c'
    elif name == 'SARD':
        with jsonlines.open(root_dir / 'data.jsonl') as reader:
            raw_datas = list(reader)
        print(f'{len(raw_datas)=}')
        max_id = max(d["id"] for d in raw_datas)
        len_max_id = len(str(max_id))
        for d in raw_datas:
            d["slug_filename"] = f'{str(d["id"]).rjust(len_max_id, "0")}_' + d["filename"].replace('/', '_')
    else:
        raise NotImplementedError(name)
    logger.info(f'{len(raw_datas)=}')
    return raw_datas


def preprocess(mode, data_root_dir, dataset_name, model_name, node_limit, graph_limit, label_style, **kwargs):
    random.seed(kwargs.get('seed'))

    root_dir = Path(data_root_dir) / dataset_name
    assert root_dir.exists(), root_dir
    logger.debug(f'{data_root_dir=} {dataset_name=} {model_name=}')
    logger.debug(f'{node_limit=} {graph_limit=} {kwargs=}')

    raw_datas = get_raw_data(dataset_name, root_dir)
    random.shuffle(raw_datas)

    if mode == 'extract_files':
        # Write source files to flat directory source_dir
        source_dir = root_dir / 'files'
        assert not source_dir.exists(), 'source directory already exists, please clear'
        source_dir.mkdir()
        for d in raw_datas:
            file_path = os.path.join(source_dir, d["slug_filename"])
            code = d["func"]
            if code is None:
                logger.error(f'no code for {d["slug_filename"]}')
                continue
            try:
                with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
                    # TODO: Possibly collapse whitespaces
                    f.write(code)
            except OSError as e:
                logger.exception(f'error writing file {file_path}', exc_info=e)
                if os.path.exists(file_path):
                    os.remove(file_path)
    if mode == 'read_parsed':
        # Assume all files have been parsed with Joern
        parsed_dir = root_dir / 'parsed'
        assert parsed_dir.exists(), 'Parse dat dataset!'

        cache_filepath = root_dir / get_cache_filename(dataset_name, model_name, node_limit, graph_limit)
        logger.debug(f'{root_dir=} {cache_filepath=}')

        if cache_filepath.exists():
            with open(cache_filepath, 'rb') as f:
                dgl_graphs, dgl_labels, slug_filenames = pickle.load(f)
        else:
            failures = defaultdict(int)

            # Parse each source graph
            cpg_file = root_dir / f'cpg_stage_{node_limit}_{graph_limit}.pkl'
            if cpg_file.exists():
                logger.info(f'Loading CPG data from {cpg_file}')
                with open(cpg_file, 'rb') as f:
                    cpgs, labels, parse_failures = pickle.load(f)
            else:
                logger.info(f'Loading CPGs from files in {parsed_dir}...')
                cpgs, labels, parse_failures, stop_i = parse_all_cpgs(
                    parsed_dir, raw_datas, label_style, node_limit=node_limit, graph_limit=graph_limit)
                assert len(cpgs) > 0, 'No CPGs parsed successfully'
                with open(cpg_file, 'wb') as f:
                    pickle.dump([cpgs, labels, parse_failures], f)
            failures.update(parse_failures)
            logger.info(f'{failures=}')

            # Get embedding getter
            if model_name == 'flow_gnn':
                embedding_getter = DataflowEmbeddingGetter(cpgs, node_type_separate=True)
            elif model_name == 'devign':
                embedding_getter = DevignEmbeddingGetter(dataset_name, kwargs.get("window_size"), raw_datas)
            elif model_name == 'random':
                embedding_getter = NoFeaturesEmbeddingGetter()
            else:
                raise NotImplementedError(model_name)

            # Get DGL graphs
            dgl_graphs = []
            slug_filenames = []
            for cpg, label in tqdm.tqdm(list(zip(cpgs, labels)), desc='Loading DGL graphs...'):
                try:
                    g = embedding_getter.get_graph(cpg)
                    dgl_graphs.append(g)
                    slug_filenames.append(cpg.graph["filepath"])
                    failures["dgl_get_graph_success"] += 1
                except AssertionError:
                    failures["dgl_get_graph_assertionerror"] += 1
                except Exception as e:
                    failures["dgl_get_graph_othererror"] += 1
                    logger.exception(e)
            logger.info(f'{failures=}')
            assert len(dgl_graphs) > 0, 'no DGL graphs loaded successfully'
            logger.info(f'Loaded {len(dgl_graphs)} DGL graphs.')

            with open(cache_filepath, 'wb') as f:
                pickle.dump([dgl_graphs, slug_filenames], f)

        logger.info(f'Total dataset length: {len(dgl_graphs)}')
        if label_style == 'graph':
            logger.info(f'Label distribution: {sum(dgl_labels) / len(dgl_labels)}')
        elif label_style == 'node':
            num_total = []
            num_1 = []
            num_all_zero = 0
            for g in dgl_graphs:
                vul_sum = g.ndata['label'].sum().item()
                num_1.append(vul_sum)
                num_total.append(g.number_of_nodes())
                if vul_sum == 0:
                    num_all_zero += 1
            logger.info(f'Average number nodes: {np.average(num_total)}')
            logger.info(f'Average number vulnerable nodes: {np.average(num_1)}')
            logger.info(f'Average number non-vulnerable nodes: {np.average([a - b for a, b in zip(num_total, num_1)])}')
            logger.info(f'Number graphs: {len(dgl_graphs)}')
            logger.info(f'Number all-non-vulnerable graphs: {num_all_zero}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['extract_files', 'read_parsed'], type=str)
    parser.add_argument("--data_root_dir", default=ml_data_dir)
    parser.add_argument("--dataset_name", choices=['D2A', 'CVEFixes', 'MSR', 'combined_new', 'SARD'], nargs='+')
    parser.add_argument("--model_name", choices=['devign', 'flow_gnn', 'random'])
    parser.add_argument("--label_style", choices=['node', 'graph'], default='graph')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--node_limit", type=int)
    parser.add_argument("--graph_limit", type=int)
    args, _ = parser.parse_known_args()

    model_class = model_class_dict[args.model_name]
    model_class.add_model_specific_args(parser)

    args = parser.parse_args()
    args_dict = vars(args)

    for ds in args.dataset_name:
        my_args_dict = copy.deepcopy(args_dict)
        my_args_dict["dataset_name"] = ds
        preprocess(**my_args_dict)
