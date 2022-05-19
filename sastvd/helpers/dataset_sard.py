import os
import re

import pandas as pd
import sastvd as svd
import sastvd.helpers.doc2vec as svdd2v
import sastvd.helpers.git as svdg
import sastvd.helpers.glove as svdglove
import sastvd.helpers.tokenise as svdt
from sklearn.model_selection import train_test_split


from code_gnn.globals import ml_data_dir, project_root_dir
import difflib
from pathlib import Path
# from lxml import etree as ET
from xml.etree import ElementTree as ET
from code_gnn.globals import ml_data_dir
import copy
import json
import numpy as np
import functools
import traceback


def train_val_test_split_df(df, idcol, labelcol):
    """Add train/val/test column into dataframe."""
    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=1
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_rat / (test_rat + val_rat), random_state=1
    )
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "val"
        if path in X_test:
            return "test"

    df["label"] = df[idcol].apply(path_to_label)
    return df


def remove_comments(text):
    """Delete comments from code."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)


def read_code(filepath):
    """Read code for file."""

    with open(svd.external_dir() / 'sard_archive/testcases' / filepath, encoding='utf-8', errors='replace') as f:
        return f.read()


def read_sard_df(merged=True):

    # load XML
    sard_dir = svd.external_dir() / 'sard_archive'
    manifest_file = sard_dir / 'full_manifest.xml'
    tree = ET.parse(str(manifest_file))
    root = tree.getroot()
    print(root.tag)
    print(len(root))

    # convert to DataFrames
    testcases = []
    files = []
    tags = []
    # .find(".//{%s}Amount"
    # for tc_xml in root.xpath("testcase"):
    # breakpoint()
    missed_tags = set()
    for tc_xml in root:
        if tc_xml.tag != "testcase":
            # print(tc_xml.tag)
            missed_tags.add(tc_xml.tag)
            continue
        tc_attrib = tc_xml.attrib
        # if "id" not in tc_attrib:
        #     print(tc_xml.tag, tc_xml.attrib)
        #     continue
        tc = {
            # "xml_tag": tc_xml.tag,
            **tc_attrib,
        }
        # for f_xml in testcase.xpath("file"):
        # files_result = tc_xml.find("file")
        # if files_result is None:
        #     continue
        for f_xml in tc_xml:
            if f_xml.tag == "description":
                tc.update({f"text_{f_xml.tag}": f_xml.text})
                continue
            elif f_xml.tag == "association":
                tc.update({f"{k}_{f_xml.tag}": v for k, v in f_xml.attrib.items()})
                continue
            elif f_xml.tag != "file":
                # print(f_xml.tag)
                missed_tags.add(f_xml.tag)
                continue
            f_id = len(files)
            f = {
                "testcase": tc["id"],
                "id": f_id,
                # "xml_tag": f_xml.tag,
                **f_xml.attrib
            }
            # for t_xml in f_xml.xpath("flaw") + f_xml.xpath("fix") + f_xml.xpath("mixed"):
            # tags_result = f_xml.find("flaw") + f_xml.find("fix") + f_xml.find("mixed")
            # if tags_result is None:
            #     continue
            for t_xml in f_xml:
                if t_xml.tag not in ("flaw", "fix", "mixed"):
                    # print(t_xml.tag)
                    missed_tags.add(t_xml.tag)
                    continue
                tag_id = len(tags)
                tags.append({
                    "file": f["id"],
                    "id": tag_id,
                    # "xml_tag": t_xml.tag,
                    **t_xml.attrib
                })
            files.append(f)
        testcases.append(tc)
    print("missed_tags", missed_tags)
    # breakpoint()
    # tcdf = pd.DataFrame.from_records(data=testcases).set_index("id")#, index="id")
    # fdf = pd.DataFrame.from_records(data=files).set_index("id")#, index="id")
    # tdf = pd.DataFrame.from_records(data=tags).set_index("id")#, index="id")
    tcdf = pd.DataFrame.from_records(data=testcases, index="id")
    fdf = pd.DataFrame.from_records(data=files, index="id")
    tdf = pd.DataFrame.from_records(data=tags, index="id")

    # merged_df = tcdf.merge(fdf, right_on="testcase", suffixes=("_testcase", "_file")).merge(tags, right_on="file", suffixes=("", "_tag"))
    return tcdf, fdf, tdf


def test_read_sard_df():
    print("separate")
    tcdf, fdf, tdf = read_sard_df(merged=False)
    print("testcases")
    print("columns", tcdf.columns)
    print(tcdf)
    print("files")
    print("columns", fdf.columns)
    print(fdf)
    print("tags")
    print("columns", tdf.columns)
    print(tdf)
    # print("merged")
    # df = read_sard_df(merged=True)
    # print(df)


import sastvd.helpers.joern as svdj


def get_functions(_id, joern_dir):
    """
    Split files into functions
    """
    try:
        n, e = svdj.get_node_edges(joern_dir / _id)
        function_nodes = n[n["_label"] == "METHOD"]
        return function_nodes.T.to_dict().values()
    except Exception as E:
        print("get_functions exception", traceback.format_exc())
        return np.nan


def sard(minimal=True, sample=False, return_raw=False, splits="default"):
    """Read SARD Data.

    Args:
        sample (bool): Only used for testing!
        splits (str): default, crossproject-(linux|Chrome|Android|qemu)

    EDGE CASE FIXING:
    id = 177860 should not have comments in the before/after
    """
    savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")
    df_savefile = savedir / f"minimal_sard_beforefunc_{sample}.pq"
    # print("loading", df_savefile)
    if minimal:
        try:
            df = pd.read_parquet(
                df_savefile, engine="fastparquet"
            ).dropna()
            # print("loaded", len(df))

            # md = pd.read_csv(svd.cache_dir() / "sard/sard_metadata.csv")
            # md.groupby("project").count().sort_values("id")

            # default_splits = svd.external_dir() / "sard_rand_splits.csv"
            # if os.path.exists(default_splits):
            #     splits = pd.read_csv(default_splits)
            #     splits = splits.set_index("id").to_dict()["label"]
            #     df["label"] = df.id.map(splits)

            # if "crossproject" in splits:
            #     project = splits.split("_")[-1]
            #     md = pd.read_csv(svd.cache_dir() / "sard/sard_metadata.csv")
            #     nonproject = md[md.project != project].id.tolist()
            #     trid, vaid = train_test_split(nonproject, test_size=0.1, random_state=1)
            #     teid = md[md.project == project].id.tolist()
            #     teid = {k: "test" for k in teid}
            #     trid = {k: "train" for k in trid}
            #     vaid = {k: "val" for k in vaid}
            #     cross_project_splits = {**trid, **vaid, **teid}
            #     df["label"] = df.id.map(cross_project_splits)

            return df
        except Exception as E:
            print("sard exception", E)
            pass

    # load from file
    filename = "sard_SAMPLE.csv" if sample else "sard_cleaned.csv"
    tcdf, fdf, tdf = read_sard_df()
    # if return_raw:
    #     yield "Initial load testcases", tcdf
    #     yield "Initial load files", fdf
    #     yield "Initial load tags", tdf

    df = tcdf.merge(fdf, left_index=True, right_on="testcase", suffixes=("_testcase", "_file"), validate="one_to_many")
    # if return_raw:
    #     yield "Merge testcases with files", df
    df["dataset"] = "sard"
    assert len(df) == len(fdf)
    
    # TODO: test SARD specific filtering
    df = df[(df["language_file"] == "C") & (~df["path"].str.startswith("shared/"))]
    
    # if return_raw:
    #     yield "Filter language and shared files", df
    
    # Add labels
    # breakpoint()
    tag_json = tdf.groupby("file").apply(lambda x: x.to_json(orient='records'))
    # df = df.merge(tdf, left_on="id_file", right_on="file", suffixes=("", "_tag"))
    # assert len(df) == len_before_add
    df["tags"] = tag_json
    
    # if return_raw:
    #     yield "Add tag json (should not change # of columns)", df

    # below preprocessing steps adapted from sastvd.helpers.datasets
    # Remove comments
    df["code"] = svd.dfmp(df, read_code, columns="path", cs=500)
    df["code"] = svd.dfmp(df, remove_comments, columns="code", cs=500)
    
    # if return_raw:
    #     yield "Load and clean code", df

    # # Save codediffs
    # cols = ["code", "id", "dataset"]
    # svd.dfmp(df, svdg._c2dhelper, columns=cols, ordr=False, cs=300)

    # # Assign info and save
    # df["info"] = svd.dfmp(df, svdg.allfunc, cs=500)
    # df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)

    # # POST PROCESSING
    # dfv = df[df.vul == 1]
    # # No added or removed but vulnerable
    # dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
    # # Remove functions with abnormal ending (no } or ;)
    # dfv = dfv[
    #     ~dfv.apply(
    #         lambda x: (
    #             x.code.strip()[-1] != "}" and
    #             x.code.strip()[-1] != ";"
    #             ),
    #         axis=1,
    #     )
    # ]
    # dfv = dfv[
    #     ~dfv.apply(
    #         lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
    #         axis=1,
    #     )
    # ]
    # # Remove functions with abnormal ending (ending with ");")
    # dfv = dfv[~dfv.code.apply(lambda x: x[-2:] == ");")]

    # # Remove samples with mod_prop > 0.5
    # dfv["mod_prop"] = dfv.apply(
    #     lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    # )
    # dfv = dfv.sort_values("mod_prop", ascending=0)
    # dfv = dfv[dfv.mod_prop < 0.7]
    # # Remove functions that are too short
    # dfv = dfv[dfv.apply(lambda x: len(x.code.splitlines()) > 5, axis=1)]
    # # Filter by post-processing filtering
    # keep_vuln = set(dfv.id.tolist())
    # df = df[(df.vul == 0) | (df.id.isin(keep_vuln))].copy()

    # TODO: Save splits
    # # Make splits
    # df = train_val_test_split_df(df, "id", "vul")

    # df = df.join(tdf, on="file", suffixes=("", "_tag"))

    # After parsing with joern, we will split files into functions and filter
    
    # if return_raw:
    #     yield "Save", df

    slim_cols = [
        "dataset",
        "code",
    ]
    df[slim_cols].to_parquet(
        df_savefile,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    metadata_cols = df.columns.tolist()  # Keep all cols
    metadata_cols.remove("code")
    df[metadata_cols].to_csv(svd.cache_dir() / "sard_metadata.csv", index=0)
    return df


def test_sard():
    df = sard()
    print(df)



def sard_func(minimal=True, sample=False, return_raw=False, splits="default"):
    """Read SARD Data.

    Args:
        sample (bool): Only used for testing!
        splits (str): default, crossproject-(linux|Chrome|Android|qemu)

    EDGE CASE FIXING:
    id = 177860 should not have comments in the before/after
    """
    savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")
    df_before_savefile = savedir / f"minimal_sard_beforefunc_{sample}.pq"
    df_metadata_file = svd.cache_dir() / "sard_metadata.csv"
    df_savefile = savedir / f"minimal_sard_{sample}.pq"
    if minimal:
        try:
            df = pd.read_parquet(
                df_savefile, engine="fastparquet"
            ).dropna()
            return df
        except Exception as E:
            print("sard exception", E)
            pass
    
    # read from file
    df = pd.read_parquet(
        df_before_savefile, engine="fastparquet"
    ).dropna()

    df_metadata = pd.read_csv(df_metadata_file)
    df_metadata = df_metadata.join(pd.json_normalize(df_metadata["tags"].fillna("[]").apply(json.loads).explode()))
    if return_raw:
        yield "explode JSON (df_metadata)", df_metadata
    df_metadata["CWE"] = df_metadata["name"].apply(lambda n: re.findall(r"(CWE-[0-9]+)", n) if isinstance(n, str) else np.nan)
    df["CWE"] = df_metadata.reset_index().groupby("index")["CWE"].agg(sum).drop_duplicates().str.join(',').fillna("").sort_values()
    if return_raw:
        yield "extract CWE", df

    # Split files into functions and filter
    joern_dir = svd.processed_dir() / f"sard/code"
    df["method"] = svd.dfmp(df.reset_index(), functools.partial(get_functions, joern_dir=joern_dir), columns="index")
    if return_raw:
        yield "get functions", df
    df = pd.json_normalize(df.explode("method"))
    def trim_code_to_method(row):
        lines = row["code"].splitlines(keepends=True)
        offset = 0
        for i, l in enumerate(lines, start=1):
            start_index = -1
            end_index = -1
            if i == m["lineNumber"]:
                start_index = offfset + row["method"]["columnNumber"]-1
            if i == m["lineNumberEnd"]:
                end_index = offfset + row["method"]["columnNumberEnd"]-1
            offset += len(l)
        return code[start_index:end_index+1]
    df["code_all"] = df["code"]
    df["code"] = svd.dfmp(df, trim_code_to_method, columns=["code", "method"], cs=500)
    if return_raw:
        yield "extract function code", df

    # Filter by:
    # function length too long or short
    df = df[df.apply(lambda x: len(x.code.splitlines()) > 5, axis=1)]
    # testcase label
    df = df[df["label"].str.contains("fix")]
    # CWE
    cwe_select = "CWE-476,CWE-690"
    df = df[df["CWE"].str.contains(cwe_select)]

    df.to_parquet(
        df_savefile,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    return df

def test_sard_func():
    for stage, df in sard_func(return_raw=True):
        print(stage)
        print("df.columns", df.columns)
        print(df)

"""
vvv DATA LOADER vvv
"""


"""Main code for training. Probably needs refactoring."""
import os
from glob import glob

import dgl
import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.codebert as cb
import sastvd.helpers.dclass as svddc
import sastvd.helpers.doc2vec as svdd2v
import sastvd.helpers.glove as svdg
import sastvd.helpers.joern as svdj
import sastvd.helpers.losses as svdloss
import sastvd.helpers.ml as ml
import sastvd.helpers.rank_eval as svdr
import sastvd.helpers.sast as sast
import sastvd.ivdetect.evaluate as ivde
import sastvd.linevd.gnnexplainer as lvdgne
import sastvd.analysis.dataflow as df
import torch as th
import torch.nn.functional as F
import torchmetrics
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, GraphConv
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from tqdm import tqdm
from multiprocessing import Pool
import networkx as nx
import traceback
import sastvd.helpers.datasets as svdds

# TODO: map all label strings to ints
label_to_id = {}

enable_dataflow = True

def get_dataflow_dim(i):
    cpg = df.get_cpg(svddc.svdds.itempath(i))
    problem = df.ReachingDefinitions(cpg)
    domain_len = len(problem.domain)*2
    return domain_len


def ne_groupnodes(n, e):
    """Group nodes with same line number."""
    nl = n[n.lineNumber != ""].copy()
    nl.lineNumber = nl.lineNumber.astype(int)
    nl = nl.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    nl = nl.groupby("lineNumber").head(1)
    el = e.copy()
    el.innode = el.line_in
    el.outnode = el.line_out
    nl["nodeId"] = nl.id
    nl.id = nl.lineNumber
    nl = svdj.drop_lone_nodes(nl, el)
    el = el.drop_duplicates(subset=["innode", "outnode", "etype"])
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    return nl, el


def dataflow_feature_extraction(_id, node_ids=None, max_dataflow_dim=None):
    cpg = df.get_cpg(_id)

    # run beginning of dataflow and return input features
    problem = df.ReachingDefinitions(cpg)
    if node_ids is None:
        # Get all nodes
        node_ids = list(cpg.nodes)

    defs = list(sorted(problem.domain))
    # print(_id, len(defs), "defs", defs)
    gen_embeddings = th.zeros((len(node_ids), len(defs)), dtype=th.int)
    kill_embeddings = th.zeros((len(node_ids), len(defs)), dtype=th.int)
    for i, node in enumerate(node_ids):
        gen = problem.gen(node)
        if len(gen) > 0:
            for rd in gen:
                gen_embeddings[i][defs.index(rd)] = 1
        kill = problem.kill(node)
        if len(kill) > 0:
            for rd in kill:
                kill_embeddings[i][defs.index(rd)] = 1
    dataflow_embeddings = th.cat((gen_embeddings, kill_embeddings), axis=1)
    
    # print(_id, dataflow_embeddings.shape, gen_embeddings.sum().item(), kill_embeddings.sum().item())
    
    # pad to max dim
    if max_dataflow_dim is not None:
        # print("pad", dataflow_embeddings.shape[1], "to", max_dataflow_dim)
        pad = th.zeros((dataflow_embeddings.shape[0], max_dataflow_dim))  # Assume 2d
        pad[:, :dataflow_embeddings.size(1)] = dataflow_embeddings
        dataflow_embeddings = pad
    return dataflow_embeddings


def feature_extraction(_id, graph_type="cfgcdg", return_nodes=False, return_node_ids=False, group=True, return_node_types=False):
    """Extract graph feature (basic).

    _id = svddc.svdds.itempath(177775)
    _id = svddc.svdds.itempath(180189)
    _id = svddc.svdds.itempath(178958)

    return_nodes arg is used to get the node information (for empirical evalu
    ation).
    """
    # Get CPG
    n, e = svdj.get_node_edges(_id)
    if group:
        n, e = svd.ne_groupnodes(n, e)
    else:
        n = n[n.lineNumber != ""].copy()
        n.lineNumber = n.lineNumber.astype(int)
        n["nodeId"] = n.id
        e.innode = e.innode.astype(int)
        e.outnode = e.outnode.astype(int)
        n = svdj.drop_lone_nodes(n, e)
        e = e.drop_duplicates(subset=["innode", "outnode", "etype"])

    # Return node metadata
    if return_nodes:
        return n

    # Filter nodes
    e = svdj.rdg(e, graph_type.split("+")[0])
    n = svdj.drop_lone_nodes(n, e)

    # Plot graph
    # svdj.plot_graph_node_edge_df(n, e)

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index()
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # Map edge types
    etypes = e.etype.tolist()
    d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    etypes = [d[i] for i in etypes]

    # Append function name to code
    if "+raw" not in graph_type:
        # try:
        #     func_name = n[n.lineNumber == 1].name.item()
        # except:
        #     # print("file", _id)
        #     func_name = ""
        # n.code = func_name + " " + n.name + " " + "</s>" + " " + n.code
        pass
    else:
        n.code = "</s>" + " " + n.code

    ret = [n.code.tolist(), n.lineNumber.tolist(), e.innode.tolist(), e.outnode.tolist(), etypes]
    if return_node_ids:
        ret.append(n.nodeId.tolist())
    if return_node_types:
        label = n._label.tolist()
        # TODO: replace calls to <operator>.assignment and such with name attribute
        ret.append(label)
    # Return plain-text code, line number list, innodes, outnodes
    return tuple(ret)


# %%
class BigVulDatasetLineVD(svddc.BigVulDataset):
    """IVDetect version of BigVul."""

    def __init__(self, gtype="pdg", feat="all", max_df_dim=None, **kwargs):
        """Init."""
        super(BigVulDatasetLineVD, self).__init__(**kwargs)
        lines = ivde.get_dep_add_lines_bigvul()
        lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
        self.lines = lines
        self.graph_type = gtype
        # glove_path = svd.processed_dir() / "bigvul/glove_False/vectors.txt"
        # self.glove_dict, _ = svdg.glove_dict(glove_path)
        # self.d2v = svdd2v.D2V(svd.processed_dir() / "bigvul/d2v_False")
        self.feat = feat
        self.max_df_dim = max_df_dim

    def item(self, _id, codebert=None, max_dataflow_dim=None):
        """Cache item."""
        
        if max_dataflow_dim is None:
            max_dataflow_dim = self.max_df_dim

        if enable_dataflow:
            savedir = svd.get_dir(
                svd.cache_dir() / f"bigvul_linevd_codebert_dataflow_{self.graph_type}"
            ) / str(_id)
        else:
            savedir = svd.get_dir(
                svd.cache_dir() / f"bigvul_linevd_codebert_{self.graph_type}"
            ) / str(_id)
        # breakpoint()
        if os.path.exists(savedir):
            g = load_graphs(str(savedir))[0][0]
            # g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
            if "_SASTRATS" in g.ndata:
                g.ndata.pop("_SASTRATS")
                g.ndata.pop("_SASTCPP")
                g.ndata.pop("_SASTFF")
            #     g.ndata.pop("_GLOVE")
            #     g.ndata.pop("_DOC2VEC")
            # print(g)
            # breakpoint()
            if g.node_attr_schemes()["_DATAFLOW"].shape[0] != max_dataflow_dim:
                print("wrong shape!", _id, g.node_attr_schemes()["_DATAFLOW"].shape[0], max_dataflow_dim)
            # else:
            if "_CODEBERT" in g.ndata:
                if self.feat == "codebert":
                    for i in ["_GLOVE", "_DOC2VEC", "_RANDFEAT"]:
                        g.ndata.pop(i, None)
                if self.feat == "glove":
                    for i in ["_CODEBERT", "_DOC2VEC", "_RANDFEAT"]:
                        g.ndata.pop(i, None)
                if self.feat == "doc2vec":
                    for i in ["_CODEBERT", "_GLOVE", "_RANDFEAT"]:
                        g.ndata.pop(i, None)
            return g
        # breakpoint()
        code, lineno, ei, eo, et, nids, ntypes = feature_extraction(
            svddc.svdds.itempath(_id), self.graph_type, return_node_ids=True, group=False, return_node_types=True,
        )

        # get dataflow features
        # breakpoint()
        if enable_dataflow:
            dataflow_features = dataflow_feature_extraction(svddc.svdds.itempath(_id), node_ids=nids, max_dataflow_dim=max_dataflow_dim)

        if _id in self.lines:
            vuln = [1 if i in self.lines[_id] else 0 for i in lineno]
        else:
            vuln = [0 for _ in lineno]
        g = dgl.graph((eo, ei))
        #gembeds = th.Tensor(svdg.get_embeddings_list(code, self.glove_dict, 200))
        #g.ndata["_GLOVE"] = gembeds
        #g.ndata["_DOC2VEC"] = th.Tensor([self.d2v.infer(i) for i in code])
        #if codebert:
        #    code = [c.replace("\\t", "").replace("\\n", "") for c in code]
        #    chunked_batches = svd.chunks(code, 128)
        #    features = [codebert.encode(c).detach().cpu() for c in chunked_batches]
        #    g.ndata["_CODEBERT"] = th.cat(features)
        #g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
        g.ndata["_LINE"] = th.Tensor(lineno).int()
        g.ndata["_VULN"] = th.Tensor(vuln).float()

        # TODO: map all label strings to ints
        # g.ndata["_LABEL"] = th.Tensor([label_to_id[l] for l in ntypes]).float()

        # Get dataflow features
        if enable_dataflow:
            # print("Adding dataflow to graph", dataflow_features)
            g.ndata["_DATAFLOW"] = dataflow_features

        # Get SAST labels
        # s = sast.get_sast_lines(svd.processed_dir() / f"bigvul/before/{_id}.c.sast.pkl")
        # rats = [1 if i in s["rats"] else 0 for i in g.ndata["_LINE"]]
        # cppcheck = [1 if i in s["cppcheck"] else 0 for i in g.ndata["_LINE"]]
        # flawfinder = [1 if i in s["flawfinder"] else 0 for i in g.ndata["_LINE"]]
        # g.ndata["_SASTRATS"] = th.tensor(rats).long()
        # g.ndata["_SASTCPP"] = th.tensor(cppcheck).long()
        # g.ndata["_SASTFF"] = th.tensor(flawfinder).long()

        g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
        g.edata["_ETYPE"] = th.Tensor(et).long()
        #emb_path = svd.cache_dir() / f"codebert_method_level/{_id}.pt"
        #g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
        g = dgl.add_self_loop(g)
        save_graphs(str(savedir), [g])
        return g
        
    def get_max_dataflow_dim(self, max_dim=0):
        # Load each graph from file and get the dimension of the dataflow features
#        with Pool(12) as pool:
#            for domain_len in tqdm(pool.imap_unordered(get_dataflow_dim, self.df.sample(len(self.df)).id.tolist()), total=len(self.df), desc="get_max_dataflow_dim"):
#                if domain_len > max_dim:
#                    print("new max", domain_len)
#                    max_dim = domain_len
        for domain_len in tqdm((get_dataflow_dim(_id) for _id in self.df.sample(len(self.df)).id.tolist()), desc="get_max_dataflow_dim"):
            if domain_len > max_dim:
                print("new max", domain_len)
                max_dim = domain_len
        return max_dim

    def cache_items(self, codebert, max_df_dim):
        """Cache all items."""
        for i in tqdm(self.df.sample(len(self.df)).id.tolist(), desc="cache_items"):
            try:
                self.item(i, codebert, max_dataflow_dim=max_df_dim)
            except Exception as E:
                print("cache_items exception", E)

    def cache_codebert_method_level(self, codebert):
        """Cache method-level embeddings using Codebert.

        ONLY NEEDS TO BE RUN ONCE.
        """
        savedir = svd.get_dir(svd.cache_dir() / "codebert_method_level")
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        done = set(done)
        batches = svd.chunks((range(len(self.df))), 128)
        for idx_batch in tqdm(batches, desc="cache_codebert_method_level"):
            batch_texts = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].before.tolist()
            batch_ids = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].id.tolist()
            if set(batch_ids).issubset(done):
                continue
            texts = ["</s> " + ct for ct in batch_texts]
            embedded = codebert.encode(texts).detach().cpu()
            assert len(batch_texts) == len(batch_ids)
            for i in range(len(batch_texts)):
                th.save(embedded[i], savedir / f"{batch_ids[i]}.pt")

    def __getitem__(self, idx):
        """Override getitem."""
        return self.item(self.idx2id[idx])


class BigVulDatasetLineVDDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(
        self,
        batch_size: int = 32,
        sample: int = -1,
        methodlevel: bool = False,
        nsampling: bool = False,
        nsampling_hops: int = 1,
        gtype: str = "cfgcdg",
        splits: str = "default",
        feat: str = "all",
        load_code=False,
    ):
        """Init class from bigvul dataset."""
        super().__init__()
        # codebert = cb.CodeBert()
        # codebert = None
        dataargs = {"sample": sample, "gtype": gtype, "splits": splits, "feat": feat, "load_code": load_code}
        self.train = BigVulDatasetLineVD(partition="train", **dataargs)
        self.val = BigVulDatasetLineVD(partition="val", **dataargs)
        self.test = BigVulDatasetLineVD(partition="test", **dataargs)
        # if enable_dataflow:
        #     max_df_dim = self.train.get_max_dataflow_dim()
        #     print("max_df_dim", max_df_dim)
        #     max_df_dim = self.val.get_max_dataflow_dim(max_df_dim)
        #     print("max_df_dim", max_df_dim)
        #     max_df_dim = self.test.get_max_dataflow_dim(max_df_dim)
        #     print("max_df_dim", max_df_dim)
        max_df_dim = 1058*2
        self.max_df_dim = max_df_dim
        self.train.max_df_dim = max_df_dim
        self.val.max_df_dim = max_df_dim
        self.test.max_df_dim = max_df_dim
        # self.train.cache_codebert_method_level(codebert)
        # self.val.cache_codebert_method_level(codebert)
        # self.test.cache_codebert_method_level(codebert)
        # self.train.cache_items(codebert, max_df_dim)
        # self.val.cache_items(codebert, max_df_dim)
        # self.test.cache_items(codebert, max_df_dim)
        self.batch_size = batch_size
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops

        # del self.train.df
        # del self.val.df
        # del self.test.df

    def node_dl(self, g, shuffle=False):
        """Return node dataloader."""
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nsampling_hops)
        return dgl.dataloading.NodeDataLoader(
            g,
            g.nodes(),
            sampler,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=4,
        )

    def train_dataloader(self):
        """Return train dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.train, batch_size=len(self.train))))
            return self.node_dl(g, shuffle=True)
        return GraphDataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        """Return val dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.val, batch_size=len(self.val))))
            return self.node_dl(g)
        return GraphDataLoader(self.val, batch_size=self.batch_size, num_workers=0)

    def val_graph_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.val, batch_size=32, num_workers=0)

    def test_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.test, batch_size=32, num_workers=0)