"""
Extract abstract dataflow features from graphs
"""

import argparse
import functools
import itertools
import sys

sys.path.append("/home/benjis/benjis/weile-lab/linevd")
import re
import traceback
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import pandas as pd
import code_gnn.analysis.dataflow as dataflow
import sastvd.helpers.datasets as svdds
import sastvd.helpers.dclass as svddc
import sastvd.helpers.joern as svdj
import sastvd.helpers.joern_session as svdjs
import sastvd as svd
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt
import json
import pexpect

sample = False

# Extract dataflow features from CPG

all_assignment_types = (
    "<operator>.assignmentDivision",
    "<operator>.assignmentExponentiation",
    "<operator>.assignmentPlus",
    "<operator>.assignmentMinus",
    "<operator>.assignmentModulo",
    "<operator>.assignmentMultiplication",
    "<operator>.preIncrement",
    "<operator>.preDecrement",
    "<operator>.postIncrement",
    "<operator>.postDecrement",
    "<operator>.assignment",
    "<operator>.assignmentOr",
    "<operator>.assignmentAnd",
    "<operator>.assignmentXor",
    "<operator>.assignmentArithmeticShiftRight",
    "<operator>.assignmentLogicalShiftRight",
    "<operator>.assignmentShiftLeft",
)


def is_decl(n_attr):
    # NOTE: this is local variable declarationsm
    # which are not considered definitions in formal DFA setting.
    # if n_attr["_label"] in ("LOCAL",):
    #     return True

    # https://github.com/joernio/joern/blob/15e241d3174ecba9e977a399793c9c6a1249d819/semanticcpg/src/main/scala/io/shiftleft/semanticcpg/language/operatorextension/package.scala
    return n_attr["_label"] == "CALL" and n_attr["name"] in all_assignment_types


def get_dataflow_features(graph_id, raise_all=False, verbose=False):
    try:
        cpg, n, e = dataflow.get_cpg(graph_id, return_n_e=True)
        ast = dataflow.sub(cpg, "AST")
        arg_graph = dataflow.sub(cpg, "ARGUMENT")
        labels = nx.get_node_attributes(cpg, "_label")
        code = nx.get_node_attributes(cpg, "code")
        names = nx.get_node_attributes(cpg, "name")

        def recurse_datatype(v):
            v_attr = cpg.nodes[v]
            if verbose:
                print("recursing", v, v_attr)

            name_idx = {
                "<operator>.indirectIndexAccess": 1,
                "<operator>.indirectFieldAccess": 1,
                "<operator>.indirection": 1,
                "<operator>.fieldAccess": 1,
                "<operator>.postIncrement": 1,
                "<operator>.postDecrement": 1,
                "<operator>.preIncrement": 1,
                "<operator>.preDecrement": 1,
                "<operator>.addressOf": 1,
                "<operator>.cast": 2,
                "<operator>.addition": 1,
            }
            if v_attr["_label"] == "IDENTIFIER":
                return v, v_attr["typeFullName"]
            elif v_attr["_label"] == "CALL":
                if v_attr["name"] in name_idx.keys():
                    # TODO: Get field data type, not struct data type
                    args = {cpg.nodes[s]["order"]: s for s in arg_graph.successors(v)}
                    arg = args[name_idx[v_attr["name"]]]
                    arg_attr = cpg.nodes[arg]
                    if verbose:
                        print("index", arg, arg_attr)
                        if v_attr["name"] == "<operator>.addition":
                            print("addition debug", v, v_attr, arg, arg_attr)
                    if arg_attr["_label"] == "IDENTIFIER":
                        return arg, arg_attr["typeFullName"]
                    elif arg_attr["_label"] == "CALL":
                        return recurse_datatype(arg)
                    else:
                        raise NotImplementedError(
                            f"recurse_datatype index could not handle {v} {v_attr} -> {arg} {arg_attr}"
                        )
            raise NotImplementedError(
                f"recurse_datatype var could not handle {v} {v_attr}"
            )

        def get_raw_datatype(decl):
            decl_attr = cpg.nodes[decl]

            if verbose:
                print("parent", decl, decl_attr)

            if decl_attr["_label"] == "LOCAL":
                return decl, decl_attr["typeFullName"]
            elif decl_attr["_label"] == "CALL" and decl_attr[
                "name"
            ] in all_assignment_types + ("<operator>.cast",):
                args = {cpg.nodes[s]["order"]: s for s in arg_graph.successors(decl)}
                return recurse_datatype(args[1])
            else:
                raise NotImplementedError(
                    f"""get_raw_datatype did not handle {decl} {decl_attr}"""
                )

        def grab_declfeats(node_id):
            fields = []
            try:
                ret = get_raw_datatype(node_id)
                if ret is not None:
                    child_id, child_datatype = ret
                    fields.append(("datatype", child_id, child_datatype))

                # create a copy of the AST with method definitions excluded.
                # this avoids an issue where some variable definitions descend to
                # method definitions (probably by mistake), shown in graph 3.
                my_ast = ast.copy()
                my_ast.remove_nodes_from(
                    [
                        n
                        for n, attr in ast.nodes(data=True)
                        if attr["_label"] == "METHOD"
                    ]
                )

                to_search = nx.descendants(my_ast, node_id)
                for n in to_search:
                    if verbose:
                        print(
                            f"{node_id} desc {n} {code.get(n, None)} {names.get(n, None)} {nx.shortest_path(ast, node_id, n)}"
                        )
                    if labels[n] == "LITERAL":
                        fields.append(("literal", n, code.get(n, pd.NA)))
                    if labels[n] == "CALL":
                        if m := re.match(r"<operator>\.(.*)", names[n]):
                            operator_name = m.group(1)
                            if operator_name not in ("indirection",):
                                fields.append(("operator", n, operator_name))
                        # handle API call
                        else:
                            fields.append(("api", n, names[n]))
            except Exception:
                print("node error", node_id, traceback.format_exc())
                if raise_all:
                    raise
            return fields

        nx.set_node_attributes(
            ast,
            {n: f"{n}: {attr['code']}" for n, attr in ast.nodes(data=True)},
            "label",
        )
        A = nx.drawing.nx_agraph.to_agraph(ast)
        A.layout("dot")
        A.draw("abcd.png")

        n = n.rename(columns={"id": "node_id"})
        n["graph_id"] = graph_id
        decls = n[
            n["node_id"].isin(n for n, attr in cpg.nodes(data=True) if is_decl(attr))
        ].copy()
        decls["fields"] = decls["node_id"].apply(grab_declfeats)
        decls = decls.explode("fields")
        decls["subkey"], decls["subkey_node_id"], decls["subkey_text"] = zip(
            *decls["fields"]
        )
        return decls
    except Exception:
        print("graph error", graph_id, traceback.format_exc())
        if raise_all:
            raise


# Get all abstract dataflow info
def get_dataflow_features_df():
    csv_file = (
        svd.cache_dir() / f"bigvul/abstract_dataflow{'_sample' if sample else ''}.csv"
    )
    if csv_file.exists() and args.cache:
        dataflow_df = pd.read_csv(csv_file)
    else:
        dataflow_df = pd.DataFrame()
        all_df = svdds.bigvul(sample=args.sample)
        with Pool(args.workers) as pool:
            for decls_df in tqdm.tqdm(
                pool.imap(
                    functools.partial(
                        get_dataflow_features,
                        raise_all=args.sample,
                        verbose=args.verbose,
                    ),
                    all_df.id,
                ),
                total=len(all_df),
                desc="get abstract dataflow features",
            ):
                dataflow_df = pd.concat([dataflow_df, decls_df], ignore_index=True)

        dataflow_df = dataflow_df[
            ["graph_id", "node_id", "subkey", "subkey_node_id", "subkey_text"]
        ]

        dataflow_df.to_csv(csv_file)

    dataflow_df.to_csv("abstract_dataflow_fixed.csv")

    return dataflow_df


def cleanup_datatype(df):
    """Assign datatype to cleaned-up version"""
    df.loc[df["subkey"] == "datatype", "subkey_text"] = dataflow_df[
        "subkey_text"
    ].apply(
        lambda dt: dt
        if pd.isna(dt)
        else re.sub(
            r"\s+", r" ", re.sub(r"^const ", r"", re.sub(r"\s*\[.*\]", r"[]", dt))
        ).strip()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abstract dataflow")
    parser.add_argument("--sample", action="store_true", help="Extract sample only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.set_defaults(cache=True)
    parser.add_argument(
        "--workers", type=int, default=6, help="How many workers to use"
    )
    args = parser.parse_args()

    dataflow_df = get_dataflow_features_df()
    print("dataflow_df", dataflow_df)
    print("dataflow_df counts", dataflow_df.value_counts("subkey"))
    print("dataflow_df na", dataflow_df[dataflow_df["subkey_text"].isna()])
    exit()

# def extract_nan_values():
#     dataflow_df = get_dataflow_features_df()
#     nandt_df = dataflow_df[dataflow_df["datatype"].isna()]
#     nandt_df.to_csv("abstract_dataflow_nandatatype.csv")
#     print(nandt_df)
#     # for i, row in nandt_df.head().iterrows():
#     #     print(svddc.BigVulDataset.itempath(row.name), row.node_id)ad(25)  # sample portion
#     df = pd.DataFrame()
#     with Pool(16) as pool:
#         ids = nandt_df.graph_id.unique()
#         for feats_df in tqdm.tqdm(
#             pool.imap(get_dataflow_features, ids),
#             total=len(ids),
#             desc="re-extract NaN abstract dataflow features",
#         ):
#             if "datatype" in feats_df.columns:
#                 na = feats_df[feats_df["datatype"].isna()]
#                 if len(na) > 0:
#                     print("NaN values:", na)
#             # df.loc[df["datatype"] == "<<<INVALID>>>"]["datatype"] = np.nan
#             df.replace('N/A', pd.NA)
#             df = pd.concat([df, feats_df], ignore_index=True)
#     df = df[df["node_id"].isin(nandt_df["node_id"])]
#     df.to_csv("abstract_dataflow_nandatatype_fixed.csv")

# def merge_nan_values():
#     df = pd.read_csv("abstract_dataflow copy.csv")
#     fixed_df = pd.read_csv("abstract_dataflow_nandatatype_fixed.csv")
#     print("loaded")
#     assigned = 0
#     processed = 0
#     node_id_to_datatype = dict(zip(zip(fixed_df["graph_id"], fixed_df["node_id"]), fixed_df["datatype"]))
#     print("before", df["datatype"].value_counts(dropna=False))
#     print(df["datatype"].isna().sum(), "na")
#     for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
#         # print(i, row)
#         # matching = node_id_to_datatype[row["node_id"]]
#         key = (row["graph_id"], row["node_id"])
#         if key in node_id_to_datatype:
#             new_datatype = node_id_to_datatype[key]
#             if isinstance(row["datatype"], str):
#                 assert row["datatype"] == new_datatype, (row["datatype"], new_datatype)
#             else:
#                 df.at[i, "datatype"] = new_datatype
#                 if assigned < 10:
#                     print("assigning", assigned, key, row["datatype"], new_datatype)
#                 assigned += 1
#         processed += 1
#     print("after", df["datatype"].value_counts(dropna=False))
#     print(df["datatype"].isna().sum(), "na")
#     print("assigned", assigned, "out of", len(df))
#     # df = pd.merge(df, fixed_df, on="node_id")
#     df.to_csv("abstract_dataflow.csv")

# %% Extract nodes from all graphs

# def get_nodes(_id):
#     itempath = svddc.BigVulDataset.itempath(_id)
#     n, e = svdj.get_node_edges(itempath)
#     return pd.DataFrame(list(zip(n["_label"], n["name"], n["id"])), columns=["_label", "name", "id"])

# if __name__ == "__main__":
#     node_df_file = Path("node_df.csv")
#     if node_df_file.exists():
#         node_df = pd.read_csv(node_df_file)
#     else:
#         print("Loading node df")
#         with Pool(16) as pool:
#             node_df = None
#             ids = dataflow_df["graph_id"].unique()
#             for i, d in enumerate(pool.imap_unordered(get_nodes, tqdm.tqdm(ids, desc="get aux node info", total=len(ids)))):
#                 if i < 3:
#                     print(f"df {i}: {node_df}")
#                 if node_df is None:
#                     node_df = d
#                 else:
#                     node_df = pd.concat((node_df, d), ignore_index=True)
#         print("writing to csv")
#         node_df.to_csv(node_df_file)
#     node_df["id"] = node_df["id"].astype(int)

#%% Get most common subkeys

if __name__ == "__main__":
    select_key = "datatype"

if __name__ == "__main__":
    dataargs = {"sample": -1, "splits": "default", "load_code": False}
    train = svddc.BigVulDataset(partition="train", **dataargs)
    train_df = train.df
    print(f"{train_df.columns=}")
    print("generate hash from train", train_df)

    train_df = pd.merge(train_df, dataflow_df, left_on="id", right_on="graph_id")
    print(train_df.columns)
    datatype_vc = train_df[select_key].value_counts()
    print(datatype_vc)
    # datatype_vc = train_df["datatype_subtypes_str"].value_counts()

    # Filter to decls only
    # NOTE: disabled because already done in get_dataflow_features... get a hint!
    # df = pd.merge(df, node_df, on="id")
    # print(df.columns)
    # print(df)
    # df["is_decl"] = df.apply(is_decl, axis=1)
    # df = df[df["is_decl"]]

    # how_many_select = 10
    # select = {
    #     "datatype": df["datatype"].value_counts().nlargest(how_many_select).index.tolist(),
    #     "literal": df["literal"].value_counts().nlargest(how_many_select).index.tolist(),
    #     "operator": df["operator"].value_counts().nlargest(how_many_select).index.tolist(),
    #     "api": df["api"].value_counts().nlargest(how_many_select).index.tolist(),
    # }
    # for feat in ("datatype", "literal", "operator", "api"):
    #     # sns.countplot(x=feat, data=df)
    #     vc = df[feat].value_counts(normalize=True)
    #     threshold = 0.005
    #     mask = vc > threshold
    #     tail_prob = vc.loc[~mask].sum()
    #     vc = vc.loc[mask]
    #     vc['other 0.5%'] = tail_prob
    #     vc.plot(kind='bar')
    #     plt.xticks(rotation=25)
    #     plt.show()
    #     plt.savefig(f"hist_{feat}.png")
    #     plt.close()
    # Get datatype only
    # how_many_select = 500
    # select = {
    #     "datatype": df["datatype"].value_counts().nlargest(how_many_select).index.tolist(),
    # }
    # Get all
    # select = {
    #     "datatype": train_df["datatype"].unique().tolist(),
    # }
    # for k in select:
    #     print(k, len(select[k]), "items selected")
    #     print("headdd", list(select[k])[:10])
    # exit()
    """
Output from first run:
{
"datatype": [
    "int",
    "size_t",
    "ssize_t",
    "ANY",
    "char *",
    "unsigned char *",
    "bool",
    "unsigned int",
    "const char *",
    "Image *"
],
"literal": [
    "0",
    "1",
    "2",
    "4",
    "8",
    "3",
    "16",
    "'\\0'",
    "7",
    "5"
],
"operator": [
    "indirectFieldAccess",
    "fieldAccess",
    "indirectIndexAccess",
    "cast",
    "indirection",
    "addressOf",
    "addition",
    "minus",
    "postIncrement",
    "subtraction"
],
"api": [
    "RelinquishMagickMemory",
    "ReadBlobByte",
    "DestroyImageList",
    "strlen",
    "SetImageProgress",
    "data.readInt32",
    "ReadBlob",
    "malloc",
    "ReadBlobLSBShort",
    "AcquireQuantumMemory"
]
}
    """

# %% Generate hash value for each node


def to_hash(row, select):
    items = []
    for key in select:
        items.append(select[key].index(row[key]) if row[key] in select[key] else pd.NA)

    # TODO: pad digits?

    # combine
    return " ".join(map(str, items))


if __name__ == "__main__":
    missing = []
    all_df = None
    split_df = svddc.BigVulDataset(partition="train", **dataargs).df
    split_df = pd.merge(split_df, dataflow_df, left_on="id", right_on="graph_id")
    portions = [
        10,
        50,
        100,
        250,
        500,
        750,
        1000,
        1500,
        2000,
        len(split_df[select_key].drop_duplicates()),
    ]
    for portion in tqdm.tqdm(
        portions, desc="measuring train coverage for various portions..."
    ):
        select = {
            select_key: datatype_vc.nlargest(portion).index.sort_values().tolist(),
        }
        split_na = split_df.apply(functools.partial(to_hash, select=select), axis=1)
        missing.append(
            (len(split_df) - split_na.replace("<NA>", pd.NA).isna().sum())
            / len(split_df)
        )
    portions[-1] = f"{portions[-1]} (all values)"
    sns.barplot(x=portions, y=missing)
    # sns.scatterplot(portions, missing)
    plt.xlabel("top k values hashed")
    plt.xticks(rotation=45)
    plt.ylabel("portion of training dataset which could be hashed")
    plt.tight_layout()
    plt.savefig("abstract_dataflow_missing_portions.png")
    plt.close()


if __name__ == "__main__":

    # Export dataset
    select = {
        select_key: datatype_vc.nlargest(500).index.sort_values().tolist(),
    }
    all_df = None
    for split in ("train", "val", "test"):
        split_df = svddc.BigVulDataset(partition=split, **dataargs).df
        split_df = pd.merge(split_df, dataflow_df, left_on="id", right_on="graph_id")
        split_df["hash"] = split_df.apply(to_hash, axis=1, select=select).replace(
            "<NA>", pd.NA
        )
        split_df = (
            split_df[["graph_id", "node_id", "hash"]]
            .sort_values(by=["graph_id", "node_id"])
            .reset_index(drop=True)
        )
        split_df["node_id"] = split_df["node_id"].astype(int)
        print(
            split,
            len(split_df),
            split_df["hash"].value_counts(dropna=False, normalize=True),
        )
        # split_df.to_csv(f"abstract_dataflow_hash_all.csv")
        if all_df is None:
            all_df = split_df
        else:
            all_df = pd.concat((all_df, split_df), ignore_index=True)
    all_df.to_csv(f"abstract_dataflow_hash_all.csv")
    exit()

# if __name__ == "__main__":
#     df["hash"] = df.apply(to_hash, axis=1)
#     print(df["hash"])
#     print(df.value_counts("hash"))
#     items_with_missing = sum(df["hash"].str.contains("-1"))

#     df.to_csv(f"abstract_dataflow_hash{'_sample' if sample else ''}.csv")

# %% generate frequency graphs


def generate_frequency_graphs():
    test = svddc.BigVulDataset(partition="test", **dataargs)
    test_df = test.df
    test_df = pd.merge(test_df, dataflow_df, left_on="id", right_on="graph_id")
    print("select from", datatype_vc)
    hhvc_true = {}
    for i, portion in enumerate(range(10, 101, 10)):
        portion = portion / 100
        number = int(len(datatype_vc) * portion)
        print(
            "ITERATION", i, "portion", portion, "number", number, "/", len(datatype_vc)
        )
        select = {
            select_key: datatype_vc.nlargest(number).index.sort_values().tolist(),
        }
        for k in select:
            print(k, len(select[k]), "items selected")
            print("headdd", list(select[k])[:10])
        # print(test_df)
        test_df["hash"] = test_df.apply(to_hash, select=select, axis=1).replace(
            "<NA>", pd.NA
        )
        print(test_df["hash"])
        # breakpoint()
        test_df["has_hash"] = ~test_df["hash"].isna()
        hhvc = test_df["has_hash"].value_counts()
        print("extract", i)
        print(hhvc)
        hhvc_true[portion] = hhvc[1]
        # print(hhvc_true)
    print(hhvc_true)
    # sns.barplot(x=hhvc_true.keys(), y=hhvc_true.values())
    plt.bar(
        [
            str(int(k * 100)) + "% (" + str(int(len(datatype_vc) * k)) + ")"
            for k in hhvc_true.keys()
        ],
        hhvc_true.values(),
    )
    plt.axhline(y=len(test_df), color="r", linestyle="-")
    plt.xlabel("portion of hashes from training dataset")
    plt.ylabel("number of hashed test examples")
    plt.xticks(rotation=45)
    plt.title("datatype only, keyed by raw type name")
    plt.tight_layout()
    plt.savefig("has_hash_test_portions.png")
    plt.close()


# %%

if __name__ == "__main__":
    generate_frequency_graphs()
    exit()


def generate_dataset_old():
    # hashes = sorted(df["hash"].unique().tolist())
    del df
    train = svddc.BigVulDataset(partition="train", **dataargs)
    val = svddc.BigVulDataset(partition="val", **dataargs)
    test = svddc.BigVulDataset(partition="test", **dataargs)
    for i, split_df in enumerate((val.df, test.df, train.df)):
        # print("extract", i, "split", i)
        split_df = pd.merge(split_df, dataflow_df, left_on="id", right_on="graph_id")
        # split_df = pd.merge(split_df, node_df, left_on="node_id", right_on="id")
        # split_df = split_df[split_df.apply(is_decl, axis=1)]
        split_df["hash"] = split_df.apply(to_hash, axis=1)
        # print("extract", i, split_df["hash"])
        split_df["has_hash"] = split_df["hash"].isin(hashes)
        print("extract", i, split_df["has_hash"].value_counts())

        # mask = split_df["hash"] == "-1"
        # tail_prob = (mask).sum() /
        # print(f"tail prob {i}: {tail_prob}")
        # prob = split_df["hash"].value_counts(normalize=True)
        # prob['<UNKNOWN>'] = tail_prob
        # print("extract", i, prob)
        # prob.plot(kind='bar')
        # # percentages_map = prob.to_dict()
        # # split_df["has_hash_percent"] = split_df["hash"].map(percentages_map)
        # # print(split_df["has_hash_percent"])

        # # split_df["hash_index"] = split_df["hash"].apply(lambda h: hashes.index(h) if h in hashes else -1)
        # # split_df.hist(column="hash_index")
        # plt.xticks(rotation=25)
        # plt.savefig(f"hash_{i}.png")
        # plt.close()

        # split_df.hist(column="has_hash_percent")
        # plt.savefig(f"has_hash_percent_{i}.png")
        # plt.close()
