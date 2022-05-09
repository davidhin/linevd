# %%
import sys

sys.path.append("/home/benjis/benjis/weile-lab/linevd")
import json
import re
import traceback
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import pandas as pd
import sastvd.analysis.dataflow as dataflow
import sastvd.helpers.datasets as svdds
import sastvd.helpers.dclass as svddc
import tqdm

# %%

def get_dataflow_features(_id):
    try:
        itempath = svddc.BigVulDataset.itempath(_id)
        # print(_id, itempath)
        cpg = dataflow.get_cpg(itempath)
        # print(cpg)

        """
        Get features from definitions
        - Data type: most common k in training dataset
            - Pointer or not
        - Variable name: exclude
        - API call: stdlib (from master list) or OTHER
            - [IBM Docs](https://www.ibm.com/docs/en/i/7.1?topic=extensions-standard-c-library-functions-table-by-name)
            - [The ANSI C Standard Library](https://www.csse.uwa.edu.au/programming/ansic-library.html)
        - Constant: most common k in training dataset
        - Operator: fixed set
        """

        ast = nx.edge_subgraph(
            cpg,
            (
                (u, v, k)
                for u, v, k, attr in cpg.edges(keys=True, data=True)
                if attr["type"] == "AST"
            ),
        )
        arg_graph = nx.edge_subgraph(
            cpg,
            (
                (u, v, k)
                for u, v, k, attr in cpg.edges(keys=True, data=True)
                if attr["type"] == "ARGUMENT"
            ),
        )

        def get_subkey(n_attr):
            if n_attr["_label"] == "LITERAL":
                assert n_attr["code"]
                return "literal", n_attr["code"]
            elif n_attr["_label"] == "CALL":
                # handle operator
                m = re.match(r"<operator>\.(.*)", n_attr["name"])
                if m:
                    return "operator", m.group(1)
                # handle API call
                else:
                    # TODO: May have to use methodFullName
                    return "api", n_attr["name"]

        def get_datatype(n_attr):
            if decl_attr["_label"] == "LOCAL":
                return decl_attr["typeFullName"]
            elif decl_attr["_label"] == "CALL":
                if decl_attr["name"] in (
                    "<operator>.assignment",
                    "<operator>.postIncrement",
                ):
                    args = {
                        cpg.nodes[s]["order"]: s for s in arg_graph.successors(decl)
                    }
                    # print("args", args)
                    var = args[1]
                    var_attr = cpg.nodes[var]
                    if var_attr["_label"] == "IDENTIFIER":
                        return var_attr["typeFullName"]
                    elif (
                        var_attr["_label"] == "CALL"
                        and var_attr["name"] == "<operator>.indirectIndexAccess"
                    ):
                        index_args = {
                            cpg.nodes[s]["order"]: s for s in arg_graph.successors(var)
                        }
                        index = index_args[1]
                        index_attr = cpg.nodes[index]
                        if index_attr["_label"] == "IDENTIFIER":
                            return index_attr["typeFullName"]
                        else:
                            raise NotImplementedError(
                                f"Could not handle {var} {var_attr} -> {index} {index_attr}"
                            )

        def is_decl(n_attr):
            if n_attr["_label"] in ("LOCAL",):
                return True
            elif n_attr["_label"] == "CALL" and n_attr["name"] in (
                "<operator>.assignment",
                "<operator>.postIncrement",
            ):
                return True
            else:
                return False

        # print(cpg.edges(keys=True, data=True))
        features = {}
        decls = [n for n, attr in cpg.nodes(data=True) if is_decl(attr)]
        for decl in decls:
            decl_attr = cpg.nodes[decl]
            # print(decl, decl_attr["code"], end=" ")
            try:
                subkeys = [("node_id", decl)]

                datatype = get_datatype(decl_attr)
                if datatype is not None:
                    subkeys.append(("datatype", datatype))

                ast_children = nx.descendants(ast, decl)
                for n, attr in cpg.nodes(data=True):
                    if n in ast_children:
                        subkey = get_subkey(attr)
                        if subkey is not None:
                            subkeys.append(subkey)

                subkeys = dict(subkeys)
                # print(subkeys)
                features[decl] = subkeys
            except Exception:
                print("error", traceback.format_exc())

        feats_df = pd.DataFrame(list(features.values()))
        feats_df["graph_id"] = _id
        return feats_df
    except Exception:
        print("error", _id, traceback.format_exc())

sample = False
csv_file = Path(f"abstract_dataflow_{sample=}.csv")
if not csv_file.exists():
    output_df = pd.DataFrame()
    df = svdds.bigvul()
    if sample:
        df = df.head(25)  # sample portion
    with Pool(16) as pool:
        for feats_df in tqdm.tqdm(
            pool.imap(get_dataflow_features, df.id),
            total=len(df),
            desc="get abstract dataflow features",
        ):
            output_df = pd.concat([output_df, feats_df], ignore_index=True)
    output_df.to_csv(csv_file)
    df = output_df

else:
    df = pd.read_csv(csv_file)

print(df)
print(df.value_counts("datatype"))
print(df.value_counts("literal"))
print(df.value_counts("api"))
print(df.value_counts("operator"))

# %%
how_many_select = 10
select = {
    "datatype": df["datatype"].value_counts().nlargest(how_many_select).index.tolist(),
    "literal": df["literal"].value_counts().nlargest(how_many_select).index.tolist(),
    "operator": df["operator"].value_counts().nlargest(how_many_select).index.tolist(),
    "api": df["api"].value_counts().nlargest(how_many_select).index.tolist(),
}
# print(json.dumps(select, indent=2))
print(select)

def to_hash(row):
    try:
        datatype_idx = select["datatype"].index(row["datatype"])
    except ValueError:
        datatype_idx = -1
    try:
        literal_idx = select["literal"].index(row["literal"])
    except ValueError:
        literal_idx = -1
    try:
        operator_idx = select["operator"].index(row["operator"])
    except ValueError:
        operator_idx = -1
    try:
        api_idx = select["api"].index(row["api"])
    except ValueError:
        api_idx = -1
    # TODO: pad digits?

    # combine
    return f"{datatype_idx} {literal_idx} {operator_idx} {api_idx}"


df["hash"] = df.apply(to_hash, axis=1)
print(df["hash"])
print(df.value_counts("hash"))
items_with_missing = sum(df["hash"].str.contains("-1"))

df.to_csv(f"abstract_dataflow_hash_{sample=}.csv")

# missing_df = df[df["hash"].str.contains("-1")]
# nmissing_df = df[~df["hash"].str.contains("-1")]
# print(missing_df)
# print(nmissing_df)
