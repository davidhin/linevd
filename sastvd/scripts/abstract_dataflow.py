# %%
import sys

sys.path.append("/home/benjis/benjis/weile-lab/linevd")
import re
import traceback
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import sastvd.analysis.dataflow as dataflow
import sastvd.helpers.datasets as svdds
import sastvd.helpers.dclass as svddc
import sastvd.helpers.joern as svdj
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt

sample = False

# %% Extract dataflow features from CPG

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

        def recurse_datatype(v, verbose=False):
            var_attr = cpg.nodes[v]
            if verbose:
                print("recursing", v, var_attr)
            
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
            }
            # blacklist = ["PS", "bgp_attr_extra_get", "STACK_OF"]
            if var_attr["_label"] == "IDENTIFIER":
                return var_attr["typeFullName"]
            elif var_attr["_label"] == "CALL":
                if var_attr["name"] in name_idx.keys():
                    # TODO: Get field data type, not struct data type
                    index_args = {
                        cpg.nodes[s]["order"]: s for s in arg_graph.successors(v)
                    }
                    index = index_args[name_idx[var_attr["name"]]]
                    index_attr = cpg.nodes[index]
                    if verbose:
                        print("index", index, index_attr)
                    if index_attr["_label"] == "IDENTIFIER":
                        return index_attr["typeFullName"]
                    elif index_attr["_label"] == "CALL":
                        return recurse_datatype(index, verbose)
                    else:
                        raise NotImplementedError(
                            f"recurse_datatype index could not handle {v} {var_attr} -> {index} {index_attr}"
                        )
                # elif var_attr["name"] in blacklist:
                else:
                    print(f"""{_id} blacklisted {var_attr["name"]} {v} {var_attr}""")
                    return "N/A"
            raise NotImplementedError(
                f"recurse_datatype var could not handle {v} {var_attr}"
            )

        def get_raw_datatype(decl):
            decl_attr = cpg.nodes[decl]

            # NOTE: debug
            verbose = decl in [
                # gid 401
                # 1000129,

                # gid 1273
                # 1000336,
                # 1000259,
                # 1000213,
            ]

            if verbose:
                print("parent", decl, decl_attr)
            if decl_attr["_label"] == "LOCAL":
                return decl_attr["typeFullName"]
            elif decl_attr["_label"] == "CALL":
                if decl_attr["name"] in (
                    "<operator>.assignment",
                    "<operator>.postIncrement",
                    "<operator>.cast",
                ):
                    args = {
                        cpg.nodes[s]["order"]: s for s in arg_graph.successors(decl)
                    }
                    return recurse_datatype(args[1], verbose)
            else:
                raise NotImplementedError(
                    f"""get_raw_datatype did not handle {decl} {decl_attr}"""
                )

        def get_datatype(n_attr):
            dt = get_raw_datatype(n_attr)
            # print("raw datatype:", dt)
            # if dt == "":
            #     pass  # decompose into component data types
            return dt

        features = {}
        decls = [n for n, attr in cpg.nodes(data=True) if is_decl(attr)]
        for decl in decls:
            subkeys = [("node_id", decl)]
            try:
                datatype = get_datatype(decl)
                # print(decl, "output datatype:", datatype)
                if datatype is not None:
                    subkeys.append(("datatype", datatype))

                ast_children = nx.descendants(ast, decl)
                for n, attr in cpg.nodes(data=True):
                    if n in ast_children:
                        subkey = get_subkey(attr)
                        if subkey is not None:
                            subkeys.append(subkey)
            except Exception:
                print("node error", decl, traceback.format_exc())
            subkeys = dict(subkeys)
            features[decl] = subkeys

        feats_df = pd.DataFrame(list(features.values()))
        # feats_df["node_id"] = feats_df["node_id"].astype(int)
        feats_df["graph_id"] = _id
        return feats_df
    except Exception:
        print("graph error", _id, traceback.format_exc())

"""
Getting types from full project
1. Load example source code (function)
2. Locate full project - assume already downloaded
3. Parse full project with Joern
4. For function in code:
  a. Locate function in CPG
  b. Locate all definitions
  c. Locate all types in definition
  d. Recursively gather all field types


## get_type.sc gets the leaf types of a type you give it.
Potential issues left:
- [x] Types may not be selected correctly if they are typedef'd to an anonymous type.
- For types which are aliased to an anonymous type, we select the anonymous type
    based on the index of anonymous types in the same file, sorted by index.
    This might not be correct.
- Function pointer types are represented as external TypeDecls in Joern, but should be decomposed to their parameter/return types.

## DEFINE problem

Without providing the correct DEFINEs to Joern, they will not be included in the CPG. For example:

#if HAVE_PCRE || HAVE_BUNDLED_PCRE
/// ...code...
typedef struct {
    pcre *re;
    pcre_extra *extra;
    int preg_options;
#if HAVE_SETLOCALE
    char *locale;
    unsigned const char *tables;
#endif
    int compile_options;
    int refcount;
} pcre_cache_entry; // <--- not included in CPG
/// ...code...

This can be fixed by providing DEFINEs:

joern/joern-cli/c2cpg.sh php-src/ext/pcre/php_pcre.h \
    --output php_pcre_HAVE_PCRE_1.h.cpg.bin.zip \
    --define HAVE_PCRE=1

But we don't want to have to give these. For now, just parse the whole project without DEFINEs
and get the type decls that we can.
"""

def test_get_dataflow_features():
    # NOTE: this code is needed to get the commit ID/URL from the metadata
    row = df.iloc[0]
    my_id = row.id
    print("id:", my_id)
    print("row:", row)

    """
Here is what the types look like in nodes.json:
    {
        "name": "zval *",
        "fullName": "zval *",
        "typeDeclFullName": "zval *",
        "id": 115,
        "_label": "TYPE"
    },
    {
        "name": "zval * *",
        "fullName": "zval * *",
        "typeDeclFullName": "zval * *",
        "id": 116,
        "_label": "TYPE"
    },

In order to load the member types of struct types, we have to load the entire project.
Try with example ID 177737.
Attributes:
    sastvd/scripts/abstract_dataflow.py id: 177737
    row: idx                                                                               0
    dataset                                                                      bigvul
    id                                                                           177737
    label                                                                         train
    vul_x                                                                             1
    Access Gained_x                                                                None
    Attack Origin_x                                                              Remote
    Authentication Required_x                                              Not required
    Availability_x                                                              Partial
    CVE ID_x                                                              CVE-2015-8382
    CVE Page_x                            https://www.cvedetails.com/cve/CVE-2015-8382/
    CWE ID_x                                                                    CWE-119
    Complexity_x                                                                    Low
    Confidentiality_x                                                           Partial
    Integrity_x                                                                    None
    Known Exploits_x                                                                NaN
    Publish Date_x                                                           2015-12-01
    Score_x                                                                         6.4
    Summary_x                         The match function in pcre_exec.c in PCRE befo...
    Update Date_x                                                            2016-12-27
    Vulnerability Classification_x                                   DoS Overflow +Info
    project_x                                                                       php
    Access Gained_y                                                                None
    Attack Origin_y                                                              Remote
    Authentication Required_y                                              Not required
    Availability_y                                                              Partial
    CVE ID_y                                                              CVE-2015-8382
    CVE Page_y                            https://www.cvedetails.com/cve/CVE-2015-8382/
    CWE ID_y                                                                    CWE-119
    Complexity_y                                                                    Low
    Confidentiality_y                                                           Partial
    Integrity_y                                                                    None
    Known Exploits_y                                                                NaN
    Publish Date_y                                                           2015-12-01
    Score_y                                                                         6.4
    Summary_y                         The match function in pcre_exec.c in PCRE befo...
    Update Date_y                                                            2016-12-27
    Vulnerability Classification_y                                   DoS Overflow +Info
    add_lines                                                                         1
    codeLink                          https://git.php.net/?p=php-src.git;a=commit;h=...
    commit_id                                  c351b47ce85a3a147cfa801fa9f0149ab4160834
    commit_message                                                                  NaN
    del_lines                                                                         0
    file_name                                                                       NaN
    files_changed                                                                   NaN
    func_after                        PHPAPI void php_pcre_match_impl(pcre_cache_ent...
    func_before                       PHPAPI void php_pcre_match_impl(pcre_cache_ent...
    lang                                                                              C
    lines_after                              memset(offsets, 0, size_offsets*sizeof(...
    lines_before                                                                    NaN
    parentID                                   1a2ec3fc60e428c47fd59c9dd7966c71ca44024d
    patch                             @@ -640,7 +640,7 @@ PHPAPI void php_pcre_match...
    project_y                                                                       php
    project_after                     https://git.php.net/?p=php-src.git;a=blob;f=ex...
    project_before                    https://git.php.net/?p=php-src.git;a=blob;f=ex...
    vul_y                                                                             1
    vul_func_with_fix                 PHPAPI void php_pcre_match_impl(pcre_cache_ent...
    Name: 0, dtype: object

Try to run this:
    importCode("php-src", "php-src_c351b47ce85a3a147cfa801fa9f0149ab4160834")
First run ran out of memory.

After load, try to query all types in project.
q: How to link types in project to types in code fragment? Node ids are not connected.
a: Just match by strings. Type names should be unique.
  If there are major problems with this, then also maybe match the function name
  of the code fragment and back-match the type name.

q: How to know if a variable type is a struct in the first place? We would have to parse
  all the code in all the projects to get all type definitions.
a: guess we don't know
    """

    feat = get_dataflow_features(my_id)
    print(feat)

# %% Get all abstract dataflow info
def get_dataflow_features_df():
    sample = False
    csv_file = Path(f"abstract_dataflow{'_sample' if sample else ''}.csv")
    if not csv_file.exists():
        dataflow_df = pd.DataFrame()
        all_df = svdds.bigvul()
        if sample:
            all_df = all_df.head(25)  # sample portion
        with Pool(16) as pool:
            for feats_df in tqdm.tqdm(
                pool.imap(get_dataflow_features, all_df.id),
                total=len(all_df),
                desc="get abstract dataflow features",
            ):
                dataflow_df = pd.concat([dataflow_df, feats_df], ignore_index=True)
        dataflow_df.to_csv(csv_file)

    else:
        dataflow_df = pd.read_csv(csv_file)

    print(dataflow_df)
    print(dataflow_df.value_counts("datatype"))
    print(dataflow_df.value_counts("literal"))
    print(dataflow_df.value_counts("api"))
    print(dataflow_df.value_counts("operator"))

    return dataflow_df

def expand_struct_datatypes(df):
    md_df = pd.read_csv("bigvul_metadata_with_commit_id.csv")
    df = pd.merge(df, md_df, on="id")
    checkout_dir = Path("repos/checkout")
    df["projectpath"] = df["repo"].apply(lambda r: checkout_dir/(r.replace("/", "__")))

    sess = svdjs.JoernSession(i)
    def get_dataflow_features_with_sess(row):
        return run_joern_gettype(sess, row["projectpath"], row["datatype"])
    try:
        df.apply(get_dataflow_features_with_sess)
    finally:
        sess.close()

def get_expanded_df():
    dataflow_df = get_dataflow_features_df()
    dataflow_df = expand_struct_datatypes(dataflow_df)
    return dataflow_df

if __name__ == "__main__":
    dataflow_df = get_expanded_df()

def extract_nan_values():
    dataflow_df = get_dataflow_features_df()
    nandt_df = dataflow_df[dataflow_df["datatype"].isna()]
    nandt_df.to_csv("abstract_dataflow_nandatatype.csv")
    print(nandt_df)
    # for i, row in nandt_df.head().iterrows():
    #     print(svddc.BigVulDataset.itempath(row.name), row.node_id)ad(25)  # sample portion
    df = pd.DataFrame()
    with Pool(16) as pool:
        ids = nandt_df.graph_id.unique()
        for feats_df in tqdm.tqdm(
            pool.imap(get_dataflow_features, ids),
            total=len(ids),
            desc="re-extract NaN abstract dataflow features",
        ):
            if "datatype" in feats_df.columns:
                na = feats_df[feats_df["datatype"].isna()]
                if len(na) > 0:
                    print("NaN values:", na)
            # df.loc[df["datatype"] == "<<<INVALID>>>"]["datatype"] = np.nan
            df.replace('N/A', pd.NA)
            df = pd.concat([df, feats_df], ignore_index=True)
    df = df[df["node_id"].isin(nandt_df["node_id"])]
    df.to_csv("abstract_dataflow_nandatatype_fixed.csv")

def merge_nan_values():
    df = pd.read_csv("abstract_dataflow copy.csv")
    fixed_df = pd.read_csv("abstract_dataflow_nandatatype_fixed.csv")
    print("loaded")
    assigned = 0
    processed = 0
    node_id_to_datatype = dict(zip(zip(fixed_df["graph_id"], fixed_df["node_id"]), fixed_df["datatype"]))
    print("before", df["datatype"].value_counts(dropna=False))
    print(df["datatype"].isna().sum(), "na")
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        # print(i, row)
        # matching = node_id_to_datatype[row["node_id"]]
        key = (row["graph_id"], row["node_id"])
        if key in node_id_to_datatype:
            new_datatype = node_id_to_datatype[key]
            if isinstance(row["datatype"], str):
                assert row["datatype"] == new_datatype, (row["datatype"], new_datatype)
            else:
                df.at[i, "datatype"] = new_datatype
                if assigned < 10:
                    print("assigning", assigned, key, row["datatype"], new_datatype)
                assigned += 1
        processed += 1
    print("after", df["datatype"].value_counts(dropna=False))
    print(df["datatype"].isna().sum(), "na")
    print("assigned", assigned, "out of", len(df))
    # df = pd.merge(df, fixed_df, on="node_id")
    df.to_csv("abstract_dataflow.csv")

# %% Extract nodes from all graphs

def get_nodes(_id):
    itempath = svddc.BigVulDataset.itempath(_id)
    n, e = svdj.get_node_edges(itempath)
    return pd.DataFrame(list(zip(n["_label"], n["name"], n["id"])), columns=["_label", "name", "id"])

if __name__ == "__main__":
    node_df_file = Path("node_df.csv")
    if node_df_file.exists():
        node_df = pd.read_csv(node_df_file)
    else:
        print("Loading node df")
        with Pool(16) as pool:
            node_df = None
            ids = dataflow_df["graph_id"].unique()
            for i, d in enumerate(pool.imap_unordered(get_nodes, tqdm.tqdm(ids, desc="get aux node info", total=len(ids)))):
                if i < 3:
                    print(f"df {i}: {node_df}")
                if node_df is None:
                    node_df = d
                else:
                    node_df = pd.concat((node_df, d), ignore_index=True)
        print("writing to csv")
        node_df.to_csv(node_df_file)
    node_df["id"] = node_df["id"].astype(int)

#%% Get most common subkeys

if __name__ == "__main__":
    dataargs = {"sample": -1, "splits": "default", "load_code": False}
    train = svddc.BigVulDataset(partition="train", **dataargs)
    train_df = train.df
    print(train_df.columns)
    print(train_df)

    train_df = pd.merge(train_df, dataflow_df, left_on="id", right_on="graph_id")
    print(train_df.columns)
    datatype_vc = train_df["datatype"].value_counts()

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

def to_hash(row):
    items = []
    for key in select:
        items.append(select[key].index(row[key]) if row[key] in select[key] else pd.NA)

    # TODO: pad digits?

    # combine
    return " ".join(map(str,items))

# if __name__ == "__main__":
#     df["hash"] = df.apply(to_hash, axis=1)
#     print(df["hash"])
#     print(df.value_counts("hash"))
#     items_with_missing = sum(df["hash"].str.contains("-1"))

#     df.to_csv(f"abstract_dataflow_hash{'_sample' if sample else ''}.csv")

def generate_frequency_graphs():
    test = svddc.BigVulDataset(partition="test", **dataargs)
    test_df = test.df
    test_df = pd.merge(test_df, dataflow_df, left_on="id", right_on="graph_id")
    print("select from", datatype_vc)
    hhvc_true = {}
    for i, portion in enumerate(range(10, 101, 10)):
        portion = portion / 100
        number = int(len(datatype_vc) * portion)
        print("ITERATION", i, "portion", portion, "number", number, "/", len(datatype_vc))
        select = {
            "datatype": datatype_vc.nlargest(number).index.sort_values().tolist(),
        }
        for k in select:
            print(k, len(select[k]), "items selected")
            print("headdd", list(select[k])[:10])
        # print(test_df)
        test_df["hash"] = test_df.apply(to_hash, axis=1).replace("<NA>", pd.NA)
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
    plt.bar([str(int(k * 100)) + "% (" + str(int(len(datatype_vc) * k)) + ")" for k in hhvc_true.keys()], hhvc_true.values())
    plt.axhline(y=len(test_df), color='r', linestyle='-')
    plt.xlabel("portion of hashes from training dataset")
    plt.ylabel("number of hashed test examples")
    plt.xticks(rotation=45)
    plt.title("datatype only, keyed by raw type name")
    plt.tight_layout()
    plt.savefig("has_hash_test_portions.png")
    plt.close()
# %% 

if __name__ == "__main__":

    # Export dataset
    select = {
        "datatype": datatype_vc.nlargest(1000).index.sort_values().tolist(),
        # "datatype": datatype_vc.index.sort_values().tolist(),
    }
    all_df = None
    for split in ("train", "val", "test"):
        split_df = svddc.BigVulDataset(partition=split, **dataargs).df
        split_df = pd.merge(split_df, dataflow_df, left_on="id", right_on="graph_id")
        split_df["hash"] = split_df.apply(to_hash, axis=1).replace("<NA>", pd.NA)
        split_df = split_df[["graph_id", "node_id", "hash"]].sort_values(by=["graph_id", "node_id"]).reset_index(drop=True)
        split_df["node_id"] = split_df["node_id"].astype(int)
        print(split, len(split_df), split_df["hash"].value_counts(dropna=False))
        split_df.to_csv(f"abstract_dataflow_hash_all.csv")
        if all_df is None:
            all_df = split_df
        else:
            all_df = pd.concat((all_df, split_df), ignore_index=True)
    all_df.to_csv(f"abstract_dataflow_hash_all.csv")
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
