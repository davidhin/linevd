import pandas as pd
import sastvd as svd
import sastvd.linevd as lvd
from tqdm import tqdm

if __name__ == "__main__":

    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=1024,
        sample=-1,
        methodlevel=False,
        nsampling=False,
        gtype="pdg",
        splits="default",
    )

    train_func = []
    train_stmt = []
    val_func = []
    val_stmt = []
    test_func = []
    test_stmt = []

    for i in tqdm(range(len(data.train))):
        train_func.append(data.train[i].ndata["_FVULN"].max().item())
        train_stmt += data.train[i].ndata["_VULN"].tolist()

    for i in tqdm(range(len(data.val))):
        val_func.append(data.val[i].ndata["_FVULN"].max().item())
        val_stmt += data.val[i].ndata["_VULN"].tolist()

    for i in tqdm(range(len(data.test))):
        test_func.append(data.test[i].ndata["_FVULN"].max().item())
        test_stmt += data.test[i].ndata["_VULN"].tolist()

    def funcstmt_helper(funcs, stmts):
        """Count vuln and nonvulns."""
        ret = {}
        ret["vul_funcs"] = funcs.count(1)
        ret["nonvul_funcs"] = funcs.count(0)
        ret["vul_stmts"] = stmts.count(1)
        ret["nonvul_stmts"] = stmts.count(0)
        return ret

    stats = []
    stats.append({"partition": "train", **funcstmt_helper(train_func, train_stmt)})
    stats.append({"partition": "val", **funcstmt_helper(val_func, val_stmt)})
    stats.append({"partition": "test", **funcstmt_helper(test_func, test_stmt)})

    df = pd.DataFrame.from_records(stats)
    df["func_ratio"] = df.vul_funcs / (df.vul_funcs + df.nonvul_funcs)
    df["stmt_ratio"] = df.vul_stmts / (df.vul_stmts + df.nonvul_stmts)
    df.to_csv(svd.outputs_dir() / "bigvul_stats.csv", index=0)
