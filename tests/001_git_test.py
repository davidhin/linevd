import pandas as pd
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.git as svdg
from tqdm import tqdm

tqdm.pandas()


def insert_bigvul_comments(diff: str):
    """Insert comment lines in place of + and - in git diff patch."""
    lines = []
    for li in diff.splitlines():
        if len(li) == 0:
            continue
        if li[0] == "-":
            lines.append("//flaw_line_below:")
            li = li[1:]
        if li[0] == "+":
            lines.append("//fix_flaw_line_below:")
            li = "//" + li[1:]
        lines.append(li)
    return "\n".join(lines)


def apply_bigvul_comments(row):
    """Apply get_codediff using pandas."""
    ret = svdg.get_codediff(row.dataset, row.id)
    return "" if len(ret) == 0 else ret["diff"]


def fine_grain_diff(row, diff=False):
    """Get diff."""
    if row.equality:
        return 0
    f1 = row.vfwf
    f2 = row.vul_func_with_fix
    f1 = "\n".join([i.strip() for i in f1.splitlines()])
    f2 = "\n".join([i.strip() for i in f2.splitlines()])
    cd = svdg.code2diff(f1, f2)
    added = cd["added"]
    removed = cd["removed"]
    if diff:
        with open(svd.cache_dir() / "difftest.c", "w") as f:
            f.write(cd["diff"])
        with open(svd.cache_dir() / "difftest1.c", "w") as f:
            f.write(f1)
        with open(svd.cache_dir() / "difftest2.c", "w") as f:
            f.write(f2)
    return len(added) + len(removed)


def test_bigvul_diff_similarity():
    """Test 1."""
    df = svdd.bigvul(minimal=False, sample=True)
    df_vul = df[df.vul == 1].copy()
    svdg.mp_code2diff(df_vul)
    df_vul["vfwf_orig"] = df_vul.progress_apply(apply_bigvul_comments, axis=1)
    df_vul["vfwf"] = df_vul.vfwf_orig.progress_apply(insert_bigvul_comments)
    df_vul["equality"] = df_vul.progress_apply(
        lambda x: " ".join(x.vfwf.split()) == " ".join(x.vul_func_with_fix.split()),
        axis=1,
    )
    assert len(df_vul[df_vul.equality]) / len(df_vul) >= 0.3


def test_bigvul_diff_similarity_2():
    """Test 2s."""
    df = svdd.bigvul(minimal=True, sample=True)
    df["len_1"] = df.before.apply(lambda x: len(x.splitlines()))
    df["len_2"] = df.after.apply(lambda x: len(x.splitlines()))
    assert len(df[df.len_1 != df.len_2]) == 0


def test_code2diff_cases():
    """Test codediffs."""
    df = svdd.bigvul(minimal=False, return_raw=True)
    df = df[df.vul == 1]
    dfd = df.set_index("id")[["func_before", "func_after"]].to_dict()

    codediff = svdg.code2diff(dfd["func_before"][177775], dfd["func_after"][177775])
    assert codediff["removed"] == [16]
    assert codediff["added"] == [17]

    codediff = svdg.code2diff(dfd["func_before"][180189], dfd["func_after"][180189])
    assert codediff["removed"] == [36]
    assert codediff["added"] == [24, 25, 26, 27, 28, 29, 37]
