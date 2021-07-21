import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.git as svdg
from tqdm import tqdm

tqdm.pandas()


def insert_bigvul_comments(diff: str):
    lines = []
    for l in diff.splitlines():
        if len(l) == 0:
            continue
        if l[0] == "-":
            lines.append("//flaw_line_below:")
            l = l[1:]
        if l[0] == "+":
            lines.append("//fix_flaw_line_below:")
            l = "//" + l[1:]
        lines.append(l)
    return "\n".join(lines)


def apply_bigvul_comments(row):
    return svdg.get_codediff(row.dataset, row.id)["diff"]


def fine_grain_diff(row, diff=False):
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
    df = svdd.bigvul()
    svdg.mp_code2diff(df)
    df["vfwf_orig"] = df.progress_apply(apply_bigvul_comments, axis=1)
    df["vfwf"] = df.vfwf_orig.progress_apply(insert_bigvul_comments)
    df["equality"] = df.progress_apply(
        lambda x: " ".join(x.vfwf.split()) == " ".join(x.vul_func_with_fix.split()),
        axis=1,
    )
    assert len(df[df.equality]) / len(df) > 0.6

    # df["diff_num"] = df.progress_apply(fine_grain_diff, axis=1)
    # row = [i for i in df.itertuples() if i.id == 188434][0]
    # fine_grain_diff(row, diff=True)
