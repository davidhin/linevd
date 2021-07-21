import sastvd.helpers.datasets as svdd
import sastvd.helpers.git as svdg
from tqdm import tqdm

tqdm.pandas()

df = svdd.bigvul()
df["allfunc"] = df.progress_apply(svdg.allfunc, comment="diff", axis=1)
df["allfunc_added"] = df.progress_apply(svdg.allfunc, comment="added", axis=1)
df["allfunc_removed"] = df.progress_apply(svdg.allfunc, comment="removed", axis=1)
df["allfunc_before"] = df.progress_apply(svdg.allfunc, comment="before", axis=1)
df["allfunc_after"] = df.progress_apply(svdg.allfunc, comment="after", axis=1)
