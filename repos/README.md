Extracting full repo and parsing datatypes with Joern

- download_all downloads repos
- archive_all writes commits to tarballs
- archive_extract extracts tarballs
- parse_all parses commits to cpg and extracts datatypes

Some repos you should expect the link to be broken.
See this ipython snapshot of me checking out my final outputs

```python

# import dataframe
In [4]: import pandas as pd
   ...: import sastvd.scripts.get_repos as gr
   ...: from pathlib import Path
   ...: df = pd.read_csv(f"bigvul_metadata_with_commit_id_unique.csv")
   ...: df["repo"] = df["repo"].map(gr.correct_repo_name)
   ...: df["clean_repo_name"] = df["repo"].map(gr.slug)
   ...: df["repo_filepath"] = df["clean_repo_name"].apply(lambda r: Path("repos/clean")/r)

In [6]: df["repo_filepath"].apply(lambda fp: fp.exists()).value_counts()
Out[6]: 
True     4043
False      10
Name: repo_filepath, dtype: int64

In [7]: df[~df["repo_filepath"].apply(lambda fp: fp.exists())]["repo_filepath"].unique()
Out[7]: 
array([PosixPath('repos/clean/http__git.infradead.org__mtd-2.6.git'),
       PosixPath('repos/clean/http__git.infradead.org__users__tgr__libnl.git'),
       PosixPath('repos/clean/https__git.haproxy.org__git__git__haproxy-1.5.git'),
       PosixPath('repos/clean/https__git.haproxy.org__git__git__haproxy-1.8.git'),
       PosixPath('repos/clean/https__git.haproxy.org__git__git__haproxy.git'),
       PosixPath('repos/clean/https__github.com__chrisd1100__uncurl'),
       PosixPath('repos/clean/https__github.com__davea42__libdwarf-code')],
      dtype=object)

In [28]: fdf = df[df["repo_filepath"].apply(lambda fp: fp.exists())]

In [34]: fdf["archive_filepath"] = df.apply(lambda r: Path("repos/archive")/(r["clean_repo_name"] + "__" + r["commit_id"] + ".tar"), axis=1)

In [33]: fdf["archive_filepath"].apply(lambda fp: fp.exists()).value_counts()
Out[33]: 
True    4043
Name: archive_filepath, dtype: int64

In [35]: cdf = fdf.copy()

In [36]: cdf["checkout_filepath"] = df.apply(lambda r: Path("repos/checkout")/(r["clean_repo_name"] + "__" + r["commit_i
    ...: d"]), axis=1)

In [4]: cdf["checkout_filepath"].apply(lambda p: p.exists()).value_counts()
Out[4]: 
True    4043
Name: checkout_filepath, dtype: int64

In [5]: cdf["checkout_filepath"].apply(lambda p: p.exists() and any(p.glob("*"))).value_counts()
Out[5]: 
True     3863
False     180
Name: checkout_filepath, dtype: int64

In [6]: cdf[~cdf["checkout_filepath"].apply(lambda p: p.exists() and any(p.glob("*")))]
Out[6]: 
      Unnamed: 0  ...                                  checkout_filepath
1              1  ...  repos/checkout/git__git.exim.org__exim.git__88...
17            17  ...  repos/checkout/git__git.infradead.org__users__...
170          170  ...  repos/checkout/http__git.ghostscript.com__mupd...
499          499  ...  repos/checkout/https__github.com__php__php-src...
912          912  ...  repos/checkout/https__github.com__FransUrbo__z...
...          ...  ...                                                ...
3881        3881  ...  repos/checkout/https__github.com__vadz__libtif...
3887        3887  ...  repos/checkout/https__github.com__vadz__libtif...
3901        3901  ...  repos/checkout/https__github.com__viabtc__viab...
3933        3933  ...  repos/checkout/https__github.com__zherczeg__je...
3948        3948  ...  repos/checkout/https__gitlab.freedesktop.org__...

[180 rows x 7 columns]
```
