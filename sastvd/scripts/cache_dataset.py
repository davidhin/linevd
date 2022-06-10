from sastvd.linevd.dataset import BigVulDatasetLineVD
import tqdm

if __name__ == "__main__":
    sample = False
    feat = "_ABS_DATAFLOW_datatype"
    ds = BigVulDatasetLineVD(feat=feat, partition="all", sample_mode=sample)
    print(ds)
    for i, d in enumerate(tqdm.tqdm(ds, desc="cache")):
        # print(i, d)
        pass
