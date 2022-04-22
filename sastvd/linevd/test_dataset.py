import sastvd.linevd as lvd

def test_load_dataset():
    config = {
        "batch_size": 1,
        "gtype": "pdg+raw",
        "splits": "default",
    }

    # Load data
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=config["batch_size"],
        sample=5,
        methodlevel=False,
        nsampling=True,
        nsampling_hops=2,
        gtype=config["gtype"],
        splits=config["splits"],
    )

    # Print first graph
    print(data.train[0])