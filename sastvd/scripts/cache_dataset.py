import sastvd.linevd as lvd

if __name__ == "__main__":
    config = {
        "batch_size": 1,
        "gtype": "pdg+raw",
        "splits": "default",
    }

    # Load all data
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=config["batch_size"],
        sample=-1,
        methodlevel=False,
        nsampling=True,
        nsampling_hops=2,
        gtype=config["gtype"],
        splits=config["splits"],
    )

    # Print first graph
    print(data.train[0])