import pickle as pkl
from glob import glob
from importlib import reload

import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd

# %% TESTING
reload(lvd)

# gat2layer method-level
best_model_dir = (
    svd.processed_dir()
    / "methodlevel_-1/202109021250_4a3e680_add_initial_testing_code_for_method_level/lightning_logs/version_0/checkpoints"
)

# Set up modules
model = lvd.LitGNN()
data = lvd.BigVulDatasetLineVDDataModule(
    batch_size=32, nsampling=False, methodlevel=True, sample=-1
)
trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")

# Load weights and test
reload(lvd)
model = lvd.LitGNN()
best_model = glob(str(best_model_dir / "*"))[0]
model = lvd.LitGNN.load_from_checkpoint(best_model, strict=False)

trainer.test(model, data)


# %% Print results
print(model.res1vo)
print(model.res1)
print(model.res2)
print(model.res2mt)
print(model.res2f)
print(model.res3)
print(model.res3vo)
print(model.res4)
# model.plot_pr_curve()

with open("results.pkl", "wb") as f:
    pkl.dump(
        [
            model.res1vo,
            model.res1,
            model.res2,
            model.res2mt,
            model.res2f,
            model.res3,
            model.res3vo,
            model.res4,
        ],
        f,
    )

with open("results.pkl", "rb") as f:
    data = pkl.load(f)
