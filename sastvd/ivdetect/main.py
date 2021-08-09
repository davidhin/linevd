"""Implementation of IVDetect."""


from importlib import reload

import dgl
import sastvd as svd
import sastvd.helpers.ml as ml
import sastvd.ivdetect.helpers as ivd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from pandarallel import pandarallel
from tqdm import tqdm

tqdm.pandas()
pandarallel.initialize()


# Load data
train_ds = ivd.BigVulGraphDataset(partition="train")
val_ds = ivd.BigVulGraphDataset(partition="val", sample=500)
test_ds = ivd.BigVulGraphDataset(partition="test")
train_dl = GraphDataLoader(train_ds, batch_size=24, drop_last=False, shuffle=True)
val_dl = GraphDataLoader(val_ds, batch_size=64, drop_last=False, shuffle=True)
test_dl = GraphDataLoader(test_ds, batch_size=64, drop_last=False, shuffle=True)

# %% Create model
reload(ivd)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ivd.IVDetect(input_size=200, hidden_size=100, num_layers=2)
model.to(dev)

# Debugging a single sample
batch = next(iter(train_dl))
batch = batch.to(dev)
logits = model(batch, train_ds)

# %% Optimiser
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# Train loop
ID = svd.get_run_id({})
logger = ml.LogWriter(
    model, svd.processed_dir() / "ivdetect" / ID, max_patience=100, val_every=20
)
while True:
    for batch in train_dl:

        # Training
        model.train()
        batch = batch.to(dev)
        logits = model(batch, train_ds)
        labels = dgl.max_nodes(batch, "_VULN").long()
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        train_mets = ml.get_metrics_logits(labels, logits)
        val_mets = train_mets
        if logger.log_val():
            model.eval()
            with torch.no_grad():
                val_mets_total = []
                for val_batch in val_dl:
                    val_batch = val_batch.to(dev)
                    val_labels = dgl.max_nodes(val_batch, "_VULN").long()
                    val_logits = model(val_batch, val_ds)
                    val_mets = ml.get_metrics_logits(val_labels, val_logits)
                    val_mets_total.append(val_mets)
                val_mets = ml.dict_mean(val_mets_total)
        logger.log(train_mets, val_mets)

    # Early Stopping
    if logger.stop():
        break
    logger.epoch()

# Print test results
logger.load_best_model()
model.eval()
with torch.no_grad():
    test_mets_total = []
    for test_batch in test_dl:
        test_batch = test_batch.to(dev)
        test_labels = dgl.max_nodes(test_batch, "_VULN").long()
        test_logits = model(test_batch, test_ds)
        test_mets = ml.get_metrics_logits(test_labels, test_logits)
        test_mets_total.append(test_mets)
        logger.test(ml.dict_mean(test_mets_total))
logger.test(ml.dict_mean(test_mets_total))
