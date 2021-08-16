"""Implementation of IVDetect."""


import pickle as pkl
from importlib import reload

import dgl
import sastvd as svd
import sastvd.helpers.ml as ml
import sastvd.helpers.rank_eval as svdr
import sastvd.ivdetect.evaluate as ivde
import sastvd.ivdetect.gnnexplainer as ge
import sastvd.ivdetect.helpers as ivd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader

# Load data
train_ds = ivd.BigVulGraphDataset(partition="train")
val_ds = ivd.BigVulGraphDataset(partition="val", sample=1000)
test_ds = ivd.BigVulGraphDataset(partition="test")
dl_args = {"drop_last": False, "shuffle": True, "num_workers": 6}
train_dl = GraphDataLoader(train_ds, batch_size=16, **dl_args)
val_dl = GraphDataLoader(val_ds, batch_size=16, **dl_args)
test_dl = GraphDataLoader(test_ds, batch_size=64, **dl_args)

# %% Create model
reload(ivd)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ivd.IVDetect(input_size=200, hidden_size=100, num_layers=2, dropout=0.2)
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
ID = "202108121558_79d3273"
logger = ml.LogWriter(
    model, svd.processed_dir() / "ivdetect" / ID, max_patience=100, val_every=30
)
logger.load_logger()
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
        logger.save_logger()

    # Early Stopping
    if logger.stop():
        break
    logger.epoch()

# Print test results
logger.load_best_model()
model.eval()
all_pred = []
all_true = []
with torch.no_grad():
    test_mets_total = []
    for test_batch in test_dl:
        test_batch = test_batch.to(dev)
        test_labels = dgl.max_nodes(test_batch, "_VULN").long()
        test_logits = model(test_batch, test_ds)
        all_pred += test_logits[:, 1].tolist()
        all_true += test_labels.tolist()
        test_mets = ml.get_metrics_logits(test_labels, test_logits)
        test_mets_total.append(test_mets)
        logger.test(ml.dict_mean(test_mets_total))
logger.test(ml.dict_mean(test_mets_total))
rank_metr_test = ml.met_dict_to_str(svdr.rank_metr(all_pred, all_true))

# %% Statement-level through GNNExplainer
correct_lines = ivde.get_dep_add_lines_bigvul()
pred_lines = dict()
for batch in test_dl:
    for g in dgl.unbatch(batch):
        sampleid = g.ndata["_SAMPLE"].max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in pred_lines:
            continue
        try:
            lines = ge.gnnexplainer(model, g.to(dev), test_ds)
        except Exception as E:
            print(E)
        pred_lines[sampleid] = lines

with open(svd.cache_dir() / "pred_lines.pkl", "wb") as f:
    pkl.dump(pred_lines, f)

MFR = []
for sampleid, pred in pred_lines.items():
    true = correct_lines[sampleid]
    true = list(true["removed"]) + list(true["depadd"])
    for i, p in enumerate(pred):
        if p in true:
            MFR += [i]
            break
print(sum(MFR) / len(MFR))
