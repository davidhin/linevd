import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.datasets as svdd
import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


class BigVulDatasetNLP:
    """Override getitem for codebert."""

    def __init__(self, partition="train", random_labels=False):
        """Init."""
        self.df = svdd.bigvul()
        self.df = self.df[self.df.label == partition]
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        text = [tokenizer.sep_token + " " + ct for ct in self.df.before.tolist()]
        tokenized = tokenizer(text, **tk_args)
        self.labels = self.df.vul.tolist()
        if random_labels:
            self.labels = torch.randint(0, 2, (1, len(self.df)))
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Override getitem."""
        return self.ids[idx], self.att_mask[idx], self.labels[idx]


class BigVulDatasetNLPDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(self, batch_size: int = 32, sample: int = -1):
        """Init class from bigvul dataset."""
        super().__init__()
        self.train = BigVulDatasetNLP(partition="train")
        self.val = BigVulDatasetNLP(partition="val")
        self.test = BigVulDatasetNLP(partition="test")
        self.batch_size = batch_size

    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(
            self.train, shuffle=True, batch_size=self.batch_size, num_workers=6
        )

    def val_dataloader(self):
        """Return val dataloader."""
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=6)

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=6)


class LitCodebert(pl.LightningModule):
    """Codebert."""

    def __init__(self, lr: float = 1e-3):
        """Initilisation."""
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.fc1 = torch.nn.Linear(768, 256)
        self.fc2 = torch.nn.Linear(256, 2)
        self.accuracy = torchmetrics.Accuracy()
        self.auroc = torchmetrics.AUROC(compute_on_step=False)

    def forward(self, ids, mask):
        """Forward pass."""
        with torch.no_grad():
            bert_out = self.bert(ids, attention_mask=mask)
        fc1_out = self.fc1(bert_out["pooler_output"])
        fc2_out = self.fc2(fc1_out)
        out = F.softmax(fc2_out, dim=1)
        return out

    def training_step(self, batch, batch_idx):
        """Training step."""
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        loss = F.cross_entropy(logits, labels)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        loss = F.cross_entropy(logits, labels)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.auroc.update(logits[:, 1], labels)
        self.log("val_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        loss = F.cross_entropy(logits, labels)
        self.auroc.update(logits[:, 1], labels)
        self.log("test_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


run_id = svd.get_run_id()
savepath = svd.get_dir(svd.processed_dir() / "codebert" / run_id)
model = LitCodebert()
data = BigVulDatasetNLPDataModule(batch_size=128)
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
trainer = pl.Trainer(
    gpus=1,
    auto_lr_find=True,
    default_root_dir=savepath,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback],
)
tuned = trainer.tune(model, data)
trainer.fit(model, data)


# import sastvd.helpers.ml as ml
# from tqdm import tqdm
# run_id = "202108191652_2a65b8c_update_default_getitem_bigvul"
# chkpoint = (
#     svd.processed_dir()
#     / f"codebert/{run_id}/lightning_logs/version_0/checkpoints/epoch=188-step=18900.ckpt"
# )
# model = LitCodebert.load_from_checkpoint(chkpoint)
# model.cuda()
# all_pred = torch.empty((0, 2)).long().cuda()
# all_true = torch.empty((0)).long().cuda()
# for batch in tqdm(data.test_dataloader()):
#     ids, att_mask, labels = batch
#     ids = ids.cuda()
#     att_mask = att_mask.cuda()
#     labels = labels.cuda()
#     with torch.no_grad():
#         logits = model(ids, att_mask)
#     all_pred = torch.cat([all_pred, logits])
#     all_true = torch.cat([all_true, labels])
# ml.get_metrics_logits(all_true, all_pred)
