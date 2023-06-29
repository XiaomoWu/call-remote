# ----------------------------------------------------------
# A self-contained (no need to import other scripts) example
# for debugging.
# ----------------------------------------------------------

import datatable as dt
import deepspeed
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb

from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap, checkpoint_wrapper
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class STSDataset(Dataset):
    def __init__(self, type: str):
        '''
        Args:
            type: "train" or "val"
        '''
        super().__init__()

        if type == 'train':
            self.data = dt.fread(
                '/home/yu/OneDrive/CC/local-dev/data/glue_data/SST-2/train.tsv')

        elif type == 'val':
            self.data = dt.fread(
                '/home/yu/OneDrive/CC/local-dev/data/glue_data/SST-2/dev.tsv')

        # change "label" from bool8 to int
        self.data = self.data[:, [dt.f.sentence,
                                  dt.as_type(dt.f.label, dt.Type.int8)]]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        '''For review subjectivity data
        return self.data.iloc[idx]
        '''
        x_text = self.data[idx, 'sentence']
        t = self.data[idx, 'label']

        return t, x_text


class STSDataModule(LightningDataModule):
    def __init__(self, bsz):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased')
        self.bsz = bsz

    def setup(self, stage=None):
        self.train_dataset = STSDataset(type='train')
        self.val_dataset = STSDataset(type='val')

    def collate_fn(self, batch):
        '''Using news sentiment regression data
        '''
        t, x_text = zip(*batch)
        t = torch.tensor(t, dtype=torch.half)

        input_tokens = self.tokenizer(
            list(x_text), max_length=512, padding=True, truncation=True, return_tensors='pt')

        return {'input_tokens': input_tokens, 't': t}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.bsz, num_workers=4, pin_memory=True, drop_last=False, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.bsz, num_workers=4, pin_memory=True, drop_last=False, collate_fn=self.collate_fn)


class TxtModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained('distilbert-base-uncased')

        '''
        self.encoder_ckpt = lambda input_ids, attention_mask: self.encoder(input_ids, attention_mask)['last_hidden_state']

        self.encoder_ckpt = checkpoint_wrapper(self.encoder_ckpt)
        '''

        self.fc_hidden = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 768),
            nn.ReLU())

        self.fc_output = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 1))


    def forward(self, input_tokens):
        emb = self.encoder(**input_tokens).last_hidden_state
        # emb = deepspeed.checkpointing.checkpoint(self.encoder_ckpt, x['input_ids'], x['attention_mask'])

        # get CLS as summary
        emb = emb[:, 0, :].squeeze()
        # emb = emb.mean(1)

        emb = self.fc_hidden(emb)
        logits = self.fc_output(emb).squeeze()
        return logits


class PlModule(LightningModule):
    def __init__(self, fc_lr, encoder_lr):
        super().__init__()
        self.automatic_optimization = False

        self.train_loss = nn.BCEWithLogitsLoss()
        self.val_acc = torchmetrics.Accuracy()

        self.model = TxtModel()

        self.fc_lr = fc_lr
        self.encoder_lr = encoder_lr

        # debug
        self.debug_params = None

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        t = batch['t']
        input_tokens = batch['input_tokens']

        logits = self.model(input_tokens)
        loss = self.train_loss(logits, t)

        # backward
        self.manual_backward(loss)
        self.log('train/loss', loss)
        
        # debug 
        # if self.global_rank == 0:
        #     debug_params = self.optimizers().state_dict()['model.fc_hidden.1.weight']

        #     if self.debug_params is None:
        #         self.debug_params = debug_params

        #     print(f'{torch.all(debug_params==self.debug_params)}')
        #     self.debug_params = debug_params


        if (batch_idx+1) % 4 == 0:
            print(f'step @{batch_idx=}')
            opt.step()
            self.zero_grad()


        # return loss


    def validation_step(self, batch, batch_idx):
        t = batch['t']

        input_tokens = batch['input_tokens']

        logits = self.model(input_tokens)
        y = torch.sigmoid(logits)

        self.val_acc.update(y, t.int())

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()
        if self.global_rank == 0:
            print(f'Val Accuracy: {acc*100}%')
        self.log('val/acc', acc)

    def configure_optimizers(self):
        fc_params = [p
                     for n, p in self.model.named_parameters()
                     if n.startswith('fc_')
                     ]
        encoder_params = [p
                          for n, p in self.model.named_parameters()
                          if n.startswith('encoder')
                          ]
        params = [{'params': fc_params, 'lr': self.fc_lr},
                  {'params': encoder_params, 'lr': self.encoder_lr}]

        optimizer = torch.optim.AdamW(params, lr=5e-5)
        # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(params, lr=5e-5)

        return optimizer


pl.seed_everything(42)

# init datamodule
bsz = 64
datamodule = STSDataModule(bsz)

# init model
fc_lr = 5e-5
encoder_lr = 5e-5
model = PlModule(fc_lr=fc_lr, encoder_lr=encoder_lr)

# init callbacks
ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    dirpath='checkpoints',
    monitor='val/acc',
    mode='max',
    save_last=False,
    save_top_k=1)

wandb_callback = WatchModel(log_freq=50)

# init ddp plugin
ddp_plugin = pl.plugins.DDPPlugin(find_unused_parameters=False)

deepspeed_plugin = pl.plugins.DeepSpeedPlugin(
    stage=2,
    offload_optimizer=False, offload_optimizer_device='cpu',
    offload_parameters=False, offload_params_device='cpu',
    cpu_checkpointing=False, partition_activations=False,
    logging_batch_size_per_gpu=1,
    logging_level=40)


# init wandb
logger = pl.loggers.WandbLogger(project='earnings-call', save_dir='',
                                log_model=False, save_code=False, name='debug-deepspeed', reinit=False)

# init trainer
trainer = pl.Trainer(accelerator='gpu',
                     devices=[0, 1],
                     strategy='ddp',
                     max_epochs=5, min_epochs=5,
                     precision=16,
                     callbacks=[wandb_callback, RichProgressBar(
                         leave=True), ckpt_callback],
                     logger=logger,
                     log_every_n_steps=10,
                     num_sanity_val_steps=0,
                     limit_train_batches=0.1,
                    #  accumulate_grad_batches=2
                     )

'''
# lr finder
lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
lr_finder.results
print(f'Suggested lr: {lr_finder.suggestion()}')
fig = lr_finder.plot(suggest=True)
fig.show()
'''

# start training
trainer.fit(model, datamodule=datamodule)
