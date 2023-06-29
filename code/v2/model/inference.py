import datatable as dt
import dotenv
import hydra
import os
import torch

from src.datamodules.datasets import FrTxtDataset
from src.models.models import MtlModel
from torch.utils.data import DataLoader
from pyarrow import feather
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

# set working dir
work_dir = '/home/yu/OneDrive/CC/local-dev'
os.chdir(work_dir)

# init .env
dotenv.load_dotenv(override=True)


def load_checkpoint(zero_ckpt_path):

    output_ckpt_path = 'checkpoints/v2/ckpt_temp.pt'

    convert_zero_checkpoint_to_fp32_state_dict(
        zero_ckpt_path, output_ckpt_path)

    return torch.load(output_ckpt_path)


def inference(ckpt, split_id, sv_name):

    # ------------------
    # collect hparams
    # ------------------

    hparams = ckpt['hyper_parameters']
    datamodule_cfg = hparams['datamodule_cfg']
    model_cfg = hparams['model_cfg']

    # ------------------
    # init model
    # ------------------
    state_dict = ckpt['state_dict']

    model = MtlModel(**hparams)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.freeze()

    # ---------------
    # init datamodule
    # ---------------
    data_dir = '/home/yu/OneDrive/CC/local-dev/data/v2'
    datamodule = hydra.utils.instantiate(
        hparams['datamodule_cfg'],
        data_dir=data_dir,
        model_cfg=hparams['model_cfg'],
        _recursive_=False)

    # override with new split_id
    datamodule_cfg['split_id'] = split_id

    dataset = FrTxtDataset(
        **datamodule_cfg,
        d_model=768,
        split_type='test',
        data_dir=data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=datamodule.collate_fn,
        pin_memory=True,
        persistent_workers=True)

    # ------------------
    # inferencing!
    # ------------------
    trainer = Trainer(accelerator='gpu', devices=[0], precision=16)
    yt = trainer.predict(model=model, dataloaders=dataloader)

    # ------------------
    # save results
    # ------------------
    docid_idx, y, t = zip(*yt)
    docid_idx = torch.concat(docid_idx, dim=0).tolist()
    y = torch.concat(y, dim=0).tolist()  # (N, tasks)
    t = torch.concat(t, dim=0).tolist()  # (N, tasks)
    y = [tuple(row) for row in y]
    t = [tuple(row) for row in t]

    tasks = list(datamodule_cfg['tasks'])
    df_docid_idx = dt.Frame(docid_idx=docid_idx)
    df_y = dt.Frame(y, names=['y_'+t for t in tasks])
    df_t = dt.Frame(t, names=['t_'+t for t in tasks])
    df = dt.cbind([df_docid_idx, df_y, df_t]).to_arrow()

    output_path = f'/home/yu/OneDrive/CC/local-dev/data/v2/eval/pred_results/{sv_name}.feather'
    feather.write_feather(df, output_path)


# ---------
# load ckpt
# ---------
zero_ckpt_path = 'checkpoints/v2/epoch=7-step=4032.ckpt'
ckpt = load_checkpoint(zero_ckpt_path)

# ---------
# inference
# ---------
split_id = '08-18/19'
sv_name = 'yt_fixed_08-18@19'
inference(ckpt, split_id, sv_name)
