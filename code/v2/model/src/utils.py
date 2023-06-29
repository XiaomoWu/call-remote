import logging
import pandas as pd
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import socket
import wandb

from omegaconf import ListConfig, DictConfig, OmegaConf
from pyarrow.feather import read_feather
from pytorch_lightning.utilities import rank_zero_only
from typing import Sequence
from .models import models


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
            'model_id',
            'seed',
            'datamodule',
            'model',
            'trainer',
            'callbacks',
            'logger'),
        resolve: bool = True) -> None:
    '''Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    '''

    style = 'yellow'
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def add_derived_cfg_before_init(cfg: DictConfig) -> None:
    '''Add derived cfgs BEFORE initializing model, trainer, etc.
    '''

    # enable adding new keys to config
    OmegaConf.set_struct(cfg, False)

    # add d_model (when using FrTxtModel)
    if 'glove' in cfg.preemb_dir:
        cfg.model.d_model = 300
    elif ('mpnet' in cfg.preemb_dir) | ('finbert' in cfg.preemb_dir):
        cfg.model.d_model = 768

    # add accumulate_grad_batches
    manual_step_every_n_batches = cfg.datamodule.eff_bsz/cfg.datamodule.bsz

    if manual_step_every_n_batches != int(manual_step_every_n_batches):
        raise Exception(f'eff_bsz must be divisible by bsz, but got {cfg.datamodule.eff_bsz=} and {cfg.datamodule.bsz=}')

    cfg.model.manual_step_every_n_batches = manual_step_every_n_batches

    # Add name to cfg.weighting_method
    weighting_method_name = list(cfg.weighting_method.keys())[0]
    cfg.weighting_method.name = weighting_method_name

    # add machine
    cfg.machine = socket.gethostname()

    # add datamodule.n_tasks
    cfg.datamodule.n_tasks = len(cfg.datamodule.tasks)

    # make sure cfg.trainer.devices is a list
    # if cfg.trainer.accelerator == 'gpu':
    #     assert isinstance(cfg.trainer.devices, ListConfig), \
    #         f'trainer/gpus must be a list, but get "{cfg.trainer.devices}" of type {type(cfg.trainer.devices)}!'

    # convert init_task_weights to numerical (if it's a str)
    if weighting_method_name in ['Fixed']:
        cfg.weighting_method[weighting_method_name].init_task_weights = [
            eval(w) if isinstance(w, str) 
            else w 
            for w in cfg.weighting_method[weighting_method_name].init_task_weights]
        init_task_weights = cfg.weighting_method[weighting_method_name].init_task_weights

        # sanity check target_weights
        assert len(init_task_weights) == len(cfg.datamodule.tasks), \
            f'Require len(targets)==len(target_weights), but got {cfg.datamodule.tasks=} and {cfg.weighting_method[weighting_method_name].init_task_weights=}'

        assert round(sum(init_task_weights), 3) in [1, len(init_task_weights)], \
            f'Require sum(init_task_weights)==1 but got {init_task_weights=}'

    # disable adding new keys to config
    OmegaConf.set_struct(cfg, True)


def add_derived_cfg_after_init(cfg):
    '''Add derived cfg AFTER initializing model, trainer, etc.

    We can't do this before init model because name is not a
    valid argument for loss/optimizer/scheduler
    '''

    OmegaConf.set_struct(cfg, False)

    # add scheduler/name
    scheduler_cfg = cfg.get('scheduler')
    if scheduler_cfg:
        cfg.scheduler.name = scheduler_cfg['_target_'].split('.')[-1]

    # add optimizer/name
    cfg.optimizer.name = cfg.optimizer._target_.split('.')[-1]
    OmegaConf.set_struct(cfg, True)


def load_tx_df(tx_df_name, tasks, data_dir, **kwargs):
    '''
    Args:
        target: list of str. Variables in the list are treated as
            prediction targest and should NOT be NA.
    '''
    # Load from feather
    tx_df = read_feather(f'{data_dir}/tx/{tx_df_name}.feather')

    # Remove NA obs
    for task in tasks:
        tx_df = tx_df[~tx_df[task].isnull()]

    return tx_df


def load_split_df(split_id, split_df_name, data_dir, **kwargs):
    split_df = read_feather(f'{data_dir}/split/{split_df_name}.feather')
    return split_df.loc[split_df.split_id==split_id]


def load_text_df(text_df_name, type_tokens, data_dir, **kwargs):
    '''
    type_token:
        1: MD
        2: Q
        3: A
    '''
    text_df = read_feather(f'{data_dir}/{text_df_name}.feather', columns=['transcriptid', 'sentenceid', 'type_token'])
    text_df = text_df[text_df.type_token.isin(type_tokens)]
    text_df.sentenceid = text_df.sentenceid.astype(int)
    text_df.transcriptid = text_df.transcriptid.astype(int)

    return text_df


# def load_tid_from_to_pair(DATA_DIR):
#     '''load DataFrame tid_from_to_pair, convert it into a Dict

#     output: {tid_from:[tid_to1, tid_to2, ...]}

#     tid_cid_pair_name: str. e.g., "3qtr"
#     '''
#     pair = read_feather(f'{DATA_DIR}/tid_from_to_pair.feather')

#     tid_from = pair.transcriptid_from
#     tid_to = [tid.tolist() for tid in pair.transcriptid_to]

#     return dict(zip(tid_from, tid_to))


@rank_zero_only
def log_hyperparameters(cfg: DictConfig,
                        model: pl.LightningModule,
                        trainer: pl.Trainer) -> None:
    '''This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    '''
    from hydra.core.hydra_config import HydraConfig

    hparams = {}

    # save the command line that runs the experiment
    hparams['cmd'] = HydraConfig.get().job.override_dirname

    # choose hparams that are dict (config_group)
    hparams['datamodule'] = cfg['datamodule']
    hparams['model'] = cfg['model']
    hparams['optimizer'] = cfg['optimizer']
    hparams['trainer'] = cfg['trainer']
    hparams['weighting_method'] = cfg['weighting_method']

    if 'scheduler' in cfg:
        hparams['scheduler'] = cfg['scheduler']
    if 'callbacks' in cfg:
        hparams['callbacks'] = cfg['callbacks']
    if 'logger' in cfg:
        hparams['logger'] = cfg['logger']
    if 'sweep' in cfg:
        hparams['sweep'] = cfg['sweep']

    # choose hparams that are NOT dict
    for k, v in cfg.items():
        if not isinstance(v, dict):
            hparams[k] = v

    # save number of model parameters
    hparams['model/params_total'] = sum(p.numel() for p in model.parameters())
    hparams['model/params_trainable'] = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad)
    hparams['model/params_not_trainable'] = sum(
        p.numel() for p in model.parameters()
        if not p.requires_grad)

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = lambda params: None


def finalize(logger):
    # without this sweeps with wandb logger might crash!
    if isinstance(logger, pl.loggers.WandbLogger):
        wandb.finish()
