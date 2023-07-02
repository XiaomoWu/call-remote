import copy
import hydra
import logging
import lightning.pytorch as pl
import shutil

from .models.models import MtlModel
from lightning.pytorch.callbacks import RichProgressBar, ModelSummary
from omegaconf import DictConfig
from typing import List



@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def train(cfg: DictConfig):
    from src import utils
    log = utils.get_logger(__name__, level=cfg.custom_loglevel)

    # Add derived cfg (cfgs that are computed from existing cfgs)
    utils.add_derived_cfg_before_init(cfg)

    # --------------------------------------
    # Pretty print config using Rich library
    # --------------------------------------
    if cfg.get("print_config"):
        utils.print_config(cfg, resolve=True)
    log.warning(f"Initial task weights: {cfg.weighting_method[cfg.weighting_method['name']].get('init_task_weights')}")

    # --------------------------------------
    # Setting the experiment env:
    #   - set log level
    #   - set random seed
    # --------------------------------------

    # set log level
    logging.getLogger('pytorch_lightning').setLevel(cfg.pl_loglevel)
    logging.getLogger('torch').setLevel(cfg.torch_loglevel)

    # fix random seed
    pl.seed_everything(cfg.seed, workers=True)
    # pl.seed_everything(211, workers=True)

    # ---------------------------------------------
    # Init datamodules, models, and trainer
    #   - init datamodule
    #   - init model
    #   - init trainer
    #     - logger
    #     - callbacks (e.g., early_stop, model_ckpt)
    #     - plugins
    #     - trainer
    #
    # Send some parameters from config to all lightning loggers
    # ---------------------------------------------

    # --------- init datamodule ---------
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule, data_dir=cfg.data_dir, model_cfg=cfg.model, _recursive_=False)

    # --------- init model ---------
    # - you must use deepcopy otherwise function add_derived_cfg_after_init
    # will change the value of cfg
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = MtlModel(
        datamodule_cfg=copy.deepcopy(cfg.datamodule),
        model_cfg=copy.deepcopy(cfg.model),
        optimizer_cfg=copy.deepcopy(cfg.optimizer),
        scheduler_cfg=copy.deepcopy(cfg.get('scheduler')),
        weighting_method_cfg=copy.deepcopy(cfg.weighting_method),
        trainer_cfg=copy.deepcopy(cfg.trainer),
        work_dir=cfg.work_dir)
    
    # --------- init lightning loggers ---------
    log.info(f"Instantiating logger <{cfg.logger._target_}>")
    logger = hydra.utils.instantiate(cfg.logger)

    # --------- init lightning callbacks (e.g., early_stop, model_ckpt) ---------
    callbacks: List[pl.Callback] = []
    if "callbacks" in cfg:
        for cb_name, cb_cfg in cfg.callbacks.items():
            if "_target_" in cb_cfg:
                log.info(f"Instantiating callback <{cb_cfg._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_cfg))
    # callbacks.append(RichProgressBar())  # rich progress bar
    callbacks.append(ModelSummary(max_depth=1))

    # --------- init plugins ---------
    strategy = None
    if cfg.get('strategy'):
        strategy = hydra.utils.instantiate(cfg.strategy)

    # --------- init lightning trainer ---------
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    # debug only
    # cfg.trainer.min_epochs = 4
    # cfg.trainer.max_epochs = 4

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        strategy=strategy,
        _convert_='partial')
    
    # Add derived cfg (some config must be added after init models)
    utils.add_derived_cfg_after_init(cfg)

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(cfg=cfg, model=model, trainer=trainer)

    # ------------------------
    # train/test OR find lr
    # ------------------------
    if cfg.mode == 'train':
        # clear checkpoint dir
        shutil.rmtree(cfg.get('ckpt_dir'))

        # train model
        trainer.fit(model, datamodule)

        # test model
        if cfg.test_after_train:
            trainer.test(ckpt_path='best', datamodule=datamodule)
            test_metric = trainer.callback_metrics[cfg.test_metric]

        # print path to best checkpoint
        log.info(
            f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    elif cfg.mode == 'lr_finder':
        # Plot learning rate
        lr_finder = trainer.tuner.lr_find(model=model, datamodule=datamodule)
        fig = lr_finder.plot(suggest=True)
        fig.show()

    # finalize
    utils.finalize(logger)

    # return test_metric (if any)
    if cfg.test_after_train:
        return test_metric

    # delimiters
    print('='*40)
    print('='*40)
    print('\n')
