import os, sys
from os.path import join as pjoin
from typing import Dict, Optional

import notifiers
from notifiers.logging import NotificationHandler
from jsonargparse import Namespace
import matplotlib.pyplot as plt
from loguru import logger
import optuna

from lightning.pytorch.tuner import Tuner
from lightning.pytorch.cli import LightningCLI

from tuner import OptunaMixin

def set_logger(verbose: bool) -> None:
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

class OptunaCLI(LightningCLI):
    def __init__(self, optuna_trial: Optional[optuna.trial.Trial] = None, *args, **kwargs) -> None:
        super().__init__(
            run=False, # no auto subcommands
            auto_configure_optimizers=False,
            *args, **kwargs)
        
        # For logging to slack
        if self.config["slack_webhook"]:
            logger.add(NotificationHandler("slack", defaults={"webhook_url": self.config["slack_webhook"], "message": "ERROR from analogyTP projects."}), level="ERROR")
        
        # Tensorboard
        self.trainer.logger.experiment.add_custom_scalars({
            "Loss": {'train&eval':['Multiline',["loss_epoch_train", "loss_epoch_eval"]]},
            self.model.monitor()[0]: {'val&test':['Multiline',[self.model.monitor()[0]+ '_eval']]},
        })
        
        self.optuna_trial = optuna_trial
        if optuna_trial:
            # Coupling with optuna
            if not isinstance(self.model, OptunaMixin):
                raise ValueError(f"The model must be a subclass of OptunaMixin, currently {type(self.model)}.")
            self.model.set_optuna_trial(optuna_trial)
            logger.info(f"Optuna suggests: {optuna_trial.params}")
        
        # torch.compile 
        #if isinstance( self.trainer.accelerator, pl.accelerators.CUDAAccelerator ):
        #    logger.info("Using CUDA. Perform torch.compile.")
            #self.model = torch.compile(self.model)
    
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--verbose",  type=bool, default=False)
        parser.add_argument("--resuming",  type=bool, default=False, help="Whether resume training from the last checkpoint")
        parser.add_argument("--tune_lr",  type=bool, default=False, help="Whether to use lr tuner to optimize hyperparameters")
        parser.add_argument("--tune_bz",  type=bool, default=False, help="Whether to use batchsize tuner to optimize hyperparameters")
        parser.add_argument("--do_fit",  type=bool, default=False)
        parser.add_argument("--do_test", type=bool, default=False)
        parser.add_argument("--ckpt_path", type=str, default=None, help="The path to the checkpoint file for validation or testing. When loading from checkpoint, hyperparameters in the checkpoint file will be used, no matter how CLI initialize `cli.model`.")
        parser.add_argument("--slack_webhook", type=str, default=None)
        #parser.link_arguments(source=("hparams_list", "optuna_trial"), target="hparams", compute_fn=compute_fn, apply_on="instantiate")
        #parser.set_defaults({"model.backbone": lazy_instance(MyModel, encoder_layers=24)})
    
    def before_instantiate_classes(self):
        def iter_helper(optuna_trial, config: Namespace, dic: Dict, prefix=""):
            for key, value in dic.items():
                if key == "HPARAMS":
                    for k, v in value.items():
                        if not isinstance(v, list): continue # no need to tune
                        logger.debug(f"!!! {prefix}HPARAMS.{k} -> {v}")
                        try:
                            assert len(v) == 2
                            l = float(v[0])
                            r = float(v[1])
                        except (ValueError, AssertionError, TypeError):
                            config.update(key=f"{prefix}HPARAMS.{k}", value=optuna_trial.suggest_categorical(k, v))
                        else:
                            config.update(key=f"{prefix}HPARAMS.{k}", value=optuna_trial.suggest_float(k, v[0], v[1]))
                elif isinstance(value, dict):
                    iter_helper(optuna_trial, config, value, prefix + key + ".")
                    
        set_logger(self.config["verbose"])
        logger.debug("before_instantiate_classes")
        if self.optuna_trial:
            iter_helper(self.optuna_trial, self.config, self.config.as_dict())
        
    def run(self) -> None:
        set_logger(self.config["verbose"])
        logger.info(f"Logging to {self.trainer.log_dir}.")
        tuner = Tuner(self.trainer)
        ret = None
        if self.config['do_fit']:
            if self.config['tune_lr']:
                lr_finder = tuner.lr_find(self.model, datamodule=self.datamodule, early_stop_threshold=None, max_lr=0.005)
                
                logger.debug(f"results: {lr_finder.results}")
                _ = lr_finder.plot(suggest=True)
                plt.savefig(pjoin(self.trainer.logger.log_dir, "lr_finder.png"))
                self.model.lr = lr_finder.suggestion()
                logger.info(f"Optimal learning rate: {self.model.lr}")
            
            if self.config['tune_bz']:
                tuner.scale_batch_size(self.model, datamodule=self.datamodule, mode="binsearch", method="fit")
                logger.info(f"Optimal batch size for train: {self.datamodule.hparams.batch_size}")
            
            ckpt_path = self.config["ckpt_path"] if self.config["resuming"] else None
            self.trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)
            if self.trainer.checkpoint_callback.best_model_path != '': logger.info(f"Up to now, the best model is saved at {self.trainer.checkpoint_callback.best_model_path}")
            ret = self.trainer.callback_metrics.get(self.model.monitor()[0]+ '_eval')
        
        ckpt_path = self.config["ckpt_path"]
        if ckpt_path is None and self.trainer.checkpoint_callback.best_model_path != '': 
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
        
        if self.config['do_test']:
            if self.config['tune_bz']:
                tuner.scale_batch_size(self.model, datamodule=self.datamodule, mode="binsearch", method="test")
                logger.info(f"Optimal batch size for etst: {self.datamodule.hparams.batch_size}")
            
            if ckpt_path: self.model = type(self.model).load_from_checkpoint(ckpt_path)
            self.trainer.test(self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)
            ret = self.trainer.callback_metrics.get(self.model.monitor()[0] + '_eval')
        
        if ret:
            if self.optuna_trial: 
                self.trainer.logger.log_hyperparams(
                    self.optuna_trial.params, 
                    metrics={self.model.monitor()[0] + '_eval': ret}
                )
            if self.config["slack_webhook"]:
                msg = f"Trial{self.optuna_trial.number if self.optuna_trial else ''}: {ret}"
                notifiers.get_notifier("slack").notify(message=msg, webhook_url=self.config["slack_webhook"])
            return ret
        else: 
            raise ValueError("No action to be done or return value is None. Please set do_fit or do_test if not yet.")

def cli_main(trial: optuna.trial.Trial = None):
    cli = OptunaCLI(optuna_trial=trial)
    return cli.run()

if __name__ == "__main__":
    logger.info(f"PID: {os.getpid()}")
    cli_main()
