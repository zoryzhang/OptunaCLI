import sys, os, functools
from os.path import join as pjoin
import argparse
from unittest.mock import patch
from typing import Tuple, List, Dict, Any
from abc import ABC, abstractmethod

from loguru import logger
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from transformers import get_cosine_schedule_with_warmup
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy

def get_optimizers(
    parameters, trainer: L.Trainer, lr: float, warmup_steps: int
) -> Dict[str, Any]:
    """Return an AdamW optimizer with cosine warmup learning rate schedule."""
    strategy = trainer.strategy

    if isinstance(strategy, DeepSpeedStrategy):
        try:
            from deepspeed.ops.adam import FusedAdam
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except ImportError:
            logger.warning("Deepspeed is not installed. Remember to turn off deepspeed in the config file.")
    
        if "offload_optimizer" in strategy.config["zero_optimization"]:
            logger.info("Optimizing with DeepSpeedCPUAdam")
            optimizer = DeepSpeedCPUAdam(parameters, lr=lr, adamw_mode=True)
        else:
            logger.info("Optimizing with FusedAdam")
            optimizer = FusedAdam(parameters, lr=lr, adam_w_mode=True)
    else:
        logger.info("Optimizing with AdamW")
        optimizer = torch.optim.AdamW(parameters, lr=lr)

    if trainer.datamodule == None:
        return optimizer
        
    if trainer.max_steps != -1:
        max_steps = trainer.max_steps
    else:
        assert trainer.max_epochs is not None
        max_steps = (
            trainer.max_epochs
            * len(trainer.datamodule.train_dataloader())
            // trainer.accumulate_grad_batches
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }

class OptunaMixin(ABC):
    """
    Set self.optuna_trial if running with optuna.
    When inherited, always put it as the first few class in the inheritance list.
    """
    @abstractmethod
    def monitor(self) -> Tuple[str, str]:
        """
        Specify the metric to monitor and the direction ("min" or "max") to optimize.
        Optuna pruner and early stopping callback will use f"{self.monitor()[0]}_eval".
        """
        pass

    def set_optuna_trial(self, trial: optuna.trial.Trial) -> None: 
        self.optuna_trial = trial

    def configure_callbacks(self):
        call_backs = [EarlyStopping(
            monitor=f"{self.monitor()[0]}_eval", 
            mode=self.monitor()[1], 
            check_on_train_epoch_end=False, verbose=True)]
        call_backs += [
            ModelCheckpoint(
                dirpath=None, 
                filename=f'{{epoch}}-{{step}}-{{{ f"{self.monitor()[0]}_eval" }:.2f}}',
                monitor=f"{self.monitor()[0]}_eval",
                mode=self.monitor()[1], 
                save_last='link',
                verbose=True,
                auto_insert_metric_name=True,)
            ]
        call_backs += [LearningRateMonitor()]
        if getattr(self, "optuna_trial", None):
            call_backs += [PyTorchLightningPruningCallback(self.optuna_trial, monitor=f"{self.monitor()[0]}_eval")]
        return call_backs

class NeuralMixin:
    """
    Deal with optimizers.
    Will not pass argument HPARAMS to base class, therefore put it as the last customized base class but before library ones.
    """
    def __init__(self, HPARAMS: Dict, warmup_steps: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("HPARAMS", "warmup_steps")
        self.lr = float(HPARAMS["lr"])

    def configure_optimizers(self):
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.hparams.warmup_steps
        )

def upartial(f, *args, **kwargs):
    """
    An upgraded version of partial which accepts not named parameters
    """
    params = f.__code__.co_varnames[1:]
    kwargs = {**{param: arg for param, arg in zip(params, args)}, **kwargs}
    return functools.partial(f, **kwargs)

def print_save_optuna(study: optuna.study.Study, result_path: str = "optuna_database"):
    # print study statistics
    logger.debug("Study statistics: ")
    logger.debug(f"  Number of finished trials: {len(study.trials)}")
    logger.debug(f"  Number of pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))}")
    logger.debug(f"  Number of complete trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}")

    # save & show study results in human readable format
    df = study.trials_dataframe()
    if result_path and not os.path.exists(result_path):
        os.makedirs(result_path)
    df.to_csv( pjoin(result_path, f"{study.study_name}.csv"), index=False )
    logger.debug(df)

class StopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        print_save_optuna(study)

        # stop when consequtive pruned trials exceed threshold
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            logger.critical(f"Trials have been pruned {self._consequtive_pruned_count} times in a row. Stopping the study.")
            study.stop()

def optuna_main(objective: optuna.study.study.ObjectiveFuncType):
    logger.info(f"PID: {os.getpid()}")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study_name",
        type=str,
    )
    parser.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
    )
    parser.add_argument(
        "--n_trials",
        default=3,
        type=int,
        help="Number of trials to run",
    )
    args, unknown = parser.parse_known_args()
    #if unknown:
    #    raise ValueError(f"Unknown arguments: {unknown}")

    study_name = args.study_name
    direction = args.direction
    n_trials = args.n_trials if args.n_trials > 0 else None
    logger.info(f"Study name: {study_name}, direction: {direction}, n_trials: {n_trials}")
    pruner = optuna.pruners.MedianPruner()
    if not os.path.exists("./optuna_database"):
        os.makedirs("./optuna_database")
    storage = f"sqlite:///./optuna_database/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
    )
    with patch.object(sys, 'argv', [sys.argv[0]] + unknown):
        # remove known args from sys.argv to avoid conflict with LightningCLI
        study.optimize(
            objective,
            n_trials=n_trials,
            gc_after_trial=True, # garbage collection after each trial
            callbacks=[StopWhenTrialKeepBeingPrunedCallback(10)]
        )

if __name__ == "__main__":
    """
    To debug, add "JSONARGPARSE_DEBUG=true"

    Trainable Modules:
    python tuner.py --study_name=ana_tacgen --direction=min --n_trials=1 --config_file=zory.yaml --model=ana_tacgen --data=anaTacgenData
    python tuner.py --study_name=tacgenV2 --direction=min --n_trials=1 --config_file=zory.yaml --model=tacgenV2 --data=TacgenV2Data

    Architecture Choice:
    python tuner.py --study_name=tacgenBFS --direction=min --n_trials=1 --config_file=zory.yaml --model=BestFirstSearcher --model.init_args.tacgen=tacgenV1 --model.init_args.tacgen=tacgenV1
    python tuner.py --study_name=tacgenDCS --direction=min --n_trials=1 --config_file=tacgenDCS.yaml --model=DivideConquerSearcher
    python tuner.py --study_name=ana_tacgenBFS --direction=min --n_trials=1 --config_file=ana_tacgenBFS.yaml --model=DivideConquerSearcher
    python tuner.py --study_name=ana_tacgenDCS --direction=min --n_trials=1 --config_file=ana_tacgenDCS.yaml --model=DivideConquerSearcher
    python tuner.py --study_name=ana_tacgenV2BFS --direction=min --n_trials=1 --config_file=ana_tacgenV2BFS.yaml --model=DivideConquerSearcher
    python tuner.py --study_name=ana_tacgenV2DCS --direction=min --n_trials=1 --config_file=ana_tacgenV2DCS.yaml --model=DivideConquerSearcher
    python tuner.py --study_name=DORA_tacgenV2BFS --direction=min --n_trials=1 --config_file=DORA_tacgenV2BFS.yaml --model=DivideConquerSearcher
    python tuner.py --study_name=DORA_tacgenV2DCS --direction=min --n_trials=1 --config_file=DORA_tacgenV2DCS.yaml --model=DivideConquerSearcher
    python tuner.py --study_name=DORA_DCS --direction=min --n_trials=1 --config_file=DORA_DCS.yaml --model=DivideConquerSearcher
    """
    from CLI import cli_main
    optuna_main(cli_main)
