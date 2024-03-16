import sys, os, functools
from os.path import join as pjoin
import argparse
from unittest.mock import patch
from typing import Tuple, List, Dict
from abc import ABC, abstractmethod

from loguru import logger
import torch.optim as optim
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback
    
class OptunaMixin(L.LightningModule, ABC):
    """
    Set self.optuna_trial if running with optuna.
    """
    @abstractmethod
    def monitor(self) -> Tuple[str, str]:
        """
        Specify the metric to monitor and the direction ("min" or "max") to optimize.
        Optuna pruner and early stopping callback will use f"val_{monitor()[0]}".
        """
        pass

    def set_optuna_trial(self, trial: optuna.trial.Trial) -> None: 
        self.optuna_trial = trial

    def configure_callbacks(self):
        call_backs = [EarlyStopping(monitor='val_' + self.monitor()[0], mode=self.monitor()[1], check_on_train_epoch_end=False, verbose=True)]
        if getattr(self, "optuna_trial", None):
            call_backs += [PyTorchLightningPruningCallback(self.optuna_trial, monitor='val_' + self.monitor()[0])]
        return call_backs

class NeuralMixin(OptunaMixin):
    """
    Deal with optimizers.
    When inherited, always put it as the last class in the inheritance list.
    """
    def __init__(self, HPARAMS: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = float(HPARAMS["lr"])
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.monitor()[1])
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.monitor()[0]}

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
        default="example",
        type=str,
    )
    parser.add_argument(
        "--direction",
        default="max",
        type=str,
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
    n_trials = args.n_trials
    pruner = optuna.pruners.MedianPruner()
    if not os.path.exists("../optuna_database"):
        os.makedirs("../optuna_database")
    storage = f"sqlite:///../optuna_database/{study_name}.db"
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