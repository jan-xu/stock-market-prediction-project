from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Union
from warnings import warn

from plotly.graph_objs import Figure

import wandb


class Logger(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def log_scalars(self, scalars: Dict[str, Any], step: int):
        pass

    @abstractmethod
    def log_plotly(self, figs: Dict[str, Figure], step: int):
        pass


class WandbLogger(Logger):

    def __init__(self):
        pass

    def log_scalars(self, scalars: Dict[str, Any], step: int):
        wandb.log(scalars, step=step)

    def log_plotly(self, figs: Dict[str, Figure], step: int):
        wandb.log(figs, step=step)


class LocalLogger(Logger):

    def __init__(self, run_folder: Union[str, PosixPath]):
        self.run_folder = Path(run_folder)
        self.longest_scalar_name = 20

    def log_scalars(self, scalars: Dict[str, Any], step: int):
        print(f"\nScalars at step {step}:")
        for name, scalar in scalars.items():
            print(f"  - {name:{self.longest_scalar_name}} : {scalar}")

    def log_plotly(self, figs: Dict[str, Figure], step: int):
        print(f"\nSaving figures at step {step}:")
        for name, fig in figs.items():
            print(f"  - {name}_{step}.html/png")
            fig.write_html(f"{self.run_folder}/figures/html/{name}_{step}.html")
            fig.write_image(f"{self.run_folder}/figures/png/{name}_{step}.png")


class LoggerHandler:

    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers
        self.accumulated_scalars = {}
        self.accumulated_figs = {}
        self.latest_step = 0

    def push_scalars(self, step: int):
        """
        Push the accumulated scalars to the loggers.
        """
        if step < self.latest_step:
            warn(f"Step {step} is less than the latest step {self.latest_step}!")

        if self.accumulated_scalars:
            for logger in self.loggers:
                logger.log_scalars(self.accumulated_scalars, step)
            self.accumulated_scalars = {}

        # Update latest step
        self.latest_step = step

        # Reset longest scalar name
        for logger in self.loggers:
            if isinstance(logger, LocalLogger):
                logger.longest_scalar_name = 20

    def push_plotly(self, step: int):
        """
        Push the accumulated plots to the loggers.
        """
        if step < self.latest_step:
            warn(f"Step {step} is less than the latest step {self.latest_step}!")

        if self.accumulated_figs:
            for logger in self.loggers:
                logger.log_plotly(self.accumulated_figs, step)
            self.accumulated_figs = {}

        # Update latest step
        self.latest_step = step

    def add_scalar(self, name: str, scalar: Any):
        """
        Add scalar to the accumulated scalars to be logged.
        """
        if name in self.accumulated_scalars:
            warn(f"Overwriting scalar value {scalar}!")
        self.accumulated_scalars[name] = scalar

        for logger in self.loggers:
            if isinstance(logger, LocalLogger):
                if len(name) > logger.longest_scalar_name:
                    logger.longest_scalar_name = len(name)

    def add_plot(self, name: str, fig: Figure):
        """
        Add plot to the accumulated plots to be logged.
        """
        if name in self.accumulated_figs:
            warn(f"Overwriting figure {name}!")
        self.accumulated_figs[name] = fig
