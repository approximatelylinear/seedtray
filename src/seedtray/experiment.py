from dataclasses import asdict
from uuid import uuid4
import functools
import random

import wandb

from seedtray.config_types import RunConfig

class Metric:
  def __init__(self, name, func):
    self.name = name
    self.func = func

  def compute(self, *args, **kwargs):
    return self.func(*args, **kwargs)


class Loss:
  def __init__(self, name, func):
    self.name = name
    self.func = func

  def compute(self, *args, **kwargs):
    return self.func(*args, **kwargs)


class ExperimentRun:
  def __init__(self, project, config, metrics, loss_funcs):
    self.project = project
    self.config = config
    self.metrics = metrics
    self.loss_funcs = loss_funcs

  def run(self):
    epochs = self.config.training.epochs
    offset = random.random() / 5
    for epoch in range(2, epochs):
      log_payload = {}
      for metric in self.metrics:
        val = metric.compute(epoch, offset)
        log_payload[f'metric/{metric.name}'] = val
      for loss in self.loss_funcs:
        val = loss.compute(epoch, offset)
        log_payload[f'loss/{loss.name}'] = val

      # log metrics to wandb
      wandb.log(log_payload)


class Experiment:
  def __init__(self, project=None, config=None):
    self.project = project
    self.config = config
    self.id = uuid4().hex
    self.metrics = []
    self.loss_funcs = []
    self.runs = []

  def configure(self, config):
    self.config = config
    wandb.init(
      # set the wandb project where this run will be logged
      project=self.config.project_id,

      # track hyperparameters and run metadata
      config=asdict(config),
    )

  def add_metric(self, name):
    def decorator(func):
      @functools.wraps(func)
      def wrapper(func):
        metric = Metric(name, func)
        self.metrics.append(metric)
      return wrapper
    return decorator

  def add_loss(self, name):
    def decorator(func):
      @functools.wraps(func)
      def wrapper(func):
        metric = Metric(name, func)
        self.loss_funcs.append(metric)
      return wrapper
    return decorator

  def run(self):
    run_config = RunConfig(self.config.model,
                           self.config.data,
                           self.config.training,
                           self.id)
    new_run = ExperimentRun(self.project, run_config, self.metrics, self.loss_funcs)
    new_run.run()
