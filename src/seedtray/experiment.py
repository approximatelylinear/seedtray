from dataclasses import asdict
from uuid import uuid4
import functools
import random

import wandb

from seedtray.config_types import RunConfig
from seedtray.data_loader import load_data

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


class DataTransform:
  def __init__(self, name, func):
    self.name = name
    self.func = func


class ExperimentRun:
  def __init__(self, project, config, metrics, loss_funcs, data_transforms=None):
    if transforms is None:
      transforms = []
    self.project = project
    self.config = config
    self.metrics = metrics
    self.loss_funcs = loss_funcs
    self.data_transforms = data_transforms

  def load_data(self):
    return load_data(
        data_name=self.config.data.name,
        data_path=self.config.data.path,
        data_download=self.config.data.download,
        mimetype=self.config.data.mimetype,
        data_loader=self.config.data.loader,
        data_transforms=self.data_transforms
      )

  def run(self):
    data = self.load_data()
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
    self.data_transforms = []

  def configure(self, config):
    self.config = config
    wandb.init(
      # set the wandb project where this run will be logged
      project=self.config.project_id,

      # track hyperparameters and run metadata
      config=asdict(config),
    )

  def add_data_transform(self, name="transform"):
    def decorator(func):
      @functools.wraps(func)
      def wrapper(func):
        self.metrics.append(DataTransform(name, func))
      return wrapper
    return decorator

  def add_metric(self, name):
    def decorator(func):
      @functools.wraps(func)
      def wrapper(func):
        self.metrics.append(Metric(name, func))
      return wrapper
    return decorator

  def add_loss(self, name):
    def decorator(func):
      @functools.wraps(func)
      def wrapper(func):
        self.loss_funcs.append(Loss(name, func))
      return wrapper
    return decorator

  def run(self):
    run_config = RunConfig(self.config.model,
                           self.config.data,
                           self.config.training,
                           self.id)
    new_run = ExperimentRun(self.project, run_config, self.metrics, self.loss_funcs)
    new_run.run()
