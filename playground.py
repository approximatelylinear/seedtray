from importlib import import_module
from dataclasses import dataclass, asdict
from typing import Optional, List
from uuid import uuid4
import functools

import random

import wandb


@dataclass
class ModelConfig:
  name: str
  params: dict

@dataclass
class DataConfig:
  name: str
  path: str
  download: bool = True

@dataclass
class OptimizerConfig:
  name: str
  params: dict

@dataclass
class TrainingConfig:
  epochs: int
  batch_size: int
  loss_fn: str
  optimizer: OptimizerConfig
  epochs: int

@dataclass
class RunConfig:
  model: ModelConfig
  data: DataConfig
  training: TrainingConfig
  experiment_id: Optional[str]

@dataclass
class ExperimentConfig:
  name: str
  model: ModelConfig
  data: DataConfig
  training: TrainingConfig
  project_id: Optional[str] = None

@dataclass
class ProjectConfig:
  name: str
  path: str
  experiments: Optional[List] = None


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


experiment = Experiment()

@experiment.add_metric('accuracy')
def accuracy(epoch, offset):
  # return (preds == targets).mean()
  return 1 - 2 ** -epoch - random.random() / epoch - offset


@experiment.add_loss('loss')
def loss(epoch, offset):
  # return (preds == targets).mean()
  return 2 ** -epoch + random.random() / epoch + offset


def load_torchvision_data(data_name, data_path, data_download=True):
  print('importing module torchvision.datasets.{}'.format(data_name))
  data_module = import_module('torchvision.datasets')
  data_cls = getattr(data_module, data_name)
  train_data = data_cls(root=data_path, train=True, download=data_download)
  test_data = data_cls(root=data_path, train=False, download=data_download)
  return train_data, test_data

def data_stats(data):
  return {
    'len': len(data),
    'shape': data.data.shape,
    'type': type(data),
    'classes': data.classes,
    'targets': data.targets,
    'class_to_idx': data.class_to_idx,
  }

def train(data_name, data_path, data_download=True):
  train_data, test_data = load_torchvision_data(data_name, data_path, data_download)
  print(train_data)
  print(data_stats(train_data))


if __name__ == '__main__':
  # train('CIFAR10', './data', True)
  experiment_name = 'Hello World'
  project_id = 'Hello World'
  model_config = ModelConfig('resnet18', {})
  data_config = DataConfig('CIFAR10', './data', True)
  optimizer_config = OptimizerConfig('Adam', {})
  training_config = TrainingConfig(
    epochs=10,
    batch_size=32,
    loss_fn='cross_entropy',
    optimizer=optimizer_config
  )
  experiment_config = ExperimentConfig(
    experiment_name, model_config, data_config, training_config, project_id)
  experiment.configure(experiment_config)
  experiment.run()
