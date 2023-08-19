from importlib import import_module
from uuid import uuid4
import random

from seedtray.experiment import Experiment
from seedtray.config_types import (
  ModelConfig, DataConfig, OptimizerConfig, TrainingConfig, ExperimentConfig
)
from seedtray.data_loader import APPLICATION_TORCHVISION

experiment = Experiment()


@experiment.add_data_transform('normalize')
def normalize(data):
  from torchvision import transforms
  return transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.225, 0.225, 0.225]
    )
  ])

@experiment.add_metric('accuracy')
def accuracy(epoch, offset):
  return 1 - 2 ** -epoch - random.random() / epoch - offset


@experiment.add_loss('loss')
def loss(epoch, offset):
  return 2 ** -epoch + random.random() / epoch + offset


def main():
  experiment_name = 'Hello World'
  project_id = 'Hello World'
  model_config = ModelConfig('resnet18', {})
  data_config = DataConfig('CIFAR10', './data', True, mimetype=APPLICATION_TORCHVISION)
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

if __name__ == '__main__':
  main()
