from importlib import import_module
from uuid import uuid4
import random

experiment = Experiment()

@experiment.add_metric('accuracy')
def accuracy(epoch, offset):
  # return (preds == targets).mean()
  return 1 - 2 ** -epoch - random.random() / epoch - offset


@experiment.add_loss('loss')
def loss(epoch, offset):
  # return (preds == targets).mean()
  return 2 ** -epoch + random.random() / epoch + offset


def train(data_name, data_path, data_download=True):
  train_data, test_data = load_torchvision_data(data_name, data_path, data_download)
  print(train_data)
  print(data_stats(train_data))


if __name__ == '__main__':
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
