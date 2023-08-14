
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
  name: str
  params: dict

@dataclass
class DataConfig:
  name: str
  path: str
  download: bool = True
  mimetype: Optional[str] = None
  loader: Optional[str] = None
  # TODO: require one of mimetype or loader

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
