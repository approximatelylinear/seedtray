from importlib import import_module
from enum import Enum

APPLICATION_TORCHVISION = 'application/vnd.torchvision'
APPLICATION_HUGGINGFACE = 'application/vnd.huggingface'
IMAGE_HEIC = "image/heic"
IMAGE_HEIF = "image/heif"
IMAGE_JPEG = "image/jpeg"
IMAGE_PNG = "image/png"
IMAGE_WEBP = "image/webp"
TEXT_PLAIN = "text/plain"
TEXT_HTML = "text/html"

types = [
  APPLICATION_TORCHVISION,
  APPLICATION_HUGGINGFACE,
  IMAGE_HEIC,
  IMAGE_HEIF,
  IMAGE_JPEG,
  IMAGE_PNG,
  IMAGE_WEBP,
  TEXT_PLAIN,
  TEXT_HTML,
]

def load_data(data_name, data_path, data_download=True, mimetype=None, loader=None):
  if mimetype is None and loader is None:
    raise ValueError('Either mimetype or loader must be specified')
  if mimetype == APPLICATION_TORCHVISION or loader == 'torchvision':
    return TorchvisionLoader(data_name, data_path, data_download).load()
  else:
    raise ValueError('Unsupported mimetype or loader: {}'.format(mimetype))

class TorchvisionLoader:
  def __init__(self, data_name, data_path, data_download=True):
    self.data_name = data_name
    self.data_path = data_path
    self.data_download = data_download

  def load(self):
    print('importing module torchvision.datasets.{}'.format(self.data_name))
    data_module = import_module('torchvision.datasets')
    data_cls = getattr(data_module, self.data_name)
    train_data = data_cls(root=self.data_path, train=True, download=self.data_download)
    test_data = data_cls(root=self.data_path, train=False, download=self.data_download)
    print(self.stats(train_data))
    return train_data, test_data

  @classmethod
  def stats(cls, data):
    return {
      'len': len(data),
      'shape': data.data.shape,
      'type': type(data),
      'classes': data.classes,
      'targets': data.targets,
      'class_to_idx': data.class_to_idx,
    }

