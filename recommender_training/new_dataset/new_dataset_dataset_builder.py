"""new_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for new_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'embedding': tfds.features.Tensor(shape=(200,), dtype = np.float32),
            'label': tfds.features.ClassLabel(num_classes = 4),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('embedding', 'label'),  # Set to `None` to disable
        #homepage='https://dataset-homepage/'
        # Specify whether to distable shuffling on examples.
        disable_shuffling = False,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples('../all_models/data/train'),
        'test' : self._generate_examples('../all_models/data/test')
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(my_dataset): Yields (key, example) tuples from the dataset
    ds = tf.data.TextLineDataset(path + "/embedding.csv")
    print(ds)
    for i, line in enumerate(ds.skip(1)):  
         
      l = tf.strings.split(line,",")      
      l = l.numpy()
      l = list(map(lambda x : x.decode('utf-8') , l))      
      l[0] = int(l[0])
      l[1] = l[1].split(":")
      l[1] = [np.float32(x) for x in l[1]]      
      
      yield path + "/" + str(i), {
          'embedding': np.array(l[1]),
          'label': l[0],
      }

