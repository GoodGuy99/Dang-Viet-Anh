from source.data_pipeline.bin_pipeline import BinaryPipeline, BinaryPipelinev2
from source.data_pipeline.mul_pipeline import MultiLabelPipeline, MultiLabelPipelinev2
from source.data_pipeline.seg_pipeline import SegmentationPipeline, SegmentationPipelinev2

class IteratorFactory():
  """Get the (train,val) iterator tuple for each epoch.
  Role in the data pipeline: From the master_dataframe_wrapper, call specific PipeLine class to get the final iterator for use
  Design Pattern: Factory Design Pattern
  Public interface:

  Version 1:
  + get_binary_iter(epoch_counter)
  + get_multilabel_iter(epoch_counter)
  + get_segmentation_iter(epoch_counter)

  Version 2:
  + get_binary_iter_v2(epoch_counter)
  + get_multilabel_iter_v2(epoch_counter)
  + get_segmentation_iter_v2(epoch_counter)
  """

  def __init__(self, 
               master_dataframe_wrapper, train_images_dir_path,
               H, W, 
               IMG_MEAN = 0, IMG_STD = 1, 
               batch_size_bin = 32, batch_size_mul = 32, batch_size_seg = 8,
               num_workers = 8,
               aug = [], oversampling = False):
    """Input:
    + master_dataframe_wrapper: one instance of MasterDataframeWrapper
    + train_images_dir_path
    + H,W, IMG_MEAN, IMG_STD: dimensions & normalization
    + batch_size, num_workers
    + oversampling
    + aug: augmentation
    """
    self.master_dataframe_wrapper = master_dataframe_wrapper
    self.train_images_dir_path = train_images_dir_path

    self.H = H
    self.W = W
    self.IMG_MEAN = IMG_MEAN
    self.IMG_STD = IMG_STD

    self.aug = aug  
    self.oversampling = oversampling

    self.batch_size_bin = batch_size_bin
    self.batch_size_mul = batch_size_mul
    self.batch_size_seg = batch_size_seg

    self.num_workers = num_workers
  

  def _get_iter_for_epoch(self, PipelineClass, batch_size, epoch_counter = 0):
    """Private function: general logic for get_iter interfaces function. Default: return iter for the first epoch
    Input: 
    + Discrete PipelineCLass
    + epoch_counter: default = 0 """
    train_split, val_split = self.master_dataframe_wrapper.get_stratified_split(epoch_counter)
    oversampling_ratio = self.master_dataframe_wrapper.get_oversampling_ratio()

    # Train
    train_pipeline = PipelineClass(train_split, self.train_images_dir_path, 
                      self.H, self.W, 
                      self.IMG_MEAN, self.IMG_STD, 
                      batch_size, self.num_workers, 
                      self.oversampling, oversampling_ratio,
                      self.aug, 
                      split = 'train')
    train_iter = train_pipeline.get_iterator()

    # Val
    val_pipeline = PipelineClass(val_split, self.train_images_dir_path, 
                     self.H, self.W, 
                    self.IMG_MEAN, self.IMG_STD, 
                    batch_size, self.num_workers, 
                    self.oversampling, oversampling_ratio,
                    self.aug, 
                    split = 'val')
    val_iter = val_pipeline.get_iterator()

    return train_iter, val_iter
  
  # Public interface procedures:

  ###### Version 1 Pipelines #####
  def get_binary_iter(self, epoch_counter = 0):
    return self._get_iter_for_epoch(BinaryPipeline, self.batch_size_bin, epoch_counter)

  def get_multilabel_iter(self, epoch_counter = 0):
    return self._get_iter_for_epoch(MultiLabelPipeline, self.batch_size_mul, epoch_counter)

  def get_segmentation_iter(self, epoch_counter = 0):
    return self._get_iter_for_epoch(SegmentationPipeline, self.batch_size_seg, epoch_counter)
  
  ###### Version 2 Pipelines #####
  def get_binary_iter_v2(self, epoch_counter = 0):
    return self._get_iter_for_epoch(BinaryPipelinev2, self.batch_size_bin, epoch_counter)

  def get_multilabel_iter_v2(self, epoch_counter = 0):
    return self._get_iter_for_epoch(MultiLabelPipelinev2, self.batch_size_mul, epoch_counter)

  def get_segmentation_iter_v2(self, epoch_counter = 0):
    return self._get_iter_for_epoch(SegmentationPipelinev2, self.batch_size_seg, epoch_counter)

