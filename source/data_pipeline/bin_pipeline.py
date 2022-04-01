from source.data.bin_dataset import BinaryDataset, BinaryDatasetv2
from source.data_pipeline.abstract_pipeline import AbstractTaskSpecificPipeline
import pandas as pd

class BinaryPipeline(AbstractTaskSpecificPipeline):
  """Child class of AbstractTaskSpecificPipeline. Contains implementation of step 1->3"""
  
  def __init__(self, df, images_dir_path, 
                    H, W, 
               IMG_MEAN = 0, IMG_STD = 1, 
               batch_size = 32, num_workers = 8, 
               oversampling = False,  oversampling_ratio = {},
               aug = [], 
               split = 'train'):
    super().__init__(df, images_dir_path, H, W, IMG_MEAN, IMG_STD, batch_size, num_workers,
                     oversampling, oversampling_ratio, aug, split)
    
  """ 1. Encode task-specific dataframe from the master dataframe split"""
  def _encode_dataframe_from_master_split(self):
    bin_df = self.df.drop(['EncodedPixels'], axis = 'columns')
    bin_df = bin_df.replace([1, 2, 3, 4], 1)
    bin_df = bin_df.drop_duplicates(subset="ImageId") #Remove dup images that has 2 types of defects
    return bin_df

  """ 2. Over-sampling given the global oversampling ratio"""
  def _over_sample(self):
    # binary dataset: no oversampling because the number of defect vs. non-defect is equal
    return self.df

  """ 3. Convert oversampled dataframe to Dataset"""
  def _get_dataset(self):
    dataset = BinaryDataset(self.df, self.images_dir_path, self.H, self.W,
                           self.IMG_MEAN, self.IMG_STD, self.aug, self.split)
    return dataset

#########################
class BinaryPipelinev2(BinaryPipeline):
  """Just replace BinaryDataset by BinaryDatasetv2"""
  
  def __init__(self, df, images_dir_path, 
                    H, W, 
               IMG_MEAN = 0, IMG_STD = 1, 
               batch_size = 32, num_workers = 8, 
               oversampling = False,  oversampling_ratio = {},
               aug = [], 
               split = 'train'):
    super().__init__(df, images_dir_path, H, W, IMG_MEAN, IMG_STD, batch_size, num_workers,
                     oversampling, oversampling_ratio, aug, split)

  """ 3. Convert oversampled dataframe to Dataset"""
  def _get_dataset(self):
    dataset = BinaryDatasetv2(self.df, self.images_dir_path, self.H, self.W,
                           self.IMG_MEAN, self.IMG_STD, self.aug, self.split)
    return dataset