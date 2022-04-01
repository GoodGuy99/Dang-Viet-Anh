from source.data_pipeline.class_id_pipeline import ClassIdOversamplePipeline
from source.data.mul_dataset import MultiLabelDataset, MultiLabelDatasetv2
import pandas as pd
import numpy as np

class MultiLabelPipeline(ClassIdOversamplePipeline):
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
    mul_df = self.df.drop(['EncodedPixels'], axis = 'columns')
    mul_df = mul_df.groupby('ImageId', as_index= False).agg(lambda x: x.tolist())
    return mul_df

  """ 2. Over-sampling given the global oversampling ratio.
  First Add a new col, then repeat based on that row, then drop that column
  Reference: https://stackoverflow.com/questions/49074021/repeat-rows-in-data-frame-n-times"""
  def _over_sample(self):
    self.df["oversample_ratio"] = self.df.apply(
                              lambda row: self._calculate_row_oversample_ratio(row), 
                                           axis=1)
    self.df = self.df.loc[self.df.index.repeat(self.df.oversample_ratio)].reset_index(drop=True)
    self.df = self.df.drop(['oversample_ratio'], axis = 'columns')
    return self.df

  """ 3. Convert oversampled dataframe to Dataset"""
  def _get_dataset(self):
    dataset = MultiLabelDataset(self.df, self.images_dir_path, self.H, self.W,
                           self.IMG_MEAN, self.IMG_STD, self.aug, self.split)
    return dataset
#########################
class MultiLabelPipelinev2(MultiLabelPipeline):
  """Just replace MultiLabelDataset by MultiLabelDatasetv2"""
  
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
    dataset = MultiLabelDatasetv2(self.df, self.images_dir_path, self.H, self.W,
                            self.IMG_MEAN, self.IMG_STD, self.aug, self.split)
    return dataset