from source.data_pipeline.abstract_pipeline import AbstractTaskSpecificPipeline
import pandas as pd
import math 
import numpy as np

class ClassIdOversamplePipeline(AbstractTaskSpecificPipeline):
  """Child class of AbstractTaskSpecificPipeline. 
  Contains _calculate_row_oversample_ratio which is used in Mul and Seg Pipeline"""
  
  def __init__(self, df, images_dir_path, 
                    H, W, 
               IMG_MEAN = 0, IMG_STD = 1, 
               batch_size = 32, num_workers = 8, 
               oversampling = False,  oversampling_ratio = {},
               aug = [], 
               split = 'train'):
    super().__init__(df, images_dir_path, H, W, IMG_MEAN, IMG_STD, batch_size, num_workers,
                     oversampling, oversampling_ratio, aug, split)
  
  """Given a row with ImageId, ClassId in list form, calculate the oversample ratio of it.
  Multi-label dataset: ratio = mean ratio of classes"""
  def _calculate_row_oversample_ratio(self, row):
    # Calculate ratio
    if(len(row["ClassId"]) == 1): #single class
      row_oversample_ratio = self.oversampling_ratio[row["ClassId"][0]]

    else: # multi classes
      row_oversample_ratio = [] # append, then mean out
      for j in row["ClassId"]:
        # print(j)
        row_oversample_ratio.append(self.oversampling_ratio[j])
      row_oversample_ratio = math.floor(np.mean(row_oversample_ratio))

    return row_oversample_ratio