from source.data_pipeline.class_id_pipeline import ClassIdOversamplePipeline
from source.data.seg_dataset import SegmentationDataset, SegmentationDatasetv2
import pandas as pd
import numpy as np

class SegmentationPipeline(ClassIdOversamplePipeline):
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
    """Segmentation dataframe: 
    1. pivot classId into column, value is EncodedPixels
    2. convert column 0 into 'IsDefected'
    
    Output: Index column is ImageId; IsDefect colum; Four more columns for each class 1..4
    """
    seg_df = self.df
    # seg_df['HasMask'] = seg_df['EncodedPixels'].astype('bool')
    seg_df = seg_df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels').reset_index()
    seg_df.columns = ['ImageId', 'IsDefected', 'Mask_1','Mask_2','Mask_3','Mask_4']    
    seg_df['IsDefected'] = seg_df['IsDefected'].astype('bool')

    return seg_df


  """ 2. Over-sampling given the global oversampling ratio.
  First Add a new col, then repeat based on that row, then drop that column
  Reference: https://stackoverflow.com/questions/49074021/repeat-rows-in-data-frame-n-times"""
  def _over_sample(self):
    # Add non_NaN_col_name
    non_NaN_col_name = self.df[["Mask_1","Mask_2","Mask_3","Mask_4"]].stack().reset_index(level=1).groupby(level=0, sort=False)['level_1'].apply(list)
    # print(non_NaN_col_name.head(20))
    non_NaN_col_name.name = "non_NaN_col_name"
    self.df = self.df.join(non_NaN_col_name,how='left', sort=False)

    # Convert non_NaN_col_name to classID
    self.df["ClassId"] =self.df.apply(lambda row: self._convert_col_name_to_classId(row), axis=1)

    # Oversampling
    self.df["oversample_ratio"] = self.df.apply(
                              lambda row: self._calculate_row_oversample_ratio(row), 
                                           axis=1)
    self.df = self.df.loc[self.df.index.repeat(self.df.oversample_ratio)].reset_index(drop=True)
    self.df = self.df.drop(['oversample_ratio'], axis = 'columns')
    self.df = self.df.drop(['non_NaN_col_name'], axis = 'columns')
    return self.df

  """given a row, calculate the classId list"""
  def _convert_col_name_to_classId(self, row):
    if not isinstance(row["non_NaN_col_name"], list) : #Nan => not a list
      return [0]
    else:
      # print(row["non_NaN_col_name"])
      classId = []
      for col_name in row["non_NaN_col_name"]:
        col_name = col_name.replace("Mask_","")
        classId.append(int(col_name))
      return classId

  """ 3. Convert oversampled dataframe to Dataset"""
  def _get_dataset(self):
    dataset = SegmentationDataset(self.df, self.images_dir_path, self.H, self.W,
                           self.IMG_MEAN, self.IMG_STD, self.aug, self.split)
    return dataset
################################
class SegmentationPipelinev2(SegmentationPipeline):
  """Just replace SegmentationDataset by SegmentationDatasetv2"""

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
    dataset = SegmentationDatasetv2(self.df, self.images_dir_path, self.H, self.W,
                            self.IMG_MEAN, self.IMG_STD, self.aug, self.split)
    return dataset

