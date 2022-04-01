from abc import ABC, abstractmethod
from torch.utils import data
import pandas as pd
import numpy as np
import torch

class AbstractTaskSpecificPipeline(ABC):
  """After getting the (train,val) df split from MasterDataFrameWrapper, we need to perform this pipeline to convert it into usable dataset:
  1. Encoding: encode_bin_df,	encode_mul_df,		encode_seg_df **(task-specific)** => Strategry pattern
  2. If Oversampling: bin no oversample, oversample_mul,    oversample_seg **(task-specific)** => Strategry pattern
                     Multi-label row: oversample = mean(ratio)
  3. Convert to data.Dataset: BinDataset	MulDataset		SegDataset **(task-specific)** 
  4. Get data.Dataloader: get train, val dataloader *(common, but each task calls seperately)*
  5. Get data.Iterator: get train, val iter *(common, but each task calls seperately)*

  This class defines the interface for all three pipelines. (Template Design Pattern)
  Input: a dataframe (split) + parameter in __init__
  Output: an iterator
  """

  def __init__(self, df, images_dir_path, 
                   H, W, 
               IMG_MEAN = 0, IMG_STD = 1, 
               batch_size = 32, num_workers = 8, 
               oversampling = False,  oversampling_ratio = {},
               aug = [], 
               split = 'train'):
    """
    Input:
    + df: the dataframe from master dataframe wrapper
    + images_dir_path: path to training/testing image folder
    + H,W
    + IMG_MEAN, IMG_STD: for normalization
    + oversampling: whether to oversample to tackle class-imbalance
    + oversampling_ratio: calculated from master dataframe wrapper
    + split: 'train','val','test'
    + aug: one of [], ['light'], ['medium'] or ['light', 'medium']. To be passed into Dataset class
    """
    self.df = df                                # dataframe from MasterDataFrameWrapper
    self.images_dir_path = images_dir_path
    # Dimensions
    self.H = H
    self.W = W
    self.IMG_MEAN = IMG_MEAN
    self.IMG_STD = IMG_STD
    # Training
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.oversampling = oversampling              #  whether to oversample
    self.oversampling_ratio = oversampling_ratio  #  ratio calculated from MasterDataFrameWrapper
    self.split = split                            # 'train','val','test'
    if self.split != 'train':
      self.oversampling = False                   # Enable oversampling for training split only
    self.aug = aug                                # list of augmentations. Pass to Dataset to handle

    if self.split == 'test':
      raise Exception("test split not yet implemented")

    # Intermediatary attributes 
    self.dataset = None
    self.data_loader = None
    self.iterator = None

  """Interface: get the iterator by following five steps"""
  def get_iterator(self):
    self.df = self._encode_dataframe_from_master_split()
    if self.oversampling:
      self.df = self._over_sample()

    self.dataset = self._get_dataset()
    self.data_loader = self._get_dataloader()
    self.iterator = self._get_iterator()

    return self.iterator

  """ 1. Encode task-specific dataframe from the master dataframe split"""
  @abstractmethod
  def _encode_dataframe_from_master_split(self):
    pass

  """ 2. Over-sampling given the global oversampling ratio"""
  @abstractmethod
  def _over_sample(self):
    pass

  """ 3. Convert oversampled dataframe to Dataset"""
  @abstractmethod
  def _get_dataset(self):
    pass
  
  """ 4. Convert Dataset to Dataloader. 'val' split has double batch_size than 'train'"""
  def _get_dataloader(self):
    if self.dataset is None:
      raise Exception("self.dataset not defined")
      return None 

    if(self.split == 'train'):
      ds_loader = data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                  shuffle= True, num_workers=self.num_workers, pin_memory=True)
    elif(self.split == 'val'):
      ds_loader = data.DataLoader(self.dataset, batch_size=self.batch_size * 2, 
                                  shuffle= False, num_workers=self.num_workers, pin_memory=True)
    else: #TODO: test dataloader
      ds_loader = None
    return ds_loader

  """ 5. Convert Dataloader to Iterator"""
  def _get_iterator(self):
    if self.data_loader is None:
      raise Exception("self.data_loader not defined")
      return None 

    return iter(self.data_loader)