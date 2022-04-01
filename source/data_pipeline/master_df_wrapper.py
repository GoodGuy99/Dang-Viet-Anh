from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import math
import os

class MasterDataframeWrapper():

  """Read the dataframe from csv
  Split the dataframe by k-fold stratified cross-validaition
  Serve each stratified split over the public function get_stratified_split()
  Input:
  + train_csv_path: path to train.csv
  + train_images_dir_path: path to train_images folder
  + stratified_n_splits: number of splits in K-fold stratified cross validation
  + stratified_shuffle: whether to shuffle the dataset
  + stratified_column_name: column to split 
  + num_samples: the limit number of samples to take. Default: -1 mean takes everything

  Public procedure:
  + get_oversampling_ratio(): calculate the oversampling ratio of each class
  + get_stratified_split(epoch_counter): return tuple (train_df,val_df) according to counter
   """
  def __init__(self, train_csv_path, train_images_dir_path, 
               stratified_n_splits = 5, stratified_shuffle = True, stratified_column_name = 'ClassId', num_samples = -1):
    self.train_csv_path = train_csv_path
    self.train_images_dir_path = train_images_dir_path
    self.stratified_n_splits = stratified_n_splits
    self.stratified_shuffle = stratified_shuffle
    self.stratified_column_name = stratified_column_name
    self.num_samples = num_samples

    # Composition master_df inside the class for manipulation
    self.master_df = self._get_raw_master_dataframe()

    self.train_dfs, self.val_dfs = self._generate_stratified_cross_validation_splits()

  def get_stratified_split(self, epoch_counter):
    """ Public function: Return a (train_df, val_df) tuple from the epoch_counter.
    We calculate the modulo of the counter over stratified_n_splits, then get that from train_dfs, val_dfs list.
    Return tuple (train_df, val_df) """
    if (self.train_dfs is None) or (self.val_dfs is None):
      raise Exception("train_dfs, val_dfs not defined")
      return None 

    counter_mod_n_splits = epoch_counter % self.stratified_n_splits
    train_df = self.train_dfs[counter_mod_n_splits]
    val_df = self.val_dfs[counter_mod_n_splits]
    return train_df, val_df
    
  def get_oversampling_ratio(self):
    """Public function: Return a dict of oversampling ratio for each class.
    E.x. {0:1, 1:5, 2:50,...} 
    => Class 1 needs to oversample 5 times, class 2 50 times, class 0 no oversample,..."""
    class_count = self.master_df.ClassId.value_counts().to_dict()
    # print(class_count)

    max_count = max(class_count.values())
    over_sample_ratio = {}

    for id,count in class_count.items():
      over_sample_ratio[id] = math.floor(max_count/count)
    return over_sample_ratio
  
  def _get_raw_master_dataframe(self, IDCol = 'ImageId'):
    """Private function: get master dataframe by merging csv file and image names"""
    if(not self.train_csv_path) or (not self.train_images_dir_path):
      raise Exception("Csv or images dir not defined")
      return None 
    csv_labels = pd.read_csv(self.train_csv_path)
    train_img_names = os.listdir(self.train_images_dir_path)
    train_img_names = pd.DataFrame(train_img_names, columns =[IDCol])
    # print(train_img_names.head())

    master_df = pd.merge(train_img_names, csv_labels, how='outer', on=[IDCol])
    master_df = master_df.fillna(0)
    master_df.ClassId = master_df.ClassId.astype(int)
    if self.num_samples > 0:
      master_df = master_df.sample(self.num_samples, axis = 0)
    return master_df

  def _generate_stratified_cross_validation_splits(self):
    """Private function: return two lists train_dfs and val_dfs in a k-fold stratified crossvalidation"""
    if (self.master_df is None):
      raise Exception("master_df not defined")
      return None, None
    kfold = StratifiedKFold(n_splits=self.stratified_n_splits,
                            shuffle=self.stratified_shuffle)
    splits = kfold.split(self.master_df, self.master_df[self.stratified_column_name])

    train_dfs, val_dfs = [],[]
    print("K-fold stratified cross validation splits:")
    for train_indices, val_indices in splits:
      print("TRAIN:", train_indices, len(train_indices), "VAL:", val_indices, len(val_indices))
      train_dfs.append(self.master_df.iloc[train_indices])
      val_dfs.append(self.master_df.iloc[val_indices])

    return train_dfs, val_dfs