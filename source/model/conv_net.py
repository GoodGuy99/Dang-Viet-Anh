import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import os
# from torch.autograd import Variable
# import torch.optim as optim
# from PIL import Image

class ConvolutionalNetwork(nn.Module): 
    """ Parent class for all Convoltional Neural Network (classifiers and segmentators
    Includes: 
    + Training logic in step
    + Set optimizer and loss_calc
    NOT includes:
    - Network architecture
    - Forward function: need a key ['logits'] in all child class's forward pass output dict"""

    def __init__(self, H, W, name = "conv_net"):
      """Input:
      + H,W
      + name: name of the model to be appended into output file name and ckpt name. Ex: 'bin_class','seg_unet',..."""
      super().__init__()
      self.H = H 
      self.W = W
      self.name = name

      #Need to be inited before training
      self.loss_calc = None
      self.optimizer = None
        
    def step(self, images, labels):
      """ Given tensor images (B,C,H,W) and labels(B,C)/(B,H,W) 
      Return: the loss value and output of model"""
      if ((not self.loss_calc) or (not self.optimizer)):
        raise Exception("Error: BinClass hasn't set loss calculation function and optimizer yet")
        return None, None
      self.optimizer.zero_grad()
      # Forward pass
      out = self(images)
      pred = out['logits']

      # Loss
      loss = torch.tensor([0], dtype=torch.float).cuda()
      loss += self.loss_calc(pred, labels)

      # Backprop
      loss.backward()
      self.optimizer.step()

      return loss, out

    # Setters
    def set_optimizer(self, optimizer):
      self.optimizer = optimizer
      # self.optimizer = optim.Adam(self.parameters(),  lr=5e-4)
    
    def set_loss_calc(self, loss_calc):
      self.loss_calc = loss_calc
    
    # Ckpt save & load
    def save_checkpoint(self, iteration,
                        save_best = "", metric_name = None, current_metric = 0.0,
                         best_metric = 0.0, checkpoint_dir = ""):
      """ Say at least one checkpoint-{model_type}-iter{iteration}.pth
      Input:
      + iteration
      + save_best: either "highest","lowest" or "". Save the model with the higest/lowest metric value
      + metric_name: the metric name on which model is measures
      + current_metric: the metric value on which model is measures
      + best_metric: the best metric value this model has seen
      """

      checkpoint = {
          'iteration': iteration,
          'optimizer': self.optimizer.state_dict(),
          'model': self.state_dict()
      }

      if metric_name:
          checkpoint['metric_name'] = metric_name
          checkpoint['current_metric'] = current_metric    
          checkpoint['best_metric'] = best_metric   

          # #DEBUG
          # print("current_metric", current_metric)
          # print("best_metric", best_metric)
          if ( (save_best == 'highest') and (current_metric >= best_metric)) or ((save_best == 'lowest') and (current_metric <= best_metric)):
            filename = "best_model_{0:s}.pth".format(self.name)
            filename = os.path.join(checkpoint_dir, filename)
            torch.save(checkpoint, filename)
            print("Saving current best model: {0:s}".format(filename))

      #Save checkpoint-iter regardless of best or not
      filename = "ckpt-{0:s}-iter{1:d}.pth".format(self.name,iteration)
      filename = os.path.join(checkpoint_dir, filename)
      print(f'\nSaving a checkpoint: {filename} ...')
      torch.save(checkpoint, filename)
      print("Checkpoint saved!")
    
    def load_checkpoint(self, ckpt_path):
      """Load checkpoint from a path. Also load the current state of the optimizer.
      Return None if ckpt not found, or 
      return tuple(iteration, metric_name, current_metric, best_metric)"""

      print(f'Loading checkpoint : {ckpt_path}')

      try:
        checkpoint = torch.load(ckpt_path)
      except FileNotFoundError:
        print("Err: Checkpoint not found at {0:s}".format(ckpt_path))
        return None

      # Found:
      iteration = checkpoint['iteration'] + 1
      print('Starting at iteration: ' + str(iteration))

      # metric
      metric_name = checkpoint['metric_name']
      current_metric = checkpoint['current_metric']
      best_metric = checkpoint['best_metric']

      # model & optimizer
      self.load_state_dict(checkpoint['model'])
      if self.optimizer is not None:
        self.optimizer.load_state_dict(checkpoint['optimizer'])

      return iteration, metric_name, current_metric, best_metric

