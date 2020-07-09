from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    if opt.amp:
      from torch.cuda.amp import autocast, GradScaler
      self.autocast = autocast
      self.scaler = GradScaler()
      print('Using Mixed Precision Training...')
    self.optimizer = optimizer
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    self.lf = lambda x: (((1 + math.cos(x * math.pi / opt.num_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    self.scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf)

    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss)
    self.nbs = opt.nbs  # nominal batch size
    self.accumulate = max(round(self.nbs / opt.batch_size), 1)  # accumulate loss before optimizing

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    n_burn = max(3 * num_iters, 1e3)  # burn-in iterations, max(3 epochs, 1k iterations)
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()

    self.optimizer.zero_grad()
    for iter_id, batch in enumerate(data_loader):
      ni = iter_id + num_iters * (epoch - 1)  # number integrated batches (since train start)
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)

      if ni <= n_burn:
        xi = [0, n_burn]  # x interp
        accumulate = max(1, np.interp(ni, xi, [1, self.opt.nbs / self.opt.batch_size]).round())
        for j, x in enumerate(self.optimizer.param_groups):
          x['lr'] = np.interp(ni, xi, [0.0, x['initial_lr'] * self.lf(epoch - 1)])

      if opt.amp:
        with self.autocast():
          output, loss, loss_stats = model_with_loss(batch)
          loss = loss.mean()
        if phase == 'train':
          self.scaler.scale(loss).backward()
          if ni & self.accumulate == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
      else:
        output, loss, loss_stats = model_with_loss(batch)
        loss = loss.mean()
        if phase == 'train':
          loss.backward()
          if ni & self.accumulate == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} |LR: {lr:.7f}'.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td, lr=self.optimizer.param_groups[0]['lr'])
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats
    
    # Scheduler
    self.scheduler.step()

    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)