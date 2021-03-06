from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, gaussian_radius_wh, draw_umich_gaussian, draw_msra_gaussian, draw_truncate_gaussian
from utils.image import draw_dense_reg
from utils.image import load_image, augment_hsv, load_mosaic, letterbox, random_affine
from utils.utils import xyxy2xywh, xywh2xyxy
import random
import math

import albumentations as aug

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    # img_id = self.images[index]
    # file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    # img_path = os.path.join(self.img_dir, file_name)
    # ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    # anns = self.coco.loadAnns(ids=ann_ids)
    # num_objs = min(len(anns), self.max_objs)

    # img = cv2.imread(img_path)

    # height, width = img.shape[0], img.shape[1]

    # Load image
    if self.opt.mosaic:
      # Load mosaic
      img, labels = load_mosaic(self, index)
      shapes = None
    else:
      img, labels0, (h0, w0), (h, w) = load_image(self, index)

      # Letterbox
      img, ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=self.opt.large_scale)
      shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

      if labels0.size > 0:
        # Normalized xywh to pixel xyxy format
        labels = labels0.copy()
        labels[:, 1] = ratio[0] * w * (labels0[:, 1] - labels0[:, 3] / 2) + pad[0]  # pad width
        labels[:, 2] = ratio[1] * h * (labels0[:, 2] - labels0[:, 4] / 2) + pad[1]  # pad height
        labels[:, 3] = ratio[0] * w * (labels0[:, 1] + labels0[:, 3] / 2) + pad[0]
        labels[:, 4] = ratio[1] * h * (labels0[:, 2] + labels0[:, 4] / 2) + pad[1]
    
    if self.split == 'train':
      if not self.opt.mosaic:
        img, labels = random_affine(img, labels,
                                    degrees=self.opt.rotate,
                                    translate=self.opt.shift,
                                    scale=self.opt.scale,
                                    shear=self.opt.shear)
      if not self.opt.no_color_aug:
        augment_hsv(img, 0.014, 0.68, 0.36)
        img = yolov4_aug()(image=img)['image']

    num_objs = len(labels)  # number of labels
    if num_objs > 0:
      # convert xyxy to xywh
      labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

      # Normalize coordinates 0 - 1
      labels[:, [2, 4]] /= img.shape[0]  # height
      labels[:, [1, 3]] /= img.shape[1]  # width

    if self.split == 'train':
      # random left-right flip
      lr_flip = True
      if lr_flip and random.random() < 0.5:
        img = np.fliplr(img)
        if num_objs > 0:
          labels[:, 1] = 1 - labels[:, 1]

      # random up-down flip
      ud_flip = True
      if ud_flip and random.random() < 0.5:
        img = np.flipud(img)
        if num_objs > 0:
          labels[:, 2] = 1 - labels[:, 2]
        

    img = (img.astype(np.float32) / 255.)
    img = (img - self.mean) / self.std
    img = np.ascontiguousarray(img[:, :, ::-1]) # BGR to RGB
    img = img.transpose(2, 0, 1)

    output_h = img.shape[1] // self.opt.down_ratio
    output_w = img.shape[2] // self.opt.down_ratio
    num_classes = self.num_classes

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    if self.opt.heatmap_wh:
      draw_gaussian = draw_truncate_gaussian

    gt_det = []
    for k in range(min(num_objs, self.max_objs)):
      label = labels[k]
      bbox = label[1:]
      cls_id = int(label[0])
      bbox[[0, 2]] = bbox[[0, 2]] * output_w
      bbox[[1, 3]] = bbox[[1, 3]] * output_h
      bbox[0] = np.clip(bbox[0], 0, output_w - 1)
      bbox[1] = np.clip(bbox[1], 0, output_h - 1)
      h = bbox[3]
      w = bbox[2]

      if h > 0 and w > 0:
        ct = np.array(
          [bbox[0], bbox[1]], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if self.opt.heatmap_wh:
          h_radius, w_radius = gaussian_radius_wh((math.ceil(h), math.ceil(w)), 0.54)
          draw_gaussian(hm[cls_id], ct_int, h_radius, w_radius)
        else:
          radius = gaussian_radius((math.ceil(h), math.ceil(w)))
          radius = max(0, int(radius))
          radius = self.opt.hm_gauss if self.opt.mse_loss else radius
          draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh and not self.opt.heatmap_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    
    ret = {'input': img, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'gt_det': gt_det}
      ret['meta'] = meta
    return ret


def yolov4_aug():
  return aug.Compose([
    aug.RandomBrightnessContrast(p=0.7),
    aug.OneOf([
      aug.GaussNoise(p=1.),
      aug.ISONoise(p=1.),
      aug.ImageCompression(quality_lower=70, quality_upper=100, p=0.7)
      ], p=.7)], p=1)