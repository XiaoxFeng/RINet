# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes, rotate_inds):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images + 2)

  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_blob1, im_blob2, im_scales = _get_image_blob(roidb, random_scale_inds, rotate_inds)

  blobs = {'data': im_blob,
           'data1': im_blob1,
           'data2': im_blob2}

  #assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != -1)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)
  blobs['image_level_labels'] = roidb[0]['image_level_labels']

  # add ss_boxes into blob
  #Changed for WSDNN

  roit = []
  rois = roidb[0]['boxes']
  if roidb[0]['rotate90'] | roidb[0]['rotate270']:
    width = roidb[0]['height']
    heigt = roidb[0]['width']
  else:
    width = roidb[0]['width']
    heigt = roidb[0]['height']
  for i in rotate_inds:
    if i == 1:
      # ------------------rotate90-------------------
      r90_rois = rois.copy()
      oldx1 = r90_rois[:, 0].copy()
      oldx2 = r90_rois[:, 2].copy()
      oldy1 = r90_rois[:, 1].copy()
      oldy2 = r90_rois[:, 3].copy()
      r90_rois[:, 0] = oldy1
      r90_rois[:, 1] = width - oldx2 - 1
      r90_rois[:, 2] = oldy2
      r90_rois[:, 3] = width - oldx1 - 1
      assert (r90_rois[:, 2] >= r90_rois[:, 0]).all()
      roit.append(r90_rois)

    elif i == 2:
      # ------------------rotate180-------------------
      r180_rois = rois.copy()
      oldx1 = r180_rois[:, 0].copy()
      oldx2 = r180_rois[:, 2].copy()
      oldy1 = r180_rois[:, 1].copy()
      oldy2 = r180_rois[:, 3].copy()
      r180_rois[:, 0] = width - oldx2 - 1
      r180_rois[:, 1] = heigt - oldy2 - 1
      r180_rois[:, 2] = width - oldx1 - 1
      r180_rois[:, 3] = heigt - oldy1 - 1
      assert (r180_rois[:, 2] >= r180_rois[:, 0]).all()
      assert (r180_rois[:, 3] >= r180_rois[:, 1]).all()
      roit.append(r180_rois)

    elif i == 3:
      # ------------------rotate270-------------------
      r270_rois = rois.copy()
      oldx1 = r270_rois[:, 0].copy()
      oldx2 = r270_rois[:, 2].copy()
      oldy1 = r270_rois[:, 1].copy()
      oldy2 = r270_rois[:, 3].copy()
      r270_rois[:, 0] = heigt - oldy2 - 1
      r270_rois[:, 1] = oldx1
      r270_rois[:, 2] = heigt - oldy1 - 1
      r270_rois[:, 3] = oldx2
      assert (r270_rois[:, 2] >= r270_rois[:, 0]).all()
      roit.append(r270_rois)
      # rot_rois = boxes
    else:
      roit.append(rois.copy())


  if True:
    #------------------0_boxes------------
    ss_inds = np.where(roidb[0]['gt_classes'] == -1)[0] # remove gt_rois in ss_boxes
    ss_boxes = np.empty((len(ss_inds), 5), dtype=np.float32)
    ss_boxes[:, 1:] = roidb[0]['boxes'][ss_inds,:] * im_scales[0]
    ss_boxes[:, 0] = 0
    blobs['ss_boxes'] = ss_boxes

    #------------------1_boxes------------
    ss_boxes_rotate_1 = np.empty((len(ss_inds), 5), dtype=np.float32)
    ss_boxes_rotate_1[:, 1:] = roit[0][ss_inds,:] * im_scales[0]
    ss_boxes_rotate_1[:, 0] = 0
    blobs['ss_boxes_rotate1'] = ss_boxes_rotate_1

    #------------------2_boxes------------
    ss_boxes_rotate_2 = np.empty((len(ss_inds), 5), dtype=np.float32)
    ss_boxes_rotate_2[:, 1:] = roit[1][ss_inds,:] * im_scales[0]
    ss_boxes_rotate_2[:, 0] = 0
    blobs['ss_boxes_rotate2'] = ss_boxes_rotate_2

  else:
    print('haha True')
    ss_boxes = np.empty((len(roidb[0]['boxes']), 5), dtype=np.float32)
    ss_boxes[:,1:] = roidb[0]['boxes'] * im_scales[0]
    ss_boxes[:,0]  = 0
    blobs['ss_boxes'] = ss_boxes

  return blobs

def _get_image_blob(roidb, scale_inds, rotate_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]

    if roidb[i]['rotate90']:
      im = im.transpose(1, 0, 2)
      im = im[::-1, :, :]
    if roidb[i]['rotate180']:
      im = im[::-1, ::-1, :]
    if roidb[i]['rotate270']:
      im = im.transpose(1, 0, 2)
      im = im[:, ::-1, :]

    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im_blob, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                         cfg.TRAIN.MAX_SIZE)
    processed_ims.append(im_blob)
    im_scales.append(im_scale)

    a = 1

    for j in rotate_inds:
      if j == 0:
        imt = im
      elif j == 1:
        imt = im.transpose(1, 0, 2)
        imt = imt[::-1, :, :]
      elif j == 2:
        imt = im[::-1, ::-1, :]
      else:
        imt = im.transpose(1, 0, 2)
        imt = imt[:, ::-1, :]


      target_size = cfg.TRAIN.SCALES[scale_inds[i]]
      im_blob, im_scale = prep_im_for_blob(imt, cfg.PIXEL_MEANS, target_size,
                      cfg.TRAIN.MAX_SIZE)
      a += 1

      processed_ims.append(im_blob)

      im_scales.append(im_scale)


  # Create a blob to hold the input images
  blob0 = im_list_to_blob([processed_ims[0]])
  blob1 = im_list_to_blob([processed_ims[1]])
  blob2 = im_list_to_blob([processed_ims[2]])

  return blob0, blob1, blob2, im_scales
