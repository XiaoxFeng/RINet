# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.timer

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes
from nets.OICR import rotation_invariant_graph_activate, smooth_l1_loss

#from layer_utils.roi_pooling.roi_pool import RoIPoolFunction
from layer_utils.roi_pooling.functions.roi_pool import RoIPoolFunction
from layer_utils.roi_align.crop_and_resize import CropAndResizeFunction
from layer_utils.roi_ring_pooling.functions.roi_ring_pool import RoIRingPoolFunction

from model.config import cfg
from utils.bbox import bbox_overlaps
import tensorboardX as tb

from scipy.misc import imresize

class Network(nn.Module):
  def __init__(self):
    nn.Module.__init__(self)
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = {}
    self._score_summaries = {}
    self._event_summaries = {}
    self._image_gt_summaries = {}
    self._variables_to_fix = {}
    self._device = 'cuda'

  def _add_gt_image(self):
    # add back mean
    image = self._image_gt_summaries['image'] + cfg.PIXEL_MEANS
    image = imresize(image[0], self._im_info[:2] / self._im_info[2])
    # BGR to RGB (opencv uses BGR)
    self._gt_image = image[np.newaxis, :,:,::-1].copy(order='C')

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    self._add_gt_image()
    image = draw_bounding_boxes(\
                      self._gt_image, self._image_gt_summaries['gt_boxes'], self._image_gt_summaries['im_info'])

    return tb.summary.image('GROUND_TRUTH', image[0].astype('float32').swapaxes(1,0).swapaxes(2,0)/255.0)

  def _add_act_summary(self, key, tensor):
    array = tensor.data.cpu().numpy()
    array = array + np.random.random(array.shape) / 10000
    return tb.summary.histogram('ACT/' + key + '/activations', array, bins='auto'),
    tb.summary.scalar('ACT/' + key + '/zero_fraction',
                      (tensor.data == 0).float().sum() / tensor.numel())

  def _add_score_summary(self, key, tensor):
    array = tensor.data.cpu().numpy()
    array = array + np.random.random(array.shape) / 10000                
    return tb.summary.histogram('SCORE/' + key + '/scores', array, bins='auto')

  def _add_train_summary(self, key, var):
    array = var.data.cpu().numpy()
    array = array + np.random.random(array.shape) / 10000
    return tb.summary.histogram('TRAIN/' + key, array, bins='auto')

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_top_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     self._feat_stride, self._anchors, self._num_anchors)
    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors)

    return rois, rpn_scores
  def _roi_ring_pool_layer(self, bottom, rois, scale_inner, scale_outer):
    return RoIRingPoolFunction(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1. / 16., scale_inner, scale_outer)(bottom, rois)

  def _roi_pool_layer(self, bottom, rois):
    return RoIPoolFunction(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1. / 16.)(bottom, rois)

  def _crop_pool_layer(self, bottom, rois, max_pool=True):
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()

    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    pre_pool_size = cfg.POOLING_SIZE * 2 if max_pool else cfg.POOLING_SIZE
    crops = CropAndResizeFunction(pre_pool_size, pre_pool_size)(bottom, 
      torch.cat([y1/(height-1),x1/(width-1),y2/(height-1),x2/(width-1)], 1), rois[:, 0].int())
    if max_pool:
      crops = F.max_pool2d(crops, 2, 2)
    return crops

  def _anchor_target_layer(self, rpn_cls_score):
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
      anchor_target_layer(
      rpn_cls_score.data, self._gt_boxes.data.cpu().numpy(), self._im_info, self._feat_stride, self._anchors.data.cpu().numpy(), self._num_anchors)

    rpn_labels = torch.from_numpy(rpn_labels).float().to(self._device) #.set_shape([1, 1, None, None])
    rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets).float().to(self._device)#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_inside_weights = torch.from_numpy(rpn_bbox_inside_weights).float().to(self._device)#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_outside_weights = torch.from_numpy(rpn_bbox_outside_weights).float().to(self._device)#.set_shape([1, None, None, self._num_anchors * 4])

    rpn_labels = rpn_labels.long()
    self._anchor_targets['rpn_labels'] = rpn_labels
    self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
    self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
    self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

    for k in self._anchor_targets.keys():
      self._score_summaries[k] = self._anchor_targets[k]

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores):
    print('rois ', rois.shape)
    print('roi_scores ', roi_scores.shape)
    rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
      proposal_target_layer(
      rois, roi_scores, self._gt_boxes, self._num_classes)

    self._proposal_targets['rois'] = rois
    self._proposal_targets['labels'] = labels.long()
    self._proposal_targets['bbox_targets'] = bbox_targets
    self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
    self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

    for k in self._proposal_targets.keys():
      self._score_summaries[k] = self._proposal_targets[k]

    return rois, roi_scores

  def _anchor_component(self, height, width):
    # just to get the shape right
    #height = int(math.ceil(self._im_info.data[0, 0] / self._feat_stride[0]))
    #width = int(math.ceil(self._im_info.data[0, 1] / self._feat_stride[0]))
    anchors, anchor_length = generate_anchors_pre(\
                                          height, width,
                                           self._feat_stride, self._anchor_scales, self._anchor_ratios)
    self._anchors = torch.from_numpy(anchors).to(self._device)
    self._anchor_length = anchor_length

  def _add_losses(self, sigma_rpn=3.0):

    self._losses['cross_entropy'] = np.zeros(1)
    self._losses['loss_box'] = np.zeros(1)
    self._losses['rpn_cross_entropy'] = np.zeros(1)
    self._losses['rpn_loss_box'] = np.zeros(1)
    
    det_cls_prob = self._predictions['det_cls_prob']
    det_cls_prob = det_cls_prob.view(-1)
    label = self._image_level_label.view(-1)
    
    det_cls_product = self._predictions['det_cls_prob_product_2']
    
    refine_prob_1 = self._predictions['refine_prob_1']
    refine_prob_2 = self._predictions['refine_prob_2']
    refine_prob_3 = self._predictions['refine_prob_3']
    rotate_prob = self._predictions['rotate']
    bbox_pred = self._predictions['bbox_pred']
    
    
    #---------------caculating the loss of the first branch
    roi_labels, roi_weights ,keep_inds = self.get_refine_supervision(det_cls_product, self._image_gt_summaries['ss_boxes'][self.ss_boxes_indexes, :],
                                                                     self._image_gt_summaries['image_level_label'])
    
    roi_weights = torch.tensor(roi_weights).cuda()
    roi_labels = torch.tensor(roi_labels, dtype=roi_weights.dtype).cuda()
    #roi_labels = torch.mul(roi_labels, roi_weights)   
    refine_loss_1 = - torch.sum(torch.mul(roi_labels, torch.log(refine_prob_1[keep_inds]))) / roi_labels.shape[0]
    
    #---------------caculating the loss of the second branch
    roi_labels, roi_weights, keep_inds = self.get_refine_supervision(refine_prob_1, self._image_gt_summaries['ss_boxes'][self.ss_boxes_indexes, :],
                                                                     self._image_gt_summaries['image_level_label'])
    
    roi_weights = torch.tensor(roi_weights).cuda()
    roi_labels = torch.tensor(roi_labels, dtype=roi_weights.dtype).cuda()
    #roi_labels = torch.mul(roi_labels, roi_weights)
    refine_loss_2 = - torch.sum(torch.mul(roi_labels, torch.log(refine_prob_2[keep_inds]))) / roi_labels.shape[0]

    # ---------------caculating the loss of the Third branch

    roi_labels, roi_weights, keep_inds = self.get_refine_supervision(refine_prob_2,self._image_gt_summaries['ss_boxes'][self.ss_boxes_indexes, :],
                                                                     self._image_gt_summaries['image_level_label'])

    roi_weights = torch.tensor(roi_weights).cuda()
    roi_labels = torch.tensor(roi_labels, dtype=roi_weights.dtype).cuda()
    # roi_labels = torch.mul(roi_labels, roi_weights)
    refine_loss_3 = - torch.sum(torch.mul(roi_labels, torch.log(refine_prob_3[keep_inds]))) / roi_labels.shape[0]

    # ---------------caculating the loss of the rotate_invariant branch


    roi_labels, roi_weights, keep_inds, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_weights = rotation_invariant_graph_activate(self, refine_prob_1, refine_prob_2, \
                                                                           refine_prob_3, self._image_gt_summaries['ss_boxes'][self.ss_boxes_indexes, :], \
                                                                           self._image_gt_summaries['image_level_label'])

    roi_weights = torch.tensor(roi_weights).cuda()
    roi_labels = torch.tensor(roi_labels, dtype=roi_weights.dtype).cuda()
    bbox_targets = torch.tensor(bbox_targets, dtype=roi_weights.dtype).cuda()
    bbox_inside_weights = torch.tensor(bbox_inside_weights, dtype=roi_weights.dtype).cuda()
    bbox_outside_weights = torch.tensor(bbox_outside_weights, dtype=roi_weights.dtype).cuda()
    cls_weights = torch.tensor(cls_weights, dtype=roi_weights.dtype).cuda()


    loss_rotate = - torch.sum(torch.mul(roi_labels, torch.log(rotate_prob[keep_inds]))) / roi_labels.shape[0]

    l = len(self.ss_boxes_indexes)

    loss_box = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_weights[:l])

    
    self._losses['refine_loss_1'] = refine_loss_1
    self._losses['refine_loss_2'] = refine_loss_2
    self._losses['refine_loss_3'] = refine_loss_3
    self._losses['rotate_loss'] = loss_rotate
    self._losses['loss_box'] = loss_box
    #print('label ', label)
    
    label = torch.tensor(label, dtype=det_cls_prob.dtype, device=det_cls_prob.device)
    zeros = torch.zeros(det_cls_prob.shape, dtype=det_cls_prob.dtype, device=det_cls_prob.device)
    max_zeros = torch.max(zeros, 1-F.mul(label, det_cls_prob))
    cls_det_loss = torch.sum(max_zeros)
    self._losses['cls_det_loss'] = cls_det_loss / 20

    loss = cls_det_loss / 20 + refine_loss_1*0.1 + refine_loss_2*0.1 + refine_loss_3*0.1 + loss_rotate*0.1 + loss_box
    self._losses['total_loss'] = loss
    
    #print('loss ', loss)
    
    for k in self._losses.keys():
      self._event_summaries[k] = self._losses[k]

    return loss

  def _region_proposal(self, net_conv):
    rpn = F.relu(self.rpn_net(net_conv))
    self._act_summaries['rpn'] = rpn

    rpn_cls_score = self.rpn_cls_score_net(rpn) # batch * (num_anchors * 2) * h * w

    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = rpn_cls_score.view(1, 2, -1, rpn_cls_score.size()[-1]) # batch * 2 * (num_anchors*h) * w
    rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
    
    # Move channel to the last dimenstion, to fit the input of python functions
    rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
    rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
    rpn_cls_score_reshape = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous()  # batch * (num_anchors*h) * w * 2
    rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), 1)[1]

    rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
    rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

    if self._mode == 'TRAIN':
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred) # rois, roi_scores are varible
      rpn_labels = self._anchor_target_layer(rpn_cls_score)
      rois, _ = self._proposal_target_layer(rois, roi_scores)
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois

    return rois


  def get_refine_supervision(self, refine_prob, ss_boxes, image_level_label):
      '''
      refine_prob: num_box x 20 or num_box x 21
      ss_boxes; num_box x 4
      image_level_label: 1 dim vector with 20 elements
      '''
      
      cls_prob = refine_prob.data.cpu().numpy()
      #rois = ss_boxes.numpy()
      
      roi_per_image = cfg.TRAIN.MIL_BATCHSIZE
      
      if refine_prob.shape[1] == self._num_classes + 1:
          cls_prob = cls_prob[:, 1:]
      roi_labels = np.zeros([refine_prob.shape[0], self._num_classes + 1], dtype = np.int32)  # num_box x 21
      roi_labels[:,0] = 1                                                                        # the 0th elements is the bg
      roi_weights = np.zeros((refine_prob.shape[0], 1), dtype=np.float32)     # num_box x 1 weights of the rois
      
      max_score_box = np.zeros((0, 4), dtype = np.float32)
      max_box_score = np.zeros((0, 1), dtype = np.float32)
      max_box_classes = np.zeros((0, 1), dtype = np.int32)
      
      #print('ss_boxes ', ss_boxes[:5,:])
      for i in range(self._num_classes):
          if image_level_label[0, i] == 1:
              cls_prob_tmp = cls_prob[:, i]
              max_index = np.argmax(cls_prob_tmp)
              
              max_score_box = np.concatenate((max_score_box, ss_boxes[max_index, 1:].reshape(1, -1)), axis=0)
              max_box_classes = np.concatenate((max_box_classes, (i+1)*np.ones((1, 1), dtype=np.int32)), axis=0)
              max_box_score = np.concatenate((max_box_score, cls_prob_tmp[max_index]*np.ones((1, 1), dtype=np.float32)), axis=0)
      #print('image_level_labels ', image_level_label)
      #print('max_box_class ', max_box_classes)
      #print('max_box_score ', max_box_score)
      overlaps = bbox_overlaps(ss_boxes[:,1:], max_score_box)
      gt_assignment = overlaps.argmax(axis=1)
      max_over_laps = overlaps.max(axis=1)
      #print('max_over_laps', max_over_laps.max())
      #print('over laps', overlaps.shape)
      roi_weights[:, 0] = max_box_score[gt_assignment, 0]
      labels = max_box_classes[gt_assignment, 0]
      
      fg_inds = np.where(max_over_laps > cfg.TRAIN.MIL_FG_THRESH)[0]
      
      roi_labels[fg_inds,labels[fg_inds]] = 1
      roi_labels[fg_inds, 0] = 0
      
      bg_inds = (np.array(max_over_laps >= cfg.TRAIN.MIL_BG_THRESH_LO, dtype=np.int32) + \
                 np.array(max_over_laps < cfg.TRAIN.MIL_BG_THRESH_HI, dtype=np.int32)==2).nonzero()[0]
      
      if len(fg_inds) > 0 and len(bg_inds) > 0:
          fg_rois_num = min(cfg.TRAIN.MIL_NUM_FG, len(fg_inds))
          fg_inds = fg_inds[np.random.choice(np.arange(0, len(fg_inds)), size=int(fg_rois_num), replace=False)]
          
          bg_rois_num = min(cfg.TRAIN.MIL_NUM_BG, len(bg_inds))
          bg_inds = bg_inds[np.random.choice(np.arange(0, len(bg_inds)), size=int(bg_rois_num), replace=False)]
      
      elif len(fg_inds) > 0:
          fg_rois_num = min(cfg.TRAIN.MIL_NUM_FG, len(fg_inds))
          fg_inds = fg_inds[np.random.choice(np.arange(0, len(fg_inds)), size=int(fg_rois_num), replace=False)]
      elif len(bg_inds) > 0:
          bg_rois_num = min(cfg.TRAIN.MIL_NUM_BG, len(bg_inds))
          bg_inds = bg_inds[np.random.choice(np.arange(0, len(bg_inds)), size=int(bg_rois_num), replace=False)]
      else:
          import pdb
          pdb.set_trace()
      
      # print(len(fg_inds), len(bg_inds))
      keep_inds = np.concatenate([fg_inds, bg_inds])
      
      return roi_labels[keep_inds, :], roi_weights[keep_inds,0].reshape(-1,1), keep_inds
  def _region_classification(self, fc7_roi, fc7_context, fc7_frame, fc7_roi_1=None, fc7_roi_2=None):
    #cls_score = self.cls_score_net(fc7)
    #det_score = self.det_score_net(fc7)
    
    alpha = cfg.TRAIN.MIL_RECURRECT_WEIGHT
    
    
    refine_score_1 = self.refine_net_1(fc7_roi)
    if self._mode =='TRAIN':
      refine_score_2 = self.refine_net_2(fc7_roi_1)
      refine_score_3 = self.refine_net_3(fc7_roi_2)
      rotate_features = torch.cat((fc7_roi, fc7_roi_1, fc7_roi_2))
      rotate_score = self.rotate_invariant(rotate_features)
    else:
      refine_score_2 = self.refine_net_2(fc7_roi)
      refine_score_3 = self.refine_net_3(fc7_roi)
      rotate_score = self.rotate_invariant(fc7_roi)
      #rotate_features = fc7_roi

    cls_score = self.cls_score_net(fc7_roi)
    context_score = self.det_score_net(fc7_context)
    frame_score = self.det_score_net(fc7_frame)
    det_score = frame_score - context_score
    
    cls_prob = F.softmax(cls_score, dim=1)   #num x class_num
    det_prob = F.softmax(det_score, dim=0)   #num x class_num
    
    refine_prob_1 = F.softmax(refine_score_1, dim=1)  #num x class_num+1
    refine_prob_2 = F.softmax(refine_score_2, dim=1)  #num x class_num+1
    refine_prob_3 = F.softmax(refine_score_3, dim=1)  #num x class_num+1
    rotate_prob = F.softmax(rotate_score, dim=1)  # num x class_num+1
    
    det_cls_prob_product = F.mul(cls_score, det_prob)  #num x class_num
    det_cls_prob_product_2 = F.mul(cls_prob, det_prob)  # num x class_num
    det_cls_prob = torch.sum(det_cls_prob_product, 0) #1 x class_num or just a one dim vector whose size is class_num
    bbox_pred = self.bbox_pred_net(fc7_roi)
    #bbox_pred = torch.zeros(cls_prob.shape[0], 80)
    cls_pred = torch.max(cls_score, 1)[1]

    #print('cls_score ', cls_score.shape)
    #print('cls_pred ', cls_pred.shape)
    #print('cls_prob', cls_prob.shape)
    
    #print('cls_score ', cls_score)
    #print('cls_pred ', cls_pred)
    #print('cls_prob ', cls_prob)
    
    #print('det_prob ', det_prob)
    #print('det_cls_prob_product ', det_cls_prob_product)
    #print('det_cls_prob ', det_cls_prob)
    
    self._predictions["cls_score"] = cls_score
    self._predictions['det_score'] = det_score
    
    
    self._predictions["cls_prob"] = cls_prob
    self._predictions["det_prob"] = det_prob
    self._predictions['refine_prob_1'] = refine_prob_1
    self._predictions['refine_prob_2'] = refine_prob_2
    self._predictions['refine_prob_3'] = refine_prob_3
    self._predictions['rotate'] = rotate_prob

    
    self._predictions["cls_pred"] = cls_pred
    self._predictions["bbox_pred"] = bbox_pred
    
    self._predictions['det_cls_prob_product'] = det_cls_prob_product
    self._predictions['det_cls_prob_product_2'] = det_cls_prob_product_2
    self._predictions['det_cls_prob'] = det_cls_prob

    return cls_prob, det_prob, bbox_pred, det_cls_prob_product, det_cls_prob



  def _image_to_head(self):
    raise NotImplementedError

  def _head_to_tail(self, pool5):
    raise NotImplementedError

  def create_architecture(self, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._tag = tag

    self._num_classes = num_classes
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    assert tag != None

    # Initialize layers
    self._init_modules()

  def _init_modules(self):
    self._init_head_tail()

    # rpn
    #self.rpn_net = nn.Conv2d(self._net_conv_channels, cfg.RPN_CHANNELS, [3, 3], padding=1)

    #self.rpn_cls_score_net = nn.Conv2d(cfg.RPN_CHANNELS, self._num_anchors * 2, [1, 1])
    
    #self.rpn_bbox_pred_net = nn.Conv2d(cfg.RPN_CHANNELS, self._num_anchors * 4, [1, 1])

    #self.cls_score_net = RecuLayer.Linear(self._fc7_channels, self._num_classes)
    #self.det_score_net = RecuLayer.Linear(self._fc7_channels, self._num_classes)
    self.cls_score_net = nn.Linear(self._fc7_channels, self._num_classes)
    self.det_score_net = nn.Linear(self._fc7_channels, self._num_classes)
    self.bbox_pred_net = nn.Linear(self._fc7_channels, (self._num_classes + 1) * 4)
    
    self.refine_net_1 = nn.Linear(self._fc7_channels, self._num_classes + 1)
    self.refine_net_2 = nn.Linear(self._fc7_channels, self._num_classes + 1)
    self.refine_net_3 = nn.Linear(self._fc7_channels, self._num_classes + 1)
    self.rotate_invariant = nn.Linear(self._fc7_channels, self._num_classes + 1)
    
    self.init_weights()

  def _run_summary_op(self, val=False):
    """
    Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
    """
    summaries = []
    # Add image gt
    summaries.append(self._add_gt_image_summary())
    # Add event_summaries
    for key, var in self._event_summaries.items():                   #__event_summaries is equal to loss itmes
      summaries.append(tb.summary.scalar(key, var.item()))
    self._event_summaries = {}
    
    #if not val:
      # Add score summaries
      #for key, var in self._score_summaries.items():                 #_score_summaries is equal to _predictions which are output of the network
        #summaries.append(self._add_score_summary(key, var))
      #self._score_summaries = {}
      # Add act summaries
      # for unsurpvised vision and for Selective Search no rpn
      #for key, var in self._act_summaries.items():                   #_act_summaries is equal to rpn
      #  summaries += self._add_act_summary(key, var)
      #self._act_summaries = {}
      # Add train summaries
      #for k, var in dict(self.named_parameters()).items():
        #if var.requires_grad:
          #summaries.append(self._add_train_summary(k, var))

      #self._image_gt_summaries = {}
    
    return summaries

  def _predict_test(self, ss_boxes):
    # This is just _build_network in tf-faster-rcnn
    torch.backends.cudnn.benchmark = False
    net_conv, _, _ = self._image_to_head()

    # build the anchors for the image
    # self._anchor_component(net_conv.size(2), net_conv.size(3))

    # rois = self._region_proposal(net_conv)
    ss_rois = torch.from_numpy(ss_boxes).to(self._device)
    rois = ss_rois

    # This if for supervised version
    # if self._mode != 'TEST':
    #  ss_rois_ = torch.from_numpy(ss_boxes).to(self._device)
    #  rois, _ = self._proposal_target_layer(ss_rois, ss_rois_)
    self._predictions["rois"] = rois
    # print('rois ',rois.data.shape)

    # print('net_conv ', net_conv.shape)

    if cfg.POOLING_MODE == 'crop':
      pool5 = self._crop_pool_layer(net_conv, rois)
    else:
      pool5_roi = self._roi_ring_pool_layer(net_conv, rois, 0., 1.0)
      pool5_context = self._roi_ring_pool_layer(net_conv, rois, 1.0, 1.8)
      pool5_frame = self._roi_ring_pool_layer(net_conv, rois, scale_inner=1.0 / 1.8, scale_outer=1.0)

    if self._mode == 'TRAIN':
      torch.backends.cudnn.benchmark = True  # benchmark because now the input size are fixed
    # print('pool5 ', pool5.shape)
    fc7_roi = self._head_to_tail(pool5_roi)
    fc7_context = self._head_to_tail(pool5_context)
    fc7_frame = self._head_to_tail(pool5_frame)

    # print('fc7 ', fc7.shape)

    cls_prob, det_prob, bbox_pred, cls_det_prob_product, det_cls_prob = self._region_classification(fc7_roi,
                                                                                                    fc7_context,
                                                                                                    fc7_frame)

    for k in self._predictions.keys():
      self._score_summaries[k] = self._predictions[k]

    # print('last rois ', rois.shape)
    # print('las cls_prob ', cls_prob.shape)
    # print('las bbox_pred ', bbox_pred.shape)

    return rois, cls_prob, det_prob, bbox_pred, cls_det_prob_product, det_cls_prob

  def _predict(self, ss_boxes, ss_boxes_1, ss_boxes_2):
    # This is just _build_network in tf-faster-rcnn
    torch.backends.cudnn.benchmark = False
    net_conv_0, net_conv_1, net_conv_2 = self._image_to_head()

    #net_conv_0, net_conv_1, net_conv_2 = net_conv.chunk(3, dim=0)



    # build the anchors for the image
    # self._anchor_component(net_conv.size(2), net_conv.size(3))

    #rois = self._region_proposal(net_conv)
    ss_rois  = torch.from_numpy(ss_boxes).to(self._device)
    ss_rois_1 = torch.from_numpy(ss_boxes_1).to(self._device)
    ss_rois_2 = torch.from_numpy(ss_boxes_2).to(self._device)
    rois = ss_rois
    rois_1 = ss_rois_1
    rois_2 = ss_rois_2
    
    #This if for supervised version
    #if self._mode != 'TEST':
    #  ss_rois_ = torch.from_numpy(ss_boxes).to(self._device)
    #  rois, _ = self._proposal_target_layer(ss_rois, ss_rois_)
    self._predictions["rois"] = rois
    #print('rois ',rois.data.shape)
    
    #print('net_conv ', net_conv.shape)
    
    if cfg.POOLING_MODE == 'crop':
      pool5 = self._crop_pool_layer(net_conv_0, rois)
    else:
      pool5_roi = self._roi_ring_pool_layer(net_conv_0, rois, 0., 1.0)
      pool5_context = self._roi_ring_pool_layer(net_conv_0, rois, 1.0, 1.8)
      pool5_frame = self._roi_ring_pool_layer(net_conv_0, rois, scale_inner = 1.0 / 1.8, scale_outer = 1.0)
      pool5_roi_1 = self._roi_ring_pool_layer(net_conv_1, rois_1, 0., 1.0)
      pool5_roi_2 = self._roi_ring_pool_layer(net_conv_2, rois_2, 0., 1.0)

    if self._mode == 'TRAIN':
      torch.backends.cudnn.benchmark = True # benchmark because now the input size are fixed
    #print('pool5 ', pool5.shape)
    fc7_roi = self._head_to_tail(pool5_roi)
    fc7_roi_1 = self._head_to_tail(pool5_roi_1)
    fc7_roi_2 = self._head_to_tail(pool5_roi_2)
    fc7_context = self._head_to_tail(pool5_context)
    fc7_frame = self._head_to_tail(pool5_frame)
    
    #print('fc7 ', fc7.shape)
    
    cls_prob, det_prob, bbox_pred ,cls_det_prob_product ,det_cls_prob = self._region_classification(fc7_roi, fc7_context, fc7_frame, fc7_roi_1, fc7_roi_2)
    
    for k in self._predictions.keys():
      self._score_summaries[k] = self._predictions[k]
      
    #print('last rois ', rois.shape)
    #print('las cls_prob ', cls_prob.shape)
    #print('las bbox_pred ', bbox_pred.shape)
     
    return rois, cls_prob, det_prob, bbox_pred, cls_det_prob_product, det_cls_prob

  def forward(self, image, image1, image2, image_level_label ,im_info, gt_boxes=None, ss_boxes=None, ss_boxes_rotate_1=None, ss_boxes_rotate_2=None, mode='TRAIN'):
    #print('forward ss_boxes ', ss_boxes.shape)
    self._image_gt_summaries['image'] = image
    self._image_gt_summaries['image_level_label'] = image_level_label
    self._image_gt_summaries['gt_boxes'] = gt_boxes
    self._image_gt_summaries['im_info'] = im_info
    self._image_gt_summaries['ss_boxes'] = ss_boxes
    self._image_gt_summaries['ss_boxes1'] = ss_boxes_rotate_1
    self._image_gt_summaries['ss_boxes2'] = ss_boxes_rotate_2

    self._image = torch.from_numpy(image.transpose([0,3,1,2]).copy()).to(self._device)
    self._image1 = torch.from_numpy(image1.transpose([0, 3, 1, 2]).copy()).to(self._device)
    self._image2 = torch.from_numpy(image2.transpose([0, 3, 1, 2]).copy()).to(self._device)
    self._image_level_label = torch.from_numpy(image_level_label) if image_level_label is not None else None
    self._im_info = im_info # No need to change; actually it can be an list
    self._gt_boxes = torch.from_numpy(gt_boxes).to(self._device) if gt_boxes is not None else None

    self._mode = mode
    
    self.ss_boxes_indexes = self.return_ss_boxes(np.arange(ss_boxes.shape[0]), mode)
    if mode=='TRAIN':
      rois, cls_prob, det_prob, bbox_pred ,cls_det_prob_product ,det_cls_prob = self._predict(ss_boxes[self.ss_boxes_indexes, :], \
                                                                                              ss_boxes_rotate_1[self.ss_boxes_indexes, :], \
                                                                                              ss_boxes_rotate_2[self.ss_boxes_indexes, :])
    else:
      rois, cls_prob, det_prob, bbox_pred, cls_det_prob_product, det_cls_prob = self._predict_test(ss_boxes[self.ss_boxes_indexes, :])
    
    bbox_pred = bbox_pred[:,4:]
    
    if mode == 'TEST':
      self._predictions["bbox_pred"] = bbox_pred
      # stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
      # means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
      # self._predictions["bbox_pred"] = bbox_pred.mul(stds).add(means)


    else:
      self._add_losses() # compute losses

      
  def return_ss_boxes(self, boxes_index, mode='TRAIN'):
        if mode == 'TEST':
            return boxes_index
        box_num = min(1000, len(boxes_index))
        indexes = np.random.choice(boxes_index, size=box_num, replace=False)
        return indexes
        

  def init_weights(self):
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter
      if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        m.weight.data.normal_(mean, stddev)
      m.bias.data.zero_()
      
    #normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    #normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    #normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.det_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)
    normal_init(self.refine_net_1, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.refine_net_2, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.refine_net_3, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rotate_invariant, 0, 0.01, cfg.TRAIN.TRUNCATED)
    
    
  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, image):
    feat = self._layers["head"](torch.from_numpy(image.transpose([0,3,1,2])).to(self._device))
    return feat

  # only useful during testing mode
  def test_image(self, image, im_info, ss_boxes):
    self.eval()
    with torch.no_grad():
      self.forward(image, image, image, None, im_info, None, ss_boxes, mode='TEST')
    bbox_pred, rois , det_cls_prob, det_cls_prob_product, refine_prob_1, refine_prob_2, refine_prob_3, rotate_score = self._predictions['bbox_pred'].data.cpu().numpy(), \
                                                     self._predictions['rois'].data.cpu().numpy(), \
                                                     self._predictions['det_cls_prob'].data.cpu().numpy(), \
                                                     self._predictions['det_cls_prob_product'].data.cpu().numpy(),\
                                                     self._predictions['refine_prob_1'].data.cpu().numpy(), \
                                                     self._predictions['refine_prob_2'].data.cpu().numpy(), \
                                                     self._predictions['refine_prob_3'].data.cpu().numpy(), \
                                                     self._predictions['rotate'].data.cpu().numpy()
                                                     
    return bbox_pred, rois, det_cls_prob, det_cls_prob_product, refine_prob_1[:,1:], refine_prob_2[:,1:], refine_prob_3[:, 1:], rotate_score[:, 1:]
  def delete_intermediate_states(self):
    # Delete intermediate result to save memory
    for d in [self._losses, self._predictions, self._anchor_targets, self._proposal_targets]:
      for k in list(d):
        del d[k]

  def get_summary(self, blobs):
    self.eval()
    self.forward(blobs['data'], blobs['data1'], blobs['data2'], blobs['image_level_labels'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'], blobs['ss_boxes_rotate1'], blobs['ss_boxes_rotate2'])
    self.train()
    summary = self._run_summary_op(True)

    return summary

  def train_step(self, blobs, train_op):
    #commanded for this is for supervised version
    #self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'])
    self.forward(blobs['data'], blobs['data1'], blobs['data2'],blobs['image_level_labels'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'], blobs['ss_boxes_rotate1'], blobs['ss_boxes_rotate2'])
    cls_det_loss, refine_loss_1, refine_loss_2, refine_loss_3, rotate_loss, loss_box, loss = self._losses['cls_det_loss'].item(), \
                                      self._losses['refine_loss_1'].item(),  \
                                      self._losses['refine_loss_2'].item(),  \
                                      self._losses['refine_loss_3'].item(), \
                                      self._losses['rotate_loss'].item(), \
                                      self._losses['loss_box'].item(), \
                                      self._losses['total_loss'].item(), \
                                      
                                    
                                                                        
    #utils.timer.timer.tic('backward')
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    #utils.timer.timer.toc('backward')
    train_op.step()

    self.delete_intermediate_states()

    return cls_det_loss, refine_loss_1, refine_loss_2, refine_loss_3, rotate_loss, loss_box, loss

  def train_step_with_summary(self, blobs, train_op):
    #print(blobs.keys())
    #print('ss_boxes', blobs['ss_boxes'].shape)
    self.forward(blobs['data'], blobs['data1'], blobs['data2'], blobs['image_level_labels'],blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'], blobs['ss_boxes_rotate1'], blobs['ss_boxes_rotate2'])
    cls_det_loss, refine_loss_1, refine_loss_2, refine_loss_3, rotate_loss, loss_box, loss = self._losses["cls_det_loss"].item(), \
                                                       self._losses['refine_loss_1'].item(), \
                                                       self._losses['refine_loss_2'].item(), \
                                                       self._losses['refine_loss_3'].item(), \
                                                       self._losses['rotate_loss'].item(), \
                                                       self._losses['loss_box'].item(), \
                                                       self._losses['total_loss'].item()
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    summary = self._run_summary_op()

    self.delete_intermediate_states()

    return cls_det_loss, refine_loss_1, refine_loss_2, refine_loss_3, rotate_loss, loss_box, loss, summary

  def train_step_no_return(self, blobs, train_op):
    self.forward(blobs['data'], blobs['data1'], blobs['data2'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'])
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    self.delete_intermediate_states()

  def load_state_dict(self, state_dict):
    """
    Because we remove the definition of fc layer in resnet now, it will fail when loading 
    the model trained before.
    To provide back compatibility, we overwrite the load_state_dict
    """
    nn.Module.load_state_dict(self, {k: state_dict[k] for k in list(self.state_dict())})

