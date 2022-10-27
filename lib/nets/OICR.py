from model.config import cfg
from utils.bbox import bbox_overlaps
import numpy as np
from sklearn.cluster import KMeans
import torch
from model.nms_wrapper import nms


def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    kmeans = KMeans(n_clusters=3, random_state=3).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)
    index = np.where(kmeans.labels_ == high_score_label)[0]
    if len(index) == 0:
        index = np.array([np.argmax(probs)])
    return index


def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    overlaps = bbox_overlaps(
        boxes.astype(dtype=np.float32, copy=False),
        boxes.astype(dtype=np.float32, copy=False))
    return (overlaps > iou_threshold).astype(np.float32)


def _get_graph_centers_rotate(all_boxes, all_prob, im_labels):
    """Get graph centers."""
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    fgt_boxes = np.zeros((0, 4), dtype=np.float32)
    fgt_classes = np.zeros((0, 1), dtype=np.int32)
    fgt_scores = np.zeros((0, 1), dtype=np.float32)
    #index = np.arange(20)

    for r in range(len(all_prob)):
        cls_prob = all_prob[r]
        boxes = all_boxes.copy()
        for i in range(num_classes):
            if im_labels_tmp[i] == 1:
                cls_prob_tmp = cls_prob[:, i].copy()
                idxs = np.where(cls_prob_tmp >= 0)[0]
                if idxs.shape[0] == 0:
                    print('kmeans problem')
                    continue
                idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
                idxs = idxs[idxs_tmp]
                boxes_tmp = boxes[idxs, :].copy()
                cls_prob_tmp = cls_prob_tmp[idxs]
                graph = _build_graph(boxes_tmp, 0.4)

                keep_idxs = []
                gt_scores_tmp = []
                count = cls_prob_tmp.size
                while True:
                    order = np.sum(graph, axis=1).argsort()[::-1]
                    tmp = order[0]
                    keep_idxs.append(tmp)
                    inds = np.where(graph[tmp, :] > 0)[0]
                    gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))

                    graph[:, inds] = 0
                    graph[inds, :] = 0
                    count = count - len(inds)
                    if count <= 5:
                        break

                gt_boxes_tmp = boxes_tmp[keep_idxs, :].copy()
                gt_scores_tmp = np.array(gt_scores_tmp).copy()

                keep_idxs_new = np.argsort(gt_scores_tmp)[-1:(-1 - min(len(gt_scores_tmp), 5)):-1]

                gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
                gt_scores = np.vstack((gt_scores, gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
                gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((len(keep_idxs_new), 1), dtype=np.int32)))

                # If a proposal is chosen as a cluster center,
                # we simply delete a proposal from the candidata proposal pool,
                # because we found that the results of different strategies are similar and this strategy is more efficient
                cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)
                boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)

                overlaps = bbox_overlaps(
                    gt_boxes.astype(dtype=np.float32, copy=False),
                    all_boxes.astype(dtype=np.float32, copy=False))
                max_over_laps = overlaps.max(axis=0)

                fg_inds = np.where(max_over_laps == 1)[0]

                #zero_index = np.where(index != i)[0]
                r1_score = all_prob[1][fg_inds, :].copy()
                r1_score[:, i] = 0
                all_prob[1][fg_inds] = all_prob[1][fg_inds] - r1_score

                r2_score = all_prob[2][fg_inds, :].copy()
                r2_score[:, i] = 0
                all_prob[2][fg_inds] = all_prob[2][fg_inds] - r2_score

    boxes = np.hstack((gt_boxes, gt_scores)).astype(np.float32, copy=False)
    keep = nms(torch.from_numpy(boxes), 0.8).numpy()
    fgt_boxes = np.vstack((fgt_boxes, gt_boxes[keep, :]))
    fgt_classes = np.vstack((fgt_classes, gt_classes[keep, :]))
    fgt_scores = np.vstack((fgt_scores, gt_scores[keep, :]))


    proposals = {'gt_boxes': fgt_boxes,
                 'gt_classes': fgt_classes,
                 'gt_scores': fgt_scores}

    return proposals

def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in range(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            if idxs.shape[0] == 0:
                print('kmeans problem')
                continue
            idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[idxs, :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]
            graph = _build_graph(boxes_tmp, 0.4)

            keep_idxs = []
            gt_scores_tmp = []
            count = cls_prob_tmp.size
            while True:
                order = np.sum(graph, axis=1).argsort()[::-1]
                tmp = order[0]
                keep_idxs.append(tmp)
                inds = np.where(graph[tmp, :] > 0)[0]
                gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))

                graph[:, inds] = 0
                graph[inds, :] = 0
                count = count - len(inds)
                if count <= 5:
                    break

            gt_boxes_tmp = boxes_tmp[keep_idxs, :].copy()
            gt_scores_tmp = np.array(gt_scores_tmp).copy()

            keep_idxs_new = np.argsort(gt_scores_tmp)[-1:(-1 - min(len(gt_scores_tmp), 5)):-1]

            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
            gt_scores = np.vstack((gt_scores, gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((len(keep_idxs_new), 1), dtype=np.int32)))

            # If a proposal is chosen as a cluster center,
            # we simply delete a proposal from the candidata proposal pool,
            # because we found that the results of different strategies are similar and this strategy is more efficient
            cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)
            boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)

    proposals = {'gt_boxes': gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals


def _get_proposal_clusters(self, refine_prob, all_rois, rois, proposals, im_labels):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    #gt_boxes_tar = np.zeros((0, 4), dtype=np.float32)
    roi_labels = np.zeros([refine_prob.shape[0], self._num_classes + 1], dtype=np.int32)  # num_box x 21
    roi_labels[:, 0] = 1  # the 0th elements is the bg
    roi_weights = np.zeros((refine_prob.shape[0], 1), dtype=np.float32)
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_over_laps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    roi_weights[:, 0] = gt_scores[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]
    ig_inds = np.where(max_over_laps < 0.1)[0]
    cls_loss_weights[ig_inds] = 0.0

    #gt_inds = np.where(max_over_laps == 1)[0]

    fg_inds = np.where(max_over_laps > cfg.TRAIN.MIL_FG_THRESH)[0]

    roi_labels[fg_inds, labels[fg_inds]] = 1
    roi_labels[fg_inds, 0] = 0

    bg_inds = (np.array(max_over_laps >= cfg.TRAIN.MIL_BG_THRESH_LO, dtype=np.int32) + \
               np.array(max_over_laps < cfg.TRAIN.MIL_BG_THRESH_HI, dtype=np.int32) == 2).nonzero()[0]
    labels[bg_inds] = 0

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

    if True:#cfg.MODEL.WITH_FRCNN:
        #gt_boxes_tar = np.vstack((gt_boxes_tar, all_rois_pred[gt_inds, :]))
        #gt_boxes_tar = all_rois_pred[gt_inds, :]
        fg_assignment=gt_assignment[:len(rois)]
        bbox_targets = _compute_targets(rois, gt_boxes[fg_assignment, :],
            labels[:len(rois)])
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(bbox_targets)
        bbox_outside_weights = np.array(
            bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype) \
            * cls_loss_weights[:len(rois)].reshape(-1, 1)
    #else:
    #    bbox_targets, bbox_inside_weights, bbox_outside_weights = np.array([0]), np.array([0]), np.array([0])

    gt_assignment[bg_inds] = -1

    return roi_labels[keep_inds, :], roi_weights[keep_inds, 0].reshape(-1, 1), keep_inds, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_loss_weights




def OICR(self, refine_prob, ss_boxes, image_level_label):
    '''
    refine_prob: num_box x 20 or num_box x 21
    ss_boxes; num_box x 4
    image_level_label: 1 dim vector with 20 elements
    '''

    cls_prob = refine_prob.data.cpu().numpy()
    # rois = ss_boxes.numpy()

    roi_per_image = cfg.TRAIN.MIL_BATCHSIZE

    if refine_prob.shape[1] == self._num_classes + 1:
        cls_prob = cls_prob[:, 1:]

    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps
    proposals = _get_graph_centers(ss_boxes[:, 1:], cls_prob, image_level_label)

    roi_labels, roi_weights, keep_inds, bbox_targets, bbox_inside_weights, bbox_outside_weights, _ = _get_proposal_clusters(self, cls_prob, ss_boxes[:, 1:], proposals,
                                                                     image_level_label)

    return roi_labels, roi_weights, keep_inds

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform_inv(ex_rois, gt_rois,
                                           cfg.TRAIN.BBOX_REG_WEIGHTS)
    return np.hstack((labels[:, np.newaxis], targets)).astype(
        np.float32, copy=False)


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.
    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = 20 + 1

    clss = bbox_target_data[:, 0]
    bbox_targets = zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def rotation_invariant_graph_activate(self, refine_prob1, refine_prob2, refine_prob3, ss_boxes, ss_boxes1, ss_boxes2, image_level_label):
    '''
    refine_prob: num_box x 20 or num_box x 21
    ss_boxes; num_box x 4
    image_level_label: 1 dim vector with 20 elements
    '''
    refine_prob = torch.cat([refine_prob1, refine_prob2, refine_prob3])

    cls_prob = refine_prob.data.cpu().numpy()
    cls_prob1 = refine_prob1.data.cpu().numpy()
    cls_prob2 = refine_prob2.data.cpu().numpy()
    cls_prob3 = refine_prob3.data.cpu().numpy()


    if refine_prob1.shape[1] == self._num_classes + 1:
        cls_prob1 = cls_prob1[:, 1:]
        cls_prob2 = cls_prob2[:, 1:]
        cls_prob3 = cls_prob3[:, 1:]

    eps = 1e-9
    cls_prob1[cls_prob1 < eps] = eps
    cls_prob1[cls_prob1 > 1 - eps] = 1 - eps
    cls_prob2[cls_prob2 < eps] = eps
    cls_prob2[cls_prob2 > 1 - eps] = 1 - eps
    cls_prob3[cls_prob3 < eps] = eps
    cls_prob3[cls_prob3 > 1 - eps] = 1 - eps
    all_prob = [cls_prob1, cls_prob2, cls_prob3]
    boxes = np.vstack((ss_boxes[:, 1:], ss_boxes[:, 1:], ss_boxes[:, 1:]))
    #pred_boxes =  np.vstack((ss_boxes[:, 1:], ss_boxes1[:, 1:], ss_boxes2[:, 1:]))
    proposals = _get_graph_centers_rotate(ss_boxes[:, 1:], all_prob, image_level_label)

    roi_labels, roi_weights, keep_inds, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_weights = _get_proposal_clusters(self, cls_prob, boxes, ss_boxes[:, 1:], proposals,
                                                                         image_level_label)

    return roi_labels, roi_weights, keep_inds, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_weights

def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, np.log(1000. / 8.))
    dh = np.minimum(dh, np.log(1000. / 8.))

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


def bbox_transform_inv(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * np.log(gt_widths / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw,
                         targets_dh)).transpose()
    return targets

def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def zeros(shape, int32=False):
    """Return a blob of all zeros of the given shape with the correct float or
    int data type.
    """
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)


def ones(shape, int32=False):
    """Return a blob of all ones of the given shape with the correct float or
    int data type.
    """
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)

def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_loss_weights, beta=1.0):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    1 / N * sum_i alpha_out[i] * SmoothL1(alpha_in[i] * (y_hat[i] - y[i])).
    N is the number of batch elements in the input predictions
    """
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < beta).detach().float()
    in_loss_box = smoothL1_sign * 0.5 * torch.pow(in_box_diff, 2) / beta + \
                  (1 - smoothL1_sign) * (abs_in_box_diff - (0.5 * beta))
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    N = (cls_loss_weights > 0).sum()
    loss_box = loss_box.view(-1).sum(0) / torch.tensor(N, dtype=loss_box.dtype).cuda()
    return loss_box


