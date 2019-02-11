import math
import torch

from pdb import set_trace as bp


def min_wh_form_to_center_form(boxes):
  boxes2 = torch.tensor(boxes)
  return torch.cat([boxes2[...,:2] + boxes2[...,2:]/2, boxes2[...,2:]], boxes2.dim() - 1)


def center_form_to_corner_form(locations):
  return torch.cat([locations[..., :2] - locations[..., 2:]/2,
                    locations[..., :2] + locations[..., 2:]/2], locations.dim() - 1)


def corner_form_to_center_form(boxes):
  return torch.cat([
    (boxes[..., :2] + boxes[..., 2:]) / 2,
     boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)


def area_of(left_top, right_bottom) -> torch.Tensor:
  hw = torch.clamp(right_bottom - left_top, min=0.0)
  return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
  overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
  overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])
  
  overlap_area = area_of(overlap_left_top, overlap_right_bottom)
  area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
  area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
  return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors, iou_threshold):
  ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
  best_target_per_prior, best_target_per_prior_index = ious.max(1)
  best_prior_per_target, best_prior_per_target_index = ious.max(0)

  for target_index, prior_index in enumerate(best_prior_per_target_index):
    best_target_per_prior_index[prior_index] = target_index
  
  best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
  labels = gt_labels[best_target_per_prior_index]
  labels[best_target_per_prior < iou_threshold] = 0
  boxes = gt_boxes[best_target_per_prior_index]
  return boxes, labels


def convert_boxes_to_locations(center_form_boxes, center_form_priors,
                               center_variance, size_variance):
  if center_form_priors.dim() + 1 == center_form_boxes.dim():
    center_form_priors = center_form_priors.unsqueeze()
  return torch.cat([
    (center_form_boxes[..., :2] - center_form_priors[..., :2]) /\
      center_form_priors[..., 2:] / center_variance,
     torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance],
   dim=center_form_boxes.dim() - 1)


def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
  if priors.dim() + 1 == locations.dim():
    priors = priors.unsqueeze(0)
  return torch.cat([
    locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
    torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]], 
   dim=locations.dim() - 1)


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    return 0


def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)



