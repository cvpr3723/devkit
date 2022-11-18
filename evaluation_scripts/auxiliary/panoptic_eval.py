import torch
import pdb


class PanopticEvaluator():
  def __init__(self):
    self.evaluator = PanopticQuality()

  def update(self, semantic_pred, semantic_gt, instance_pred, instance_gt):
    self.evaluator.compute_pq(semantic_pred, semantic_gt, instance_pred, instance_gt)

  def compute(self):
    pq = self.evaluator.average_pq(self.evaluator.panoptic_qualities)

    return pq

  @property
  def results_classwise(self):
    return self.evaluator.panoptic_qualities


class PanopticQuality:
    def __init__(self):
        # initialization of a list of dictionaries with len = num classes
        self.panoptic_qualities = {}

    def reset(self):
        self.panoptic_qualities.clear()

    def compute_pq_single_class(self, prediction, groundtruth):
        numerator_iou = torch.tensor(0.)
        class_matches = []
        unmatched_class_predictions = []

        # take the non-zero INSTANCE labels for both prediction and groundtruth
        labels = torch.unique(prediction)
        gt_labels = torch.unique(groundtruth)
        labels = labels[labels > 0]
        gt_labels = gt_labels[gt_labels > 0]
        for label in labels:    # for each predicted label
            iou = torch.tensor(0.)
            best_gt_label = -1
            for gt_label in gt_labels:
                # compute iou with all instance gt labels and store the best
                intersection = (((prediction == label) & (
                    groundtruth == gt_label)).sum()).float()
                union = ((prediction == label).sum() + (groundtruth ==
                                                        gt_label).sum() - intersection).float()
                iou_tmp = intersection / union
                if iou_tmp > iou:
                    iou = iou_tmp
                    best_gt_label = gt_label
            # if the best iou is above 0.5, store the match pred_label-gt_label-iou
            if iou > 0.5:
                result = {'pred_label': label.item(
                ), 'gt_label': best_gt_label.item(), 'iou': iou.item()}
                class_matches.append(result)
                numerator_iou += iou
            # else, the predicted label remains unmatched
            else:
                unmatched_class_predictions.append(label.item())

        true_positives = len(class_matches)     # TP = number of matches
        # FP = number of unmatched predictions
        false_positives = len(unmatched_class_predictions)
        # FN = number of unmatched gt labels
        false_negatives = len(gt_labels) - len(class_matches)
        panoptic_quality_one_class = numerator_iou / \
            (true_positives + 0.5 * false_positives + 0.5 * false_negatives)

        return panoptic_quality_one_class, class_matches

    def compute_pq(self, semantic_pred, semantic_gt, instance_pred, instance_gt):
        # take the semantic labels (except 0 = void label)
        semantic_gt_labels = torch.unique(semantic_gt).int()
        semantic_gt_labels = semantic_gt_labels[semantic_gt_labels > 0].int()
        
        if semantic_gt_labels.sum() == 0:
            if instance_pred.sum() == 0:
                return 1.0, {}
            else:
                return 0.0 , {}


        for gt_label in semantic_gt_labels:
            # take pred and gt where they have sem_class = label
            semantic_tmp = (semantic_pred == gt_label).int()
            semantic_gt_tmp = (semantic_gt == gt_label).int()

            # take pred and gt instances corresponding to the considered semantic class
            instance_tmp = (instance_pred * semantic_tmp).int()
            instance_gt_tmp = (instance_gt * semantic_gt_tmp).int()

            pq, _ = self.compute_pq_single_class(instance_tmp, instance_gt_tmp)

            # update: count = number of images that contains the class, pq is the overall panoptic quality of the images
            if gt_label.item() not in self.panoptic_qualities.keys():
                self.panoptic_qualities[gt_label.item()] = {}
                self.panoptic_qualities[gt_label.item()]['count'] = 1
                self.panoptic_qualities[gt_label.item()]['pq'] = pq.item()
            else:
                self.panoptic_qualities[gt_label.item()]['count'] += 1
                self.panoptic_qualities[gt_label.item()]['pq'] = ((self.panoptic_qualities[gt_label.item()]['pq'] * (
                    self.panoptic_qualities[gt_label.item()]['count']-1) + pq) / self.panoptic_qualities[gt_label.item()]['count']).item()

        # panoptic_quality = self.average_pq(self.panoptic_qualities)
        # return panoptic_quality, self.panoptic_qualities

        return self.panoptic_qualities

    def average_pq(self, panoptic_qualities):
        # overall panoptic quality averaged from all classes
        pq = 0.
        n = 0
        for item in panoptic_qualities.keys():
            pq += panoptic_qualities[item]['pq']
            n += 1

        if n == 0:
            return -1
        pq /= n
        return pq

    def panoptic_quality_forward(self, batch_semantic_pred, batch_semantic_gt, batch_instance_pred, batch_instance_gt):
        batch_size = batch_semantic_pred.shape[0]
        for item in range(batch_size):
            semantic_gt = batch_semantic_gt[item]
            semantic_pred = batch_semantic_pred[item]
            instance_gt = batch_instance_gt[item]
            instance_pred = batch_instance_pred[item]
            # for each batch item
            pq, pqs = self.compute_pq(
                semantic_pred, semantic_gt, instance_pred, instance_gt)
        return pq, pqs


# class PanopticEvaluator():

#   def __init__(self, mode: PanopticEvalMode):
#     self.mode = mode

#     self.pq = PanopticQuality()
#     self.pq_list = []
#     self.class_pq_list = []
#     self.class_sq_list = []
#     self.class_rq_list = []

#   def update_pq(self, semantic_pred: torch.Tensor, semantic_gt: torch.Tensor, instance_pred: torch.Tensor, instance_gt: torch.Tensor):
#     self.pq.reset()
#     pq_class, sq_class, rq_class = self.pq.compute_pq(
#         semantic_pred,
#         semantic_gt,
#         instance_pred,
#         instance_gt,
#     )
#     # self.pq_list.append(pq_image)
#     self.class_pq_list.append(pq_class)
#     self.class_sq_list.append(sq_class)
#     self.class_rq_list.append(rq_class)

#   def update(self, semantic_pred: torch.Tensor, semantic_gt: torch.Tensor, instance_pred: torch.Tensor, instance_gt: torch.Tensor):
#     self.update_pq(semantic_pred, semantic_gt, instance_pred, instance_gt)

#   def compute(self):
#     metrics = {}
#     #pdb.set_trace()
#     class_pq = torch.tensor(self.class_pq_list).sum(dim=0) / len(self.class_pq_list)
#     class_sq = torch.tensor(self.class_sq_list).sum(dim=0) / len(self.class_sq_list)
#     class_rq = torch.tensor(self.class_rq_list).sum(dim=0) / len(self.class_rq_list)
#     self.class_pq_list = []
#     self.class_sq_list = []
#     self.class_rq_list = []
#     metrics["pq"] = class_pq.mean()
#     if self.mode == PanopticEvalMode.CROPS_WEEDS:
#       metrics["pq_crop"] = class_pq[0]
#       metrics["pq_weed"] = class_pq[1]
#       metrics["sq_crop"] = class_sq[0]
#       metrics["sq_weed"] = class_sq[1]
#       metrics["rq_crop"] = class_rq[0]
#       metrics["rq_weed"] = class_rq[1]
#     elif self.mode == PanopticEvalMode.LEAVES:
#       metrics["pq_leaves"] = class_pq[0]
#       metrics["sq_leaves"] = class_sq[0]
#       metrics["rq_leaves"] = class_rq[0]
#     elif self.mode == PanopticEvalMode.CROPS:
#       metrics["pq_crops"] = class_pq[0]
#       metrics["sq_crops"] = class_sq[0]
#       metrics["rq_crops"] = class_rq[0]
#     return metrics


# class PanopticQuality:

#   def __init__(self):
#     # initialization of a list of dictionaries with len = num classes
#     self.panoptic_qualities = {}

#   def reset(self):
#     self.panoptic_qualities.clear()

#   def compute_pq_single_class(self, prediction, groundtruth):
#     if prediction.sum() == 0 and groundtruth.sum() == 0:
#       # if no predictions and no plants in groundtruth return pq 1
#       return 1.0, 0
#     numerator_iou = torch.tensor(0.0)
#     class_matches = []
#     unmatched_class_predictions = []

#     # take the non-zero INSTANCE labels for both prediction and groundtruth
#     labels = torch.unique(prediction)
#     gt_labels = torch.unique(groundtruth)
#     labels = labels[labels > 0]
#     gt_labels = gt_labels[gt_labels > 0]
#     for label in labels:  # for each predicted label
#       iou = torch.tensor(0.0)
#       best_gt_label = -1
#       for gt_label in gt_labels:
#         # compute iou with all instance gt labels and store the best
#         intersection = (((prediction == label) & (groundtruth == gt_label)).sum()).float()
#         union = ((prediction == label).sum() + (groundtruth == gt_label).sum() - intersection).float()
#         iou_tmp = intersection / union
#         if iou_tmp > iou:
#           iou = iou_tmp
#           best_gt_label = gt_label
#       # if the best iou is above 0.5, store the match pred_label-gt_label-iou
#       if iou > 0.5:
#         result = {
#             "pred_label": label.item(),
#             "gt_label": best_gt_label.item(),
#             "iou": iou.item(),
#         }
#         class_matches.append(result)
#         numerator_iou += iou
#       # else, the predicted label remains unmatched
#       else:
#         unmatched_class_predictions.append(label.item())

#     true_positives = len(class_matches)  # TP = number of matches
#     false_positives = len(unmatched_class_predictions)  # FP = number of unmatched predictions
#     false_negatives = len(gt_labels) - len(class_matches)  # FN = number of unmatched gt labels
#     panoptic_quality_one_class = numerator_iou / (true_positives + 0.5 * false_positives + 0.5 * false_negatives)
#     if true_positives == 0:
#       sq = 0.0
#       rq = 0.0
#     else:
#       sq = numerator_iou / true_positives
#       rq = true_positives / (true_positives + 0.5 * false_positives + 0.5 * false_negatives)

#     return panoptic_quality_one_class, sq, rq

#   def compute_pq(self, semantic_pred, semantic_gt, instance_pred, instance_gt):
#     # take the semantic labels (except 0 = void label)
#     semantic_gt_labels = torch.unique(semantic_gt).int()
#     semantic_gt_labels = semantic_gt_labels[semantic_gt_labels > 0].int()
#     for gt_label in semantic_gt_labels:
#       # take pred and gt where they have sem_class = label
#       semantic_tmp = (semantic_pred == gt_label).int()
#       semantic_gt_tmp = (semantic_gt == gt_label).int()

#       # take pred and gt instances corresponding to the considered semantic class
#       instance_tmp = (instance_pred * semantic_tmp).int()
#       instance_gt_tmp = (instance_gt * semantic_gt_tmp).int()

#       pq, sq, rq = self.compute_pq_single_class(instance_tmp, instance_gt_tmp)
      
#       # update: count = number of images that contains the class, pq is the overall panoptic quality of the images
#       if gt_label.item() not in self.panoptic_qualities.keys():
#         self.panoptic_qualities[gt_label.item()] = {}
#         self.panoptic_qualities[gt_label.item()]["count"] = 1
#         self.panoptic_qualities[gt_label.item()]["pq"] = pq.item()
#         self.panoptic_qualities[gt_label.item()]["sq"] = sq
#         self.panoptic_qualities[gt_label.item()]["rq"] = rq
#       else:
#         assert False  # "Call this function with batch size 1"
#         # self.panoptic_qualities[gt_label.item()]["count"] += 1
#         # self.panoptic_qualities[gt_label.item()]["pq"] = (
#         #     (
#         #         self.panoptic_qualities[gt_label.item()]["pq"]
#         #         * (self.panoptic_qualities[gt_label.item()]["count"] - 1)
#         #         + pq
#         #     )
#         #     / self.panoptic_qualities[gt_label.item()]["count"]
#         # ).item()

#     # panoptic_quality = self.average_pq(self.panoptic_qualities)
#     pdb.set_trace()
#     return (
#         [val["pq"] for key, val in self.panoptic_qualities.items()],
#         [val["sq"] for key, val in self.panoptic_qualities.items()],
#         [val["rq"] for key, val in self.panoptic_qualities.items()],
#     )

#   def average_pq(self, panoptic_qualities):
#     # overall panoptic quality averaged from all classes
#     pq = 0.0
#     n = 0
#     for item in panoptic_qualities.keys():
#       pq += panoptic_qualities[item]["pq"]
#       n += 1
#     if n == 0:
#       return -1
#     pq /= n
#     return pq
