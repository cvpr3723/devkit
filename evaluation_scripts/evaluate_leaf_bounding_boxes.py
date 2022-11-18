import argparse
import pdb
from pathlib import Path
from typing import Dict

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision  # type: ignore
from tqdm import tqdm

from auxiliary.common import (get_png_files_in_dir, get_txt_files_in_dir,
                              load_file_as_tensor, load_yolo_txt_file)
from auxiliary.convert import cvt_gt_to_bbox_map, cvt_yolo_to_bbox_map
from auxiliary.filter import filter_partials_boxes

def parse_args() -> Dict:
  parser = argparse.ArgumentParser()
  parser.add_argument('--ground_truth_dir', required=True, type=Path, help='Path to ground truth directory.')
  parser.add_argument('--prediction_dir', required=True, type=Path, help='Path to prediction directory.')

  # Required directoy structure
  # ├── ground-truth
  # │   ├── leaf_instances
  # │   ├── leaf_visibility
  # └── prediction
  #     └── leaf_bboxes

  args = vars(parser.parse_args())

  return args


def main():
  args = parse_args()

  # ------- Ground Truth -------
  gt_instance_fnames = get_png_files_in_dir(args['ground_truth_dir'] / 'leaf_instances')
  gt_visibility_fnames = get_png_files_in_dir(args['ground_truth_dir'] / 'leaf_visibility')

  # ------- Predictions -------
  pred_bboxes_fnames = get_txt_files_in_dir(args['prediction_dir'] / 'leaf_bboxes')

  # ------- Setup evaluator -------
  evaluator = MeanAveragePrecision(box_format='cxcywh')

  for gt_instance_fname, gt_visibility_fname, pred_bboxes_fnames in tqdm(zip(gt_instance_fnames, gt_visibility_fnames, pred_bboxes_fnames),total=len(gt_instance_fnames)):
    assert gt_instance_fname.split('.')[0] == gt_visibility_fname.split('.')[0] == pred_bboxes_fnames.split('.')[0]

    gt_instance_map = load_file_as_tensor(args['ground_truth_dir'] / 'leaf_instances' / gt_instance_fname).squeeze()  # [H x W]
    gt_visibility = load_file_as_tensor(args['ground_truth_dir'] / 'leaf_visibility' / gt_visibility_fname).squeeze()  # [H x W]
    gt_semantics = (gt_instance_map > 0).type(torch.uint8)  # derive semantics from instance map
    targets = cvt_gt_to_bbox_map(gt_instance_map, gt_semantics, gt_visibility)

    yolo_pred = load_yolo_txt_file(args['prediction_dir'] / 'leaf_bboxes' / pred_bboxes_fnames)
    preds = cvt_yolo_to_bbox_map(yolo_pred)

    filter_partials_boxes(preds[0], targets[0])

    evaluator.update(preds, targets)

  metrics = evaluator.compute()
  print(f"{5*'#'} Bounding Box Evaluation Leaf Instances {5*'#'}")
  print(metrics)


if __name__ == '__main__':
  main()
