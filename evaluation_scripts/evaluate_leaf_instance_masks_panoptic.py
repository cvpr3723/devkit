import argparse
import pdb
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from auxiliary.common import get_png_files_in_dir, load_file_as_tensor, load_file_as_int_tensor
from auxiliary.filter import filter_partial_masks
from auxiliary.panoptic_eval import PanopticEvaluator


def parse_args() -> Dict:
  parser = argparse.ArgumentParser()
  parser.add_argument('--ground_truth_dir', required=True, type=Path, help='Path to ground truth directory.')
  parser.add_argument('--prediction_dir', required=True, type=Path, help='Path to prediction directory.')

  # Required directoy structure
  # ├── ground-truth
  # |   ├── leaf_visibility
  # │   └── leaf_instances
  # └── prediction
  #     └── leaf_instances

  args = vars(parser.parse_args())

  return args

def main():
  args = parse_args()

  # ------- Ground Truth -------
  gt_instance_fnames = get_png_files_in_dir(args['ground_truth_dir'] / 'leaf_instances')
  gt_visibility_fnames = get_png_files_in_dir(args['ground_truth_dir'] / 'leaf_visibility')

  # ------- Prediction -------
  pred_fnames = get_png_files_in_dir(args['prediction_dir'] / 'leaf_instances')

  # ------- Setup evaluator -------
  evaluator = PanopticEvaluator()

  n_total = len(gt_instance_fnames)
  for gt_instance_fname, gt_visibility_fname, pred_fname in tqdm(zip(gt_instance_fnames, gt_visibility_fnames, pred_fnames), total=n_total):
    assert gt_instance_fname == gt_visibility_fname == pred_fname

    gt_instance_map = load_file_as_tensor(args['ground_truth_dir'] / 'leaf_instances' / gt_instance_fname).squeeze()  # [H x W]
    gt_visibility = load_file_as_tensor(args['ground_truth_dir'] / 'leaf_visibility' / gt_visibility_fname).squeeze()  # [H x W]
    gt_semantics = (gt_instance_map > 0).type(torch.uint8)  # derive semantics from instance map

    pred_instance_map = load_file_as_int_tensor(args['prediction_dir'] / 'leaf_instances' / pred_fname).squeeze()  # [H x W]
    pred_semantics = (pred_instance_map > 0).type(torch.uint8)  # derive semantics from instance map

    filter_partial_masks(pred_instance_map, pred_semantics, gt_instance_map, gt_semantics, gt_visibility)

    evaluator.update(pred_semantics, gt_semantics, pred_instance_map, gt_instance_map)

  metrics = evaluator.compute()
  print(f"{5*'#'} Panoptic Evaluation Leaf Instances {5*'#'}")
  print('Average metric over all classes: ')
  print(metrics)
  print('Classwise metric: ')
  print(evaluator.results_classwise)


if __name__ == '__main__':
  main()
