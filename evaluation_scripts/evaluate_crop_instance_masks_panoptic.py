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
  # │   ├── plant_instances
  # |   ├── plant_visibility
  # │   └── semantics
  # └── prediction
  #     └── plant_instances

  args = vars(parser.parse_args())

  return args


def main():
  args = parse_args()

  # ------- Ground Truth -------
  gt_instance_fnames = get_png_files_in_dir(args['ground_truth_dir'] / 'plant_instances')
  gt_visibility_fnames = get_png_files_in_dir(args['ground_truth_dir'] / 'plant_visibility')
  gt_semantic_fnames = get_png_files_in_dir(args['ground_truth_dir'] / 'semantics')

  # ------- Predictions -------
  pred_instance_fnames = get_png_files_in_dir(args['prediction_dir'] / 'plant_instances')

  # ------- Setup evaluator -------
  evaluator = PanopticEvaluator()

  n_total = len(gt_instance_fnames)
  for gt_instance_fname, gt_semantic_fname, gt_visibility_fname, pred_instance_fname, in tqdm(zip(gt_instance_fnames, gt_semantic_fnames, gt_visibility_fnames, pred_instance_fnames), total=n_total):
    assert gt_instance_fname == gt_semantic_fname == gt_visibility_fname == pred_instance_fname

    gt_instance_map = load_file_as_tensor(args['ground_truth_dir'] / 'plant_instances' / gt_instance_fname).squeeze()  # [H x W]
    gt_semantics = load_file_as_tensor(args['ground_truth_dir'] / 'semantics' / gt_semantic_fname).squeeze()  # [H x W]
    gt_visibility = load_file_as_tensor(args['ground_truth_dir'] / 'plant_visibility' / gt_instance_fname).squeeze()  # [H x W]

    gt_crop_mask = (gt_semantics == 1) | (gt_semantics == 3) # remove any weeds from ground truth
    gt_instance_map[~gt_crop_mask] = 0
    gt_semantics[~gt_crop_mask] = 0

    pred_instance_map = load_file_as_int_tensor(args['prediction_dir'] / 'plant_instances' / pred_instance_fname).squeeze()  # [H x W]
    pred_semantics = (pred_instance_map > 0).type(torch.uint8)  # derive semantics from instance map

    filter_partial_masks(pred_instance_map, pred_semantics, gt_instance_map, gt_semantics, gt_visibility)

    evaluator.update(pred_semantics, gt_semantics, pred_instance_map, gt_instance_map)

  metrics = evaluator.compute()
  print(f"{5*'#'} Panoptic Evaluation Plant Instances {5*'#'}")
  print(metrics)
  print('Classwise metric: ')
  print(evaluator.results_classwise)


if __name__ == '__main__':
  main()
