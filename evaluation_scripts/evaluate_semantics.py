import argparse
import pdb
from pathlib import Path
from typing import Dict

import torch
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassJaccardIndex  # type: ignore
from tqdm import tqdm

from auxiliary.common import (get_png_files_in_dir, load_file_as_int_tensor,
                              load_file_as_tensor)
from auxiliary.convert import convert_partial_semantics

torch.set_printoptions(precision=4, sci_mode=False)

def parse_args() -> Dict:
  parser = argparse.ArgumentParser()
  parser.add_argument('--ground_truth_dir', required=True, type=Path, help='Path to ground truth directory.')
  parser.add_argument('--prediction_dir', required=True, type=Path, help='Path to prediction directory.')

  # Required directoy structure
  # ├── ground-truth
  # │   └── semantics
  # └── prediction
  #     └── semantics

  args = vars(parser.parse_args())

  return args


def main():
  args = parse_args()

  # ------- Get all ground truth and prediction files -------
  gt_fnames = get_png_files_in_dir(args['ground_truth_dir'] / 'semantics')
  pred_fnames = get_png_files_in_dir(args['prediction_dir'] / 'semantics')

  # ------- Setup evaluator -------
  evaluator = MulticlassJaccardIndex(num_classes=3, average=None)
  conf_mat_evaluator = ConfusionMatrix(num_classes=3, normalize='true')

  n_total = len(gt_fnames)

  for gt_fname, pred_fname in tqdm(zip(gt_fnames, pred_fnames), total=n_total):
    assert gt_fname == pred_fname

    semantics_gt = convert_partial_semantics(load_file_as_tensor(args['ground_truth_dir'] / 'semantics' / gt_fname)).squeeze()
    semantics_pred = convert_partial_semantics(load_file_as_int_tensor(args['prediction_dir'] / 'semantics' / pred_fname)).squeeze()

    evaluator.update(semantics_pred, semantics_gt)
    conf_mat_evaluator.update(semantics_pred, semantics_gt)

  metrics = evaluator.compute()  # tensor of shape [3]
  confusion_matrix = conf_mat_evaluator.compute()

  eval_results = {}
  eval_results['soil'] = float(metrics[0])
  eval_results['crop'] = float(metrics[1])
  eval_results['weed'] = float(metrics[2])
  eval_results['mIoU'] = float(metrics.mean())

  print(f"{5*'#'} Semantic Evaluation {5*'#'}")
  print(eval_results)
  print(f"{5*'#'} Confusion Matrix {5*'#'}")
  print(confusion_matrix)
  
if __name__ == '__main__':
  main()
