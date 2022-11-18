import argparse
import pdb
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from auxiliary.common import get_png_files_in_dir
from auxiliary.visualizers import InstanceSegmentationVisualizer


def parse_args() -> Dict[str, Path]:
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_dir", required=True, type=Path, help='Path to dir containing RGB images.')
  parser.add_argument("--pred_dir", required=True, type=Path, help='Path to dir containing predictions')
  parser.add_argument("--export_dir", required=True, type=Path, help='Path to export directory')
  
  args = vars(parser.parse_args())

  args['export_dir'].mkdir(parents=True, exist_ok=True)

  return args

def main():
  args = parse_args()

  img_fnames = get_png_files_in_dir(args['image_dir'])
  pred_fnames = get_png_files_in_dir(args['pred_dir'])

  visualizer = InstanceSegmentationVisualizer()

  n_total = len(img_fnames)
  for img_fname, pred_fname in tqdm(zip(img_fnames, pred_fnames), total=n_total):
    assert img_fname == pred_fname

    img = np.array(Image.open(args['image_dir'] / img_fname))
    pred = np.array(Image.open(args['pred_dir'] / pred_fname))

    vis = visualizer.visualize(img, pred)

    vis_fpath = args['export_dir'] / img_fname
    vis.save(vis_fpath)

if __name__ == '__main__':
  main()


