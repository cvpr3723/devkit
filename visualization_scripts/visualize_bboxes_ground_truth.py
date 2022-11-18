import argparse
import pdb
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from auxiliary.common import get_png_files_in_dir, get_txt_files_in_dir
from auxiliary.visualizers import BoundingBoxFromMaskVisualizer


def parse_args() -> Dict[str, Path]:
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_dir", required=True, type=Path, help='Path to dir containing RGB images.')
  parser.add_argument("--semantic_dir", required=True, type=Path, help='Path to dir containing semantics')
  parser.add_argument("--instance_dir", required=True, type=Path, help='Path to dir containing instance masks')
  parser.add_argument("--export_dir", required=True, type=Path, help='Path to export directory')
  
  args = vars(parser.parse_args())

  args['export_dir'].mkdir(parents=True, exist_ok=True)

  return args

def main():
  args = parse_args()

  img_fnames = get_png_files_in_dir(args['image_dir'])
  sem_fnames = get_png_files_in_dir(args['semantic_dir'])
  ins_fnames = get_png_files_in_dir(args['instance_dir'])

  visualizer = BoundingBoxFromMaskVisualizer()

  n_total = len(img_fnames)
  for img_fname, sem_fname, ins_fnames in tqdm(zip(img_fnames, sem_fnames, ins_fnames), total=n_total):
    assert img_fname == sem_fname == ins_fnames

    img = np.array(Image.open(args['image_dir'] / img_fname))
    sem = np.array(Image.open(args['semantic_dir'] / sem_fname))
    ins = np.array(Image.open(args['instance_dir'] / ins_fnames))

    vis = visualizer.visualize(img, ins, sem)

    vis_fpath = args['export_dir'] / img_fname
    plt.savefig(vis_fpath, bbox_inches='tight', pad_inches = 0)
    plt.close('all')

if __name__ == '__main__':
  main()


