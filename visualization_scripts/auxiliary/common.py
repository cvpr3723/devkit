import os
from pathlib import Path
from typing import List


def is_png(filename: str) -> bool:
  return filename.endswith('.png')


def is_txt(filename: str) -> bool:
  return filename.endswith('.txt')


def get_png_files_in_dir(path_to_dir: Path) -> List[str]:
  filenames = sorted([filename for filename in os.listdir(path_to_dir) if is_png(filename)])

  return filenames


def get_txt_files_in_dir(path_to_dir: Path) -> List[str]:
  filenames = sorted([filename for filename in os.listdir(path_to_dir) if is_txt(filename)])

  return filenames

