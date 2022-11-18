
import pdb
import random
from abc import ABC, abstractmethod
from typing import Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImagePIL


def random_rgb(seed: int = 42) -> Tuple[int, int, int]:
  random.seed(seed)

  red = random.randint(0, 255)
  green = random.randint(0,255)
  blue = random.randint(0,255)

  return red, green, blue

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def blend_images(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
  """ Blend two images 

  Args:
      image1 (np.ndarray): 1st image of shape [H x W x 3]
      image2 (np.ndarray): 2nd image of shape [H x W x 3]
      alpha (float, optional): strength of blending for 1st image. Defaults to 0.5.

  Returns:
      np.ndarray: blended image of shape [H x W x 3]
  """
  assert alpha <= 1.0
  assert alpha >= 0.0
  assert image1.shape == image2.shape

  image1 = image1.astype(np.float32)
  image2 = image2.astype(np.float32)

  blended = alpha * image1 + (1 - alpha) * image2
  blended = np.round(blended).astype(np.uint8)

  return blended

class Visualizer(ABC):
  """ Basic representation of a visualizer.
  """
  @abstractmethod
  def visualize(self, image: np.ndarray, prediction: np.ndarray) -> Union[ImagePIL, None]:
    """ Visualize the prediction with the original image in the background.

    Args:
        image (np.ndarray): RGB image of shape [H x W x 3]
        prediction (np.ndarray): Prediction of shape [H x W]

    Raises:
        NotImplementedError: ...

    Returns:
        ImagePIL : Visualization of shape [H x W]
    """
    raise NotImplementedError

class BoundingBoxVisualizer(Visualizer):
  def __init__(self):
    self.classes_to_color = {1: (0,255,0), 2: (255,0,0)}
    self.img_width = 1024 
    self.img_height = 1024

  def bbox_xywh(self, cx: float, cy:float, width: float, height: float) -> Tuple[int,int,int,int]:
    cx_rescaled = round(cx * self.img_width)
    cy_rescaled = round(cy * self.img_height)
    width_rescaled = int(round(width * self.img_width))
    height_rescaled = int(round(height * self.img_height))

    x_min = int(round(cx_rescaled - (width_rescaled / 2)))
    y_min = int(round(cy_rescaled - (height_rescaled / 2)))

    return x_min, y_min, width_rescaled, height_rescaled


  def visualize(self, image: np.ndarray, prediction: np.ndarray) -> Union[ImagePIL, None]:

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(1,1, figsize=(self.img_width*px*1.3, self.img_height*px*1.3))
    plt.axis('off')
    ax.imshow(image)
    
    n_preds = prediction.shape[0]
    for i in range(n_preds):
      # get the score
      score = prediction[i,-1]
      if score < 0.5:
        continue

      # get the categroy
      class_id = prediction[i,0]
      if class_id == 0:
        continue

      # get the bounding box and draw it
      cx, cy = prediction[i, 1], prediction[i, 2]
      width, height =  prediction[i, 3], prediction[i, 4]

      color = rgb_to_hex(random_rgb(i))
      #color = rgb_to_hex(self.classes_to_color[class_id])
      x_bbox, y_bbox, w_bbox, h_bbox = self.bbox_xywh(cx, cy, width, height)
    
      rect = patches.Rectangle((x_bbox, y_bbox), w_bbox, h_bbox, linewidth=4, edgecolor=color, facecolor='none')
      ax.add_patch(rect)
            
class SemanticSegmentationVisualizer(Visualizer):
  def __init__(self) -> None:
    self.classes_to_color = {0: (0,0,0), 1: (0,255,0), 2: (255,0,0)}
    self.classes_to_color[3] = self.classes_to_color[1]
    self.classes_to_color[4] = self.classes_to_color[2]

  def visualize(self, image: np.ndarray, prediction: np.ndarray) -> Union[ImagePIL, None]:
    canvas = np.zeros(image.shape, dtype=np.uint8)

    for class_id in np.unique(prediction):
      class_mask = prediction == class_id
      canvas[class_mask, :] = self.classes_to_color[class_id]

    vis = blend_images(image, canvas, alpha=0.6)

    background_mask = prediction == 0
    vis[background_mask, :] = image[background_mask, :]

    return Image.fromarray(vis)

class InstanceSegmentationVisualizer(Visualizer):
  def __init__(self):
    pass 

  def visualize(self, image: np.ndarray, prediction: np.ndarray) -> Union[ImagePIL, None]:
    canvas = np.zeros(image.shape, dtype=np.uint8)

    for instance_id in np.unique(prediction):
      if instance_id == 0:
        continue

      instance_mask = prediction == instance_id
      canvas[instance_mask, :] = random_rgb(seed=int(instance_id))
    
    vis = blend_images(image, canvas, alpha=0.25)

    background_mask = prediction == 0
    vis[background_mask, :] = image[background_mask, :]
    
    return Image.fromarray(vis)

class VisibilityVisualizer(Visualizer):
  def __init__(self):
    self.img_width = 1024 
    self.img_height = 1024

  def visualize(self, image: np.ndarray, prediction: np.ndarray) -> Union[ImagePIL, None]:

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(1,1, figsize=(self.img_width*px*1.3, self.img_height*px*1.3))
    plt.axis('off')

    alphas = prediction > 0

    ax.imshow(image)
    ax.imshow(prediction, alpha=alphas * 0.6)

class BoundingBoxFromMaskVisualizer():
  def __init__(self):
    self.classes_to_color = {1: (0,255,0), 2: (255,0,0)}
    self.classes_to_color[3] = self.classes_to_color[1]
    self.classes_to_color[4] = self.classes_to_color[2]

    self.img_width = 1024 
    self.img_height = 1024

  @staticmethod
  def bbox_from_mask_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    mask = torch.Tensor(mask)
    rows, cols = mask.nonzero(as_tuple=True)

    x_min = int(torch.min(cols))
    y_min = int(torch.min(rows))
    x_max = int(torch.max(cols))
    y_max = int(torch.max(rows))

    return x_min, y_min, x_max, y_max


  def visualize(self, image: np.ndarray, instance_mask: np.ndarray, semantics: np.ndarray) -> Union[ImagePIL, None]:

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(1,1, figsize=(self.img_width*px*1.3, self.img_height*px*1.3))
    plt.axis('off')
    ax.imshow(image)

    for instance_id in np.unique(instance_mask):
      if instance_id == 0:
        continue

      mask = instance_mask == instance_id
      class_id = np.unique(semantics[mask])
      if len(class_id) > 1:
        continue
      if class_id == 0:
        continue

      x_min, y_min, x_max, y_max = self.bbox_from_mask_xyxy(mask)
      #color = rgb_to_hex(self.classes_to_color[int(class_id)])
      color = rgb_to_hex(random_rgb(instance_id))

      rect = patches.Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min), linewidth=4, edgecolor=color, facecolor='none')
      ax.add_patch(rect)