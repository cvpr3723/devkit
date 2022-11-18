import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from tqdm import tqdm

from diskcache import FanoutCache
cache = FanoutCache(
    directory="/logs/cache",
    shards=64,
    timeout=1,
    size_limit=3e12,
)

class PlantsBase(Dataset):
    def __init__(self, data_path, overfit=False):
        super().__init__()

        self.data_path = data_path
        self.overfit = overfit

        self.image_list = [
            x for x in os.listdir(os.path.join(self.data_path, "images")) if ".png" in x
        ]

        self.image_list.sort()

        self.len = len(self.image_list)

        # preload the data to memory
        self.field_list = os.listdir(self.data_path)
        self.data_frame = self.load_data(
            self.data_path, self.image_list, self.field_list
        )
        self.data_frame["image_list"] = self.image_list

    @staticmethod
    @cache.memoize(typed=True)
    def load_data(data_path, image_list, field_list):
        data_frame = {}
        for field in tqdm(field_list):
            data_frame[field] = []
            for image in tqdm(image_list):
                image = cv2.imread(
                    os.path.join(os.path.join(data_path, field), image),
                    cv2.IMREAD_UNCHANGED,
                )
                if len(image.shape) > 2:
                    sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    sample = torch.tensor(sample).permute(2, 0, 1)
                else:
                    sample = torch.tensor(image.astype("int16"))

                data_frame[field].append(sample)
        return data_frame

    def get_sample(self, index):
        sample = {}

        for field in self.field_list:
            sample[field] = self.data_frame[field][index]

        partial_crops = sample["semantics"] == 3
        partial_weeds = sample["semantics"] == 4

        # 1 where there's stuff to be ignored by instance segmentation, 0 elsewhere
        sample["ignore_mask"] = torch.logical_or(partial_crops, partial_weeds).bool()

        # remove partial plants
        sample["semantics"][partial_crops] = 1
        sample["semantics"][partial_weeds] = 2

        # remove instances that aren't crops or weeds
        sample["plant_instances"][sample["semantics"] == 0] = 0
        sample["leaf_instances"][sample["semantics"] == 0] = 0

        # make ids successive
        sample["plant_instances"] = torch.unique(
            sample["plant_instances"] + sample["semantics"] * 1e6, return_inverse=True
        )[1]
        sample["leaf_instances"] = torch.unique(
            sample["leaf_instances"] + sample["semantics"] * 1e6, return_inverse=True
        )[1]
        sample["leaf_instances"][sample["semantics"] == 2] = 0
        
        sample["image_name"] = self.data_frame["image_list"][index]

        return sample

    def __len__(self):
        return self.len
