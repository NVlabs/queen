# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import glob
import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms


class MultiViewVideoDataset(Dataset):
    def __init__(
        self,
        datadir,
        split,
        test_indices,
        max_frames=300,
        start_idx=0,
        img_format='png',
        verbose=False,
    ):
        videos = glob.glob(os.path.join(datadir, "cam*"))
        videos = sorted(videos)
        self.n_cams = 0
        self.n_frames = None
        self.image_transform = transforms.ToTensor()
        self.image_paths = []
        # For Google Immersive some cameras might not exist in models.json, remove them
        if os.path.exists(os.path.join(datadir,"models.json")):
            import json 
            with open(os.path.join(datadir, "models.json"), "r") as f:
                meta = json.load(f)
            camera_names_meta = [camera['name'] for camera in meta]
            camera_names_images = sorted([os.path.basename(video) for video in videos])
            filtered_videos = []
            for i,camera_name_image in enumerate(camera_names_images):
                if camera_name_image in camera_names_meta:
                    filtered_videos.append(os.path.join(datadir, camera_name_image))
            videos = sorted(filtered_videos)
    
        if verbose:
            print(f"MultiViewVideoDataset::__init__(): parsed {len(videos)} videos (sequences)")

        for idx, video in enumerate(videos):
            if (idx in test_indices and split == 'test') or \
                (idx not in test_indices and split=='train'):
                self.n_cams += 1
                if os.path.exists(os.path.join(datadir,"models.json")):
                    image_list = sorted(glob.glob(os.path.join(video,'images_scaled_2','*.png')))
                else:
                    image_list = sorted(glob.glob(os.path.join(video,'images','*.png')))
                image_list = image_list[start_idx:start_idx+max_frames]
                if verbose and idx == 0:
                    print(f"MultiViewVideoDataset::__init__(): View {idx}: by start_idx ({start_idx}) and max_frames({max_frames}), selected image_list = {image_list}")
                self.image_paths += image_list
                if verbose:
                    print(f"MultiViewVideoDataset::__init__(): added {len(image_list)} image paths from: {video}")

                if self.n_frames is None:
                    self.n_frames = len(image_list)
                else:
                    assert self.n_frames == len(image_list), f"{video} contains"+\
                         f" {len(image_list)} frames but should contain {self.n_frames} frames."
        if verbose:
            print(f"MultiViewVideoDataset::__init__(): total parsed image paths: {len(self.image_paths)}")

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.image_transform(img)
        return img, self.image_paths[index]
    
    def __len__(self):
        return len(self.image_paths)
    
class SequentialMultiviewSampler(Sampler):
    
    def __init__(self, dataset: MultiViewVideoDataset) -> None:
        super().__init__(dataset)
        self.n_cams, self.n_frames = dataset.n_cams, dataset.n_frames

    def __iter__(self):
        rearrange = []
        for frame_idx in range(self.n_frames):
            rearrange += list(range(frame_idx, self.n_cams*self.n_frames, self.n_frames))
        return iter(rearrange) 
    
    def __len__(self):
        return self.n_cams*self.n_frames
    

class IndexedMultiviewSampler(Sampler):
    
    def __init__(self, dataset: MultiViewVideoDataset, frame_idx: int) -> None:
        super().__init__(dataset)
        self.n_cams, self.n_frames = dataset.n_cams, dataset.n_frames
        self.frame_idx = frame_idx

    def __iter__(self):
        return iter(list(range(self.frame_idx, self.n_cams*self.n_frames, self.n_frames)))
    
    def __len__(self):
        return self.n_cams