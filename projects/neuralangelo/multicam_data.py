import json
import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
from PIL import Image, ImageFile

from projects.nerf.datasets import base
from projects.nerf.utils import camera
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(base.Dataset):

    def __init__(self, cfg, is_inference=False):
        super().__init__(cfg, is_inference=is_inference, is_test=False)
        cfg_data = cfg.data
        self.root = cfg_data.root
        self.preload = cfg_data.preload
        self.H, self.W = cfg_data.val.image_size if is_inference else cfg_data.train.image_size
        meta_fname = f"{cfg_data.root}/transforms.json"
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.canonical = getattr(cfg_data, "canonical", False)
        self.target_frame = getattr(cfg_data, "target_frame", None)

        self.list = self.meta["frames"]

        if self.canonical:
            print("Using canonical dataset")
            self.list = [frame for frame in self.list if frame["time_step"] == 1]
        else:
            self.list = [frame for frame in self.list if frame["time_step"] == self.target_frame]
            
    
        if cfg_data[self.split].subset:
            subset = cfg_data[self.split].subset
            # Initialize a Random generator with a fixed seed to get the same evaluation cameras every time.
            rng = random.Random(0)
            self.list = rng.sample(self.list, subset)
        self.num_rays = cfg.model.render.rand_rays
        self.readjust = getattr(cfg_data, "readjust", None)
        # Preload dataset if possible.
        if cfg_data.preload:
            self.images = self.preload_threading(self.get_image, cfg_data.num_workers)
            self.cameras = self.preload_threading(self.get_camera, cfg_data.num_workers, data_str="cameras")
            self.masks = self.preload_threading(self.get_mask, cfg_data.num_workers)

    def preprocess_mask(self, mask):
        # Resize the mask.
        mask = mask.resize((self.W, self.H))
        mask = torchvision_F.to_tensor(mask)
        return mask

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 image (R tensor): Image idx for per-image embedding.
                 image (Rx3 tensor): Image with pixel values in [0,1] for supervision.
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
        """
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        # Get the images.
        image, image_size_raw = self.images[idx] if self.preload else self.get_image(idx)
        mask, _ = self.masks[idx] if self.preload else self.get_mask(idx)
        image = self.preprocess_image(image)
        mask = self.preprocess_mask(mask)
        # Get the cameras (intrinsics and pose).
        intr, pose = self.cameras[idx] if self.preload else self.get_camera(idx)
        intr, pose = self.preprocess_camera(intr, pose, image_size_raw)
        time_step = self.list[idx]['time_step']
        # Pre-sample ray indices.
        if self.split == "train":
            ray_idx = torch.randperm(self.H * self.W)[:self.num_rays]  # [R]
            image_sampled = image.flatten(1, 2)[:, ray_idx].t()  # [R,3]
            mask_sampled = mask.flatten(1, 2)[:, ray_idx].t()  # [R,1]
            sample.update(
                ray_idx=ray_idx,
                image_sampled=image_sampled,
                mask_sampled=mask_sampled,
                intr=intr,
                pose=pose,
                time_step=time_step
            )
        else:  # keep image during inference
            sample.update(
                image=image,
                mask=mask,
                intr=intr,
                pose=pose,
                time_step=time_step
            )
        return sample

    def get_mask(self, idx):
        fpath = self.list[idx]["file_path"].replace("images", "masks").replace("rgb", "mask")
        mask_fname = f"{self.root}/{fpath}"
        mask = Image.open(mask_fname)
        mask.load()
        return mask, mask.size

    def get_image(self, idx):
        fpath = self.list[idx]["file_path"]
        image_fname = f"{self.root}/{fpath}"
        image = Image.open(image_fname)
        image.load()
        image_size_raw = image.size
        return image, image_size_raw

    def preprocess_image(self, image):
        # Resize the image.
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        rgb = image[:3]
        return rgb

    def get_camera(self, idx):
        # Camera intrinsics.
        intr = torch.tensor([[self.list[idx]["fx"], 0, self.list[idx]["cx"]],
                             [0, self.list[idx]["fy"], self.list[idx]["py"]],
                             [0, 0, 1]]).float()
        # Camera pose.
        c2w_gl = torch.tensor(self.list[idx]["transform_matrix"], dtype=torch.float32)
        c2w = self._gl_to_cv(c2w_gl)
        # center scene
        center = np.array(self.meta["sphere_center"])
        center += np.array(getattr(self.readjust, "center", [0])) if self.readjust else 0.
        c2w[:3, -1] -= center
        # scale scene
        scale = np.array(self.meta["sphere_radius"])
        scale *= getattr(self.readjust, "scale", 1.) if self.readjust else 1.
        c2w[:3, -1] /= scale
        w2c = camera.Pose().invert(c2w[:3])
        return intr, w2c

    def preprocess_camera(self, intr, pose, image_size_raw):
        # Adjust the intrinsics according to the resized image.
        intr = intr.clone()
        raw_W, raw_H = image_size_raw
        intr[0] *= self.W / raw_W
        intr[1] *= self.H / raw_H
        return intr, pose

    def _gl_to_cv(self, gl):
        # convert to CV convention used in Imaginaire
        cv = gl * torch.tensor([1, -1, -1, 1])
        return cv