import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class RGBDAugmentor:
    """Apply geometry and sequence color augmentation."""

    def __init__(
        self,
        crop_size,
        aug_crop=16,
        seq_aug_crop=False,
        color_jitter=True,
    ):
        self.crop_size = tuple(crop_size)
        self.aug_crop = aug_crop
        self.seq_aug_crop = seq_aug_crop
        self.color_jitter = color_jitter

    @staticmethod
    def _principal_point_crop(image, depth, depth_valid, intrinsics, frame):
        width, height = image.size
        center_x, center_y = np.round(intrinsics[2:4]).astype(int)
        margin_x = min(center_x, width - center_x)
        margin_y = min(center_y, height - center_y)
        assert margin_x > width / 5, f"bad principal point in frame {frame}"
        assert margin_y > height / 5, f"bad principal point in frame {frame}"

        left = center_x - margin_x
        top = center_y - margin_y
        right = center_x + margin_x
        bottom = center_y + margin_y

        image = image.crop((left, top, right, bottom))
        depth = depth[top:bottom, left:right]
        depth_valid = depth_valid[top:bottom, left:right]
        intrinsics = intrinsics.copy()
        intrinsics[2] -= left
        intrinsics[3] -= top
        return image, depth, depth_valid, intrinsics

    @staticmethod
    def _rescale(image, depth, depth_valid, intrinsics, target_size):
        input_width, input_height = image.size
        target_height, target_width = target_size
        scale = max(
            target_width / input_width,
            target_height / input_height,
        ) + 1e-8
        output_width = int(np.floor(input_width * scale))
        output_height = int(np.floor(input_height * scale))
        assert output_width >= target_width
        assert output_height >= target_height
        output_size = (output_width, output_height)

        interpolation = (
            Image.Resampling.LANCZOS
            if scale < 1
            else Image.Resampling.BICUBIC
        )
        image = image.resize(output_size, resample=interpolation)
        depth = cv2.resize(depth, output_size, interpolation=cv2.INTER_NEAREST)
        depth_valid = cv2.resize(
            depth_valid.astype(np.uint8),
            output_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        offset_x = (input_width * scale - output_width) / 2
        offset_y = (input_height * scale - output_height) / 2
        intrinsics = intrinsics.copy()
        intrinsics[:2] *= scale
        intrinsics[2] = (intrinsics[2] + 0.5) * scale - offset_x - 0.5
        intrinsics[3] = (intrinsics[3] + 0.5) * scale - offset_y - 0.5
        return image, depth, depth_valid, intrinsics

    @staticmethod
    def _center_crop(image, depth, depth_valid, intrinsics, crop_size):
        crop_height, crop_width = crop_size
        width, height = image.size
        assert width >= crop_width and height >= crop_height
        left = int(np.round((width - crop_width) / 2))
        top = int(np.round((height - crop_height) / 2))
        right = left + crop_width
        bottom = top + crop_height

        image = image.crop((left, top, right, bottom))
        depth = depth[top:bottom, left:right]
        depth_valid = depth_valid[top:bottom, left:right]
        intrinsics = intrinsics.copy()
        intrinsics[2] -= left
        intrinsics[3] -= top
        return image, depth, depth_valid, intrinsics

    def _spatial_transform_view(
        self,
        image,
        depth,
        depth_valid,
        intrinsics,
        frame,
        crop_delta,
    ):
        image = Image.fromarray(image[..., ::-1])
        image, depth, depth_valid, intrinsics = self._principal_point_crop(
            image,
            depth,
            depth_valid,
            intrinsics,
            frame,
        )

        if crop_delta is None:
            crop_delta = (
                np.random.randint(0, self.aug_crop)
                if self.aug_crop > 1
                else 0
            )
        resize_target = (
            self.crop_size[0] + crop_delta,
            self.crop_size[1] + crop_delta,
        )
        image, depth, depth_valid, intrinsics = self._rescale(
            image,
            depth,
            depth_valid,
            intrinsics,
            resize_target,
        )
        return self._center_crop(
            image,
            depth,
            depth_valid,
            intrinsics,
            self.crop_size,
        )

    @staticmethod
    def _sequence_color_jitter(images):
        transform_order = torch.randperm(4).tolist()
        brightness = float(torch.empty(1).uniform_(0.5, 1.5))
        contrast = float(torch.empty(1).uniform_(0.5, 1.5))
        saturation = float(torch.empty(1).uniform_(0.5, 1.5))
        hue = float(torch.empty(1).uniform_(-0.1, 0.1))

        transformed = []
        for image in images:
            for transform in transform_order:
                if transform == 0:
                    image = TF.adjust_brightness(image, brightness)
                elif transform == 1:
                    image = TF.adjust_contrast(image, contrast)
                elif transform == 2:
                    image = TF.adjust_saturation(image, saturation)
                else:
                    image = TF.adjust_hue(image, hue)
            transformed.append(image)
        return transformed

    def __call__(self, images, depths, depths_valid, intrinsics):
        crop_delta = None
        if self.seq_aug_crop and self.aug_crop > 1:
            crop_delta = np.random.randint(0, self.aug_crop)

        output_images = []
        output_depths = []
        output_depths_valid = []
        output_intrinsics = []
        for frame in range(len(images)):
            image, depth, depth_valid, intrinsic = self._spatial_transform_view(
                images[frame],
                depths[frame],
                depths_valid[frame],
                intrinsics[frame],
                frame,
                crop_delta,
            )
            output_images.append(image)
            output_depths.append(depth)
            output_depths_valid.append(depth_valid)
            output_intrinsics.append(intrinsic)

        if self.color_jitter:
            output_images = self._sequence_color_jitter(output_images)
        output_images = np.stack(
            [np.asarray(image)[..., ::-1] for image in output_images]
        ).copy()
        return (
            output_images,
            np.stack(output_depths),
            np.stack(output_depths_valid),
            np.stack(output_intrinsics),
        )
