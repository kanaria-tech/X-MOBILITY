# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import gin
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DepthAnythingForDepthEstimation, Dinov2Model
from PIL import Image
from torchvision import transforms as TF
from vggt.vggt.models.vggt import VGGT
from model.x_mobility.utils import pack_sequence_dim, unpack_sequence_dim

# Need to compute this dynamically and adapt to different image size.
DEPTH_ANYTHING_IMAGE_SIZE = [322, 518]

@gin.configurable
class VGGTEncoder(nn.Module):
    def __init__(self, out_channels_pose: int, out_channels_vggt: int):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pose_encoder = nn.Sequential(
            nn.Linear(9, out_channels_pose),
            nn.ReLU(True),
            nn.Linear(out_channels_pose, out_channels_pose),
            nn.ReLU(True),
        )
        self.vggt_encoder = nn.Sequential(
            nn.Linear(2048, out_channels_vggt),
            nn.ReLU(True),
            nn.Linear(out_channels_vggt, out_channels_vggt),
            nn.ReLU(True),
        )
        # Do not train the VGGT model
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.out_channels = [out_channels_pose, out_channels_vggt]

    def forward(self, images):
        images = self.load_vggt_images(images)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                pose_enc_list = self.model.camera_head(aggregated_tokens_list)
                pose = pose_enc_list[-1].squeeze()  # pose encoding of the last iteration
                pose_embedding = self.pose_encoder(pose)
                vggt_embeddings = self.vggt_encoder(torch.mean(aggregated_tokens_list[-1], dim=2).squeeze())

        return pose_embedding, vggt_embeddings


    def load_vggt_images(self, images_array: torch.Tensor):
        """
        A quick start function to load and preprocess images for model input.
        This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

        Args:
            images_array: torch.Tensor of image tensors

        Returns:
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

        Raises:
            ValueError: If the input list is empty

        Notes:
            - Images with different dimensions will be padded with white (value=1.0)
            - A warning is printed when images have different shapes
            - The function ensures width=518px while maintaining aspect ratio
            - Height is adjusted to be divisible by 14 for compatibility with model requirements
        """
        # Check for empty list
        if len(images_array) == 0:
            raise ValueError("At least 1 image is required")

        images = []
        shapes = set()
        # images_array  = images_array.squeeze()
        B, S, C_in, height, width = images_array.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images_packed = pack_sequence_dim(images_array)
        no_of_images = B * S
        # First process all images and collect their shapes
        for iter_image in range(no_of_images):
            img = images_packed[iter_image]
            new_width = 518
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14
            # Resize with new dimensions (width, height)
            img = img.unsqueeze(0)
            img = F.interpolate(img, size=(new_height, new_width), mode='bicubic')  # Downsample to 32x32
            img = img.squeeze(0)
            # Center crop height if it's larger than 518
            if new_height > 518:
                start_y = (new_height - 518) // 2
                img = img[:, start_y: start_y + 518, :]

            shapes.add((img.shape[1], img.shape[2]))
            images.append(img)

        # Check if we have different shapes
        # In theory our model can also work well with different shapes

        if len(shapes) > 1:
            print(f"Warning: Found images with different shapes: {shapes}")
            # Find maximum dimensions
            max_height = max(shape[0] for shape in shapes)
            max_width = max(shape[1] for shape in shapes)

            # Pad images if necessary
            padded_images = []
            for img in images:
                h_padding = max_height - img.shape[1]
                w_padding = max_width - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )
                padded_images.append(img)
            images = padded_images

        images = torch.stack(images)  # concatenate images

        # Ensure correct shape when single image
        if B == 1:
            # Verify shape is (1, C, H, W)
            if images.dim() == 3:
                images = images.unsqueeze(0)

        return images


@gin.configurable
class SpeedEncoder(nn.Module):
    '''Encoder of robot's speed

        Args:
            out_channels (int): output channels size
            speed_normalisation (float): speed normialisation factor
    '''
    def __init__(self, out_channels: int, speed_normalisation: float):
        super().__init__()
        self.speed_encoder = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(True),
        )
        self.out_channels = out_channels
        self.speed_normalisation = speed_normalisation

    def forward(self, speed: torch.Tensor) -> torch.Tensor:
        return self.speed_encoder(speed / self.speed_normalisation)


@gin.configurable
class ImageDINOEncoder(nn.Module):
    ''' Image encoding with DINO v2
    '''
    def __init__(self, enable_fine_tune=False):
        super().__init__()
        self.dino_model = Dinov2Model.from_pretrained('facebook/dinov2-small',
                                                      output_attentions=True)
        # Freeze the pre-trained dino model.
        if not enable_fine_tune:
            for param in self.dino_model.parameters():
                param.requires_grad = False
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_channels = self.dino_model.config.hidden_size * 2

    def forward(self, image):
        # Preprocess image.
        dino_image_size = self.dino_model.config.image_size
        processed_image = F.interpolate(
            image,
            size=[dino_image_size, dino_image_size],
            mode='bicubic',
            align_corners=False)
        processed_image = (processed_image - self.mean) / self.std

        # Encode image.
        outputs = self.dino_model(processed_image)

        # Extract features.
        last_hidden_states = outputs[0]
        cls_token = last_hidden_states[:, 0]
        patch_tokens = last_hidden_states[:, 1:]
        features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        # Extract attentions
        n, _, h, w = processed_image.shape
        avg_attension = torch.mean(outputs.attentions[-1], dim=1)
        cls_attention = avg_attension[:, 0, 1:].view(n, h // 14, w // 14)

        return {'image_features': features, 'image_attentions': cls_attention}


@gin.configurable
class ImageDepthAnythingEncoder(nn.Module):
    ''' Image encoding with Depth Anything.
    '''
    def __init__(self, enable_fine_tune=False):
        super().__init__()
        self.enable_fine_tune = enable_fine_tune
        self.depth_model = DepthAnythingForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-small-hf",
            output_hidden_states=True,
            output_attentions=True)
        if not enable_fine_tune:
            for param in self.depth_model.parameters():
                param.requires_grad = False
        else:
            self.depth_gt_model = DepthAnythingForDepthEstimation.from_pretrained(
                "LiheYoung/depth-anything-small-hf",
                output_hidden_states=False,
                output_attentions=False)
            for param in self.depth_gt_model.parameters():
                param.requires_grad = False
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_channels = self.depth_model.backbone.config.hidden_size * 2

    def forward(self, image):
        # Preprocess image.
        processed_image = F.interpolate(image,
                                        size=DEPTH_ANYTHING_IMAGE_SIZE,
                                        mode='bicubic',
                                        align_corners=False)
        processed_image = (processed_image - self.mean) / self.std

        # Encode image.
        outputs = self.depth_model(processed_image)

        # Extract features.
        last_hidden_states = outputs.hidden_states[-1]
        cls_token = last_hidden_states[:, 0]
        patch_tokens = last_hidden_states[:, 1:]
        features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        # Extract attentions
        n, _, h, w = processed_image.shape
        avg_attension = torch.mean(outputs.attentions[-1], dim=1)
        cls_attention = avg_attension[:, 0, 1:].view(n, h // 14, w // 14)

        ret_dict = {
            'image_features': features,
            'image_attentions': cls_attention,
            'depth': outputs.predicted_depth,
        }

        # Get depth GT based on the pretrained DepthAnything model.
        if self.enable_fine_tune:
            depth_gt = self.depth_gt_model(processed_image).predicted_depth
            ret_dict['depth_gt'] = depth_gt

        return ret_dict


@gin.configurable
class ObservationEncoder(nn.Module):
    '''Encoder of the input observations

        Inputs:
            batch (Dict): dict of the input tensors:
                image: (b, s, 3, h, w)
                speed: (b, s, 1)
                action: (b, s, 6)

        Returns:
            embedding: 1D embedding of the observations
    '''
    def __init__(self, image_encoder: nn.Module, use_vggt: bool):
        super().__init__()

        self.use_vggt = use_vggt

        # Image
        self.image_encoder = image_encoder()
        features_channels = self.image_encoder.out_channels

        # Speed.
        self.speed_encoder = SpeedEncoder()
        features_channels += self.speed_encoder.out_channels

        if self.use_vggt:
            self.vggt_encoder = VGGTEncoder()
            features_channels += self.vggt_encoder.out_channels[0] + self.vggt_encoder.out_channels[1]

        self.embedding_dim = features_channels

    def forward(self, batch: Dict) -> torch.Tensor:
        b, s = batch['image'].shape[:2]
        image = pack_sequence_dim(batch['image'])
        speed = pack_sequence_dim(batch['speed'])

        # Image encoding
        image_encoding_outputs = self.image_encoder(image)

        # Speed encoding
        speed_features = self.speed_encoder(speed)

        if self.use_vggt:
            # VGGT encoding
            pose_features, vggt_features = self.vggt_encoder(batch['image'])

            # Final observation embedding.
            embedding = torch.cat(
                [image_encoding_outputs['image_features'], speed_features, pose_features, vggt_features], dim=1)
        else:
            # Final observation embedding.
            embedding = torch.cat(
                [image_encoding_outputs['image_features'], speed_features], dim=1)

        # Compose outputs.
        outputs = {}
        outputs['embedding'] = unpack_sequence_dim(embedding, b, s)
        outputs['speed_features'] = unpack_sequence_dim(speed_features, b, s)
        for k, v in image_encoding_outputs.items():
            outputs[k] = unpack_sequence_dim(v, b, s)
        return outputs
