from abc import ABC, abstractmethod
from typing import Literal, Tuple

import numpy as np
import PIL.Image as PIL
import torch
import torch.nn as nn
from PIL import ImageFilter
from PIL.Image import BICUBIC, Image, composite
from PIL.ImageOps import invert
from torch import Tensor


class DATModel(ABC):
    @abstractmethod
    def __init__(self, pth_path: str, scale: Literal[2, 3, 4]) -> None:
        """
        Initializes the instance with the specified PyTorch model weights from the given path,
        intended for image upscaling.

        Args:
            pth_path (str): Path to the PyTorch model weights file.
            scale (Literal[2, 3, 4]): The scaling factor to be used for upscaling images.

        Note:
            This constructor assumes that the model's architecture and parameters match
            the configuration required by the application. It directly loads the parameters
            into the model's state dictionary. Make sure that the model architecture and the
            saved state dictionary match, or else unexpected behavior may occur.
        """
        self.scale = scale
        self._model.load_state_dict(
            torch.load(pth_path)["params"]
        )  # Raises error in case of incorrect weights

    def upscale(self, image: Image) -> Image:
        """
        Upscales the given input image using the initialized PyTorch model.

        Args:
            image (Image): An instance of PIL Image representing the input image.

        Returns:
            Image: An instance of PIL Image representing the upscaled image.

        Note:
            Internally converts the input image to RGBA mode for processing.
            Extracts the RGB channels and upscales them using the initialized model.
            Resizes the alpha channel separately using bicubic interpolation.
            Combines the upscaled RGB image with the resized alpha channel
            and converts the result back to the original image mode.

            Note that if the input image mode is not compatible with RGBA conversion,
            the method may not perform as expected. Ensure that input images are in modes
            supporting conversion to and from RGBA, such as RGB, L (grayscale), or RGBA modes.

        Example:
            >>> upscale_model: DATModel = ...
            >>> input_image = Image.open("input_image.jpg")
            >>> upscaled_image = upscale_model.upscale(input_image)
        """

        def extract_rgb_and_alpha(image: Image) -> Tuple[Image, Image]:
            rgba_image = image.convert("RGBA")
            return rgba_image.convert("RGB"), rgba_image.split()[-1]

        def rgb2tensor(image: Image) -> Tensor:
            input = (
                torch.tensor(np.array(image), dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255
            )
            return input

        def tensor2rgb(tensor: Tensor) -> Image:
            tensor = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0) * 255
            tensor = tensor.cpu()
            output = PIL.fromarray(tensor.byte().numpy())
            return output

        def merge_rgb_and_alpha(rgb: Image, alpha: Image) -> Image:
            # Merge rgb and alpha
            output = rgb.copy()
            output.putalpha(alpha)

            # Denoise on the alpha edges
            edges = alpha.filter(ImageFilter.FIND_EDGES)
            edges = invert(edges)
            edges = edges.filter(
                ImageFilter.MinFilter(7)
            )  # 7 because best experimental thickness
            output_denoised = output.filter(
                ImageFilter.MedianFilter(5)
            )  # 5 because best experimental correctness
            output = composite(output, output_denoised, edges)
            return output

        source_mode = image.mode
        rgb, alpha = extract_rgb_and_alpha(image)
        tensor = rgb2tensor(rgb)
        with torch.no_grad():
            tensor = self._model(tensor)
        rgb = tensor2rgb(tensor)
        alpha = alpha.resize(rgb.size, BICUBIC)
        image = merge_rgb_and_alpha(rgb, alpha)
        image = image.convert(source_mode)
        return image

    @property
    @abstractmethod
    def _model(self) -> nn.Module:
        pass
