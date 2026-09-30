"""Convert an RGB image to greyscale on the CPU or an AMD GPU."""

import argparse
import logging
from pathlib import Path

import numpy as np
from numba import hip
from numpy.typing import NDArray
from PIL import Image

hip.pose_as_cuda()
from numba import cuda, int32  # noqa: E402  (HIP must be configured before this import)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMG = Path(__file__).with_name("IMG_20260603_174154573.jpg")


class Greyscaler:
    def __init__(self, device: str = "cpu", image_path: str | Path = IMG):
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        self.device = device
        with Image.open(image_path) as image:
            self.image = image.convert("RGB")
        self.image_arr = np.ascontiguousarray(self.image)

    @staticmethod
    def grayscale_cpu(img: NDArray, dst: NDArray):
        for i in range(img.shape[0]):
            dst[i] = (int(img[i, 0]) + int(img[i, 1]) + int(img[i, 2])) // 3

    @staticmethod
    @cuda.jit
    def grayscale_gpu(img: NDArray, dst: NDArray):
        i = cuda.grid(1)
        if i < img.shape[0]:
            dst[i] = (int32(img[i, 0]) + int32(img[i, 1]) + int32(img[i, 2])) // 3

    def run(self, block_size: int = 256) -> NDArray:
        height, width, _ = self.image_arr.shape
        pixels = self.image_arr.reshape(-1, 3)
        output = np.empty(height * width, dtype=np.uint8)

        if self.device == "cpu":
            self.grayscale_cpu(pixels, output)
        else:
            if not cuda.is_available():
                raise RuntimeError("no HIP-compatible GPU is available")
            max_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK
            if not 1 <= block_size <= max_threads:
                raise ValueError(
                    f"block size must be between 1 and {max_threads} threads "
                    "for this GPU"
                )
            device_pixels = cuda.to_device(pixels)
            device_output = cuda.device_array(output.shape, dtype=output.dtype)
            threads_per_block = block_size
            blocks = (len(output) + threads_per_block - 1) // threads_per_block
            logger.info(f"Executing Greyscaler with {blocks} blocks!")
            self.grayscale_gpu[blocks, threads_per_block](device_pixels, device_output)
            device_output.copy_to_host(output)

        return output.reshape(height, width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an image to greyscale.")
    parser.add_argument("image", nargs="?", type=Path, default=IMG)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("lab3/greyscale.png"))
    args = parser.parse_args()

    result = Greyscaler(args.device, args.image).run(args.block_size)
    Image.fromarray(result).save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
