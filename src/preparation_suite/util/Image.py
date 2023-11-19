from collections.abc import Iterator
from mmap import ACCESS_READ, mmap
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class BaseTransform:
    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
        raise NotImplementedError


class FillTransparent(BaseTransform):
    def __init__(self, color: int = 0) -> None:
        self.color = color

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
        if image.shape[2] != 4:
            return None

        return self._fill_transparent(image, self.color)

    @staticmethod
    def _fill_transparent(image: NDArray[np.uint8], color: int) -> NDArray[np.uint8]:
        trans_mask = image[:, :, 3] == 0
        image[trans_mask, :3] = color
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


class PadToSquare(BaseTransform):
    def __init__(self, color: int | tuple[int, int, int] = 0) -> None:
        if isinstance(color, int):
            color = (color, color, color)
        self.color = color

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return self._pad(image, self.color)

    @staticmethod
    def _pad(image: NDArray[np.uint8], color: tuple[int, int, int]) -> NDArray[np.uint8]:
        h, w = image.shape[:2]
        diff = abs(h - w)
        half_diff = diff // 2

        left = (h > w) * half_diff
        right = (h > w) * (diff - half_diff)

        top = (h <= w) * half_diff
        bottom = (h <= w) * (diff - half_diff)

        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


class QualityResize(BaseTransform):
    def __init__(self, target_size: tuple[int, int]) -> None:
        self.target_size = target_size

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = image.shape[:2]
        mag = max(self.target_size[0] / h, self.target_size[1] / w)

        if mag.is_integer():
            interpolation = cv2.INTER_CUBIC if mag > 1 else cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4 if mag > 1 else cv2.INTER_AREA

        return cv2.resize(
            image,
            (self.target_size[1], self.target_size[0]),
            interpolation=interpolation,
        )


class Resize(BaseTransform):
    def __init__(self, target_size: int | tuple[int, int]) -> None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        self.target_size = target_size

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = image.shape[:2]
        mag = max(self.target_size[0] / h, self.target_size[1] / w)

        return cv2.resize(
            image,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_CUBIC if mag > 1 else cv2.INTER_AREA,
        )


class AspectRatioResize(Resize):
    def __init__(self, short: int | None = None, long: int | None = None) -> None:
        if short is None and long is None:
            msg = "Either short or long must be specified."
            raise ValueError(msg)

        self.short = short
        self.long = long

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = image.shape[:2]
        mag = max(self.target_size[0] / h, self.target_size[1] / w)

        return cv2.resize(
            image,
            (int(w * mag), int(h * mag)),
            interpolation=cv2.INTER_CUBIC if mag > 1 else cv2.INTER_AREA,
        )


class ClipAtMinimumBorder:
    def __init__(self, full_scale: bool = False, repeat: int = 2) -> None:
        self.full_scale = full_scale
        self.repeat = repeat

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        corners = self._extract_corners(image)
        for i in range(self.repeat):
            background = corners[i % len(corners)]
            image = self._clip_image_based_on_background(image, background)
        return image

    @staticmethod
    def _extract_corners(image: NDArray[np.uint8]) -> list[NDArray[np.uint8]]:
        """Extracts the four corner colors of the image."""
        return [image[0, 0], image[-1, -1], image[0, -1], image[-1, 0]]

    def _clip_image_based_on_background(
        self,
        image: NDArray[np.uint8],
        background: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Clip the image by creating a mask that excludes the background color."""
        if self.full_scale:
            return self._clip_at_full_scale(image, background)
        else:
            return self._clip_with_resized_mask(image, background)

    @staticmethod
    def _clip_at_full_scale(image: NDArray[np.uint8], background: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Create a mask at the full scale of the image and clip the image using it."""
        mask = np.logical_not(np.all(image == background, axis=-1))
        return image[np.ix_(mask.any(1), mask.any(0))]

    @staticmethod
    def _clip_with_resized_mask(image: NDArray[np.uint8], background: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Create a mask at a smaller scale and then resize it to clip the image."""
        scale_factor = min(image.shape[:2]) // 32
        scale = max(scale_factor, 16)
        reduced_image = cv2.resize(
            image,
            (image.shape[1] // scale, image.shape[0] // scale),
            interpolation=cv2.INTER_AREA,
        )

        mask = np.all(reduced_image == background, axis=-1)
        mask = np.logical_not(mask)

        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return image[np.ix_(resized_mask.any(1), resized_mask.any(0))]


class HWC2CHW(BaseTransform):
    def __call__(self, image: NDArray[np.uint8]) -> None:
        image[:] = image.transpose(2, 0, 1)


class CHW2HWC(BaseTransform):
    def __call__(self, image: NDArray[np.uint8]) -> None:
        image[:] = image.transpose(1, 2, 0)


class ChannelSwap(BaseTransform):
    def __call__(self, image: NDArray[np.uint8]) -> None:
        image[:] = image[:, :, ::-1]


class Compose:
    def __init__(self, transforms: list[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
        for transform in self.transforms:
            tmp = transform(image)
            if tmp is not None:
                image = tmp
        return image


class LoadImageWrapper:
    def __init__(self, compose: Compose) -> None:
        self.compose = compose

    def __call__(self, image: str | NDArray[np.uint8]) -> NDArray[np.uint8] | None:
        if not isinstance(image, np.ndarray):
            with open(image, "rb") as f, mmap(f.fileno(), 0, access=ACCESS_READ) as mapped_file:
                image = cv2.imdecode(np.frombuffer(mapped_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            assert image is not None, f"Failed to read image: {image}"

            if image.dtype == "uint16":
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            if image.ndim < 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return self.compose(image)


class GetSupportImages:
    _support_extensions: list[str]

    def __init__(self) -> None:
        self._support_extensions = [".png", ".jpg", ".jpeg", ".webp"]

    def __call__(self, paths: Path | list[Path] | Iterator[Path]) -> list[Path]:
        if paths.is_dir():
            return self._check_list(paths.iterdir())
        elif paths.is_file():
            return self._check_list([paths])
        else:
            return self._check_list(paths)

    def _check(self, path: Path) -> bool:
        if path.suffix in self._support_extensions:
            return True
        if cv2.haveImageReader(str(path)):
            self._support_extensions.append(path.suffix)
            return True
        return False

    def _check_list(self, paths: list[Path] | Iterator[Path]) -> list[Path]:
        check_passed = []
        for path in paths:
            if path.is_file():
                if self._check(path):
                    check_passed.append(path)
            elif path.is_dir():
                check_passed.extend(self._check_list(path.iterdir()))

        return sorted(check_passed)
