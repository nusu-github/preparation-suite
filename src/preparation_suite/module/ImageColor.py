from pathlib import Path

import cv2
import numpy as np

from preparation_suite.module.Processor import Processor
from preparation_suite.util.Image import (
    Compose,
    FillTransparent,
    LoadImageWrapper,
    Resize,
)


class ImageColor(Processor):
    _processor: LoadImageWrapper = LoadImageWrapper(Compose([FillTransparent(0), Resize(64)]))

    def __init__(self) -> None:
        super().__init__()

    def _process_images(self, items: list[Path], max_workers: int | None = None):
        def _process_image(path: str):
            image = ImageColor._processor(path)
            return not np.allclose(
                image,
                cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
                rtol=0.15,
                equal_nan=True,
            )

        return self._map(_process_image, items, max_workers=max_workers)

    def process(
        self,
        image: Path,
    ) -> bool:
        if isinstance(image, Path):
            image = [image]

        return self.process_batch(image)[0]

    def process_batch(
        self,
        images: list[Path],
        max_workers: int | None = None,
    ) -> list[bool]:
        return self._process_images(images, max_workers)
