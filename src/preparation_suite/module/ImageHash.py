from pathlib import Path

import cv2

from preparation_suite.module.Processor import Processor
from preparation_suite.util.Image import Compose, FillTransparent, LoadImageWrapper, PadToSquare, Resize


class ImageHash(Processor):
    def __init__(self) -> None:
        super().__init__()
        self._processor: LoadImageWrapper = LoadImageWrapper(Compose([FillTransparent(0), PadToSquare(0), Resize(32)]))

    def _process_images(self, items: list[Path], max_workers: int | None = None):
        def _process_image(path: str):
            image = self._processor(path)
            return cv2.img_hash.colorMomentHash(image)

        return self._map(_process_image, items, max_workers=max_workers)

    def process(
        self,
        image: Path,
    ) -> float:
        if isinstance(image, Path):
            image = [image]

        return self.process_batch(image)[0]

    def process_batch(
        self,
        images: list[Path],
        max_workers: int | None = None,
    ) -> list[float]:
        return self._process_images(images, max_workers)
