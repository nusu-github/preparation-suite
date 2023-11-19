from pathlib import Path

import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoImageProcessor, AutoModelForImageClassification

from preparation_suite.module.Processor import Processor
from preparation_suite.util.Image import Compose, FillTransparent, LoadImageWrapper


class ImageEmbeddingDeit(Processor):
    repo: str = "facebook/deit-base-distilled-patch16-384"

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, repo: str | None = repo, device: str = _device) -> None:
        super().__init__()
        repo = repo or self.repo

        self._device = device
        self._load_models(repo)

        self._processor: LoadImageWrapper = LoadImageWrapper(Compose([FillTransparent(0)]))

    def __del__(self) -> None:
        del self._model

    def _load_models(self, repo: str):
        model = AutoModelForImageClassification.from_pretrained(repo, device_map=self._device)
        image_processor = AutoImageProcessor.from_pretrained(repo)

        model = BetterTransformer.transform(model)

        self._model = model
        self._image_processor = image_processor

    def _open_images(self, items: list[Path], max_workers: int | None = None):
        images = self._map(self._processor, items, max_workers=max_workers)
        return self._image_processor.preprocess(images, return_tensors="pt")

    def process(
        self,
        image: Path,
    ) -> float:
        if isinstance(image, Path):
            image = [image]

        return self.process_batch(image)[0]

    @torch.inference_mode()
    def process_batch(
        self,
        images: list[Path],
        max_workers: int | None = None,
    ) -> list[float]:
        thumbs = self._open_images(images, max_workers).to(self._device)
        image_features = self._model(**thumbs)

        return image_features.logits.detach().cpu().numpy()
