from pathlib import Path

import torch
from optimum.bettertransformer import BetterTransformer
from torch import nn
from transformers import CLIPImageProcessor, CLIPModel

from preparation_suite.module.Processor import Processor
from preparation_suite.util.Image import Compose, FillTransparent, LoadImageWrapper


class AestheticMeanPredictionLinearModel(nn.Module):
    def __init__(self, feats_in) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(feats_in, 1)

    def forward(self, tensor):
        x = nn.functional.normalize(tensor, dim=-1) * tensor.shape[-1] ** 0.5
        return self.linear(x)


class AestheticScorer(Processor):
    repo = "openai/clip-vit-large-patch14"
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(
        self,
        model_path: str | Path,
        repo: str | None = None,
        device: str = _device,
        cast: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(model_path, str):
            model_path = Path(model_path)

        repo = repo or self.repo

        self._cast = cast
        self._device = device
        self._load_models(model_path, repo)

        self._processor: LoadImageWrapper = LoadImageWrapper(Compose([FillTransparent(0)]))

    def _load_models(self, model_path: Path, repo: str):
        clip_model = CLIPModel.from_pretrained(repo, device_map=self._device)
        aesthetic_model = AestheticMeanPredictionLinearModel(clip_model.config.projection_dim)
        aesthetic_model.load_state_dict(torch.load(model_path))
        image_processor = CLIPImageProcessor.from_pretrained(repo)

        clip_model = BetterTransformer.transform(clip_model)

        self._clip_model = clip_model
        self._aesthetic_model = aesthetic_model.to(self._device)
        self._image_processor = image_processor

    def _open_images(self, items: list[Path], max_workers: int | None = None):
        images = self._map(self._processor, items, max_workers=max_workers)
        return self._image_processor.preprocess(images, do_normalize=True, return_tensors="pt")

    def process(
        self,
        image: Path,
    ) -> float:
        if isinstance(image, str | Path):
            image = [image]

        return self.process_batch(image)[0]

    @torch.inference_mode()
    def process_batch(
        self,
        images: list[Path],
        max_workers: int | None = None,
    ) -> list[float]:
        thumbs = self._open_images(images, max_workers).to(self._device)

        with torch.autocast(device_type=self._device, enabled=self._cast):
            encoded = self._clip_model.get_image_features(**thumbs)

        clip_image_embed = nn.functional.normalize(encoded.float(), dim=-1)
        score = self._aesthetic_model(clip_image_embed)

        return score.detach().cpu().numpy().astype(float).flatten().tolist()
