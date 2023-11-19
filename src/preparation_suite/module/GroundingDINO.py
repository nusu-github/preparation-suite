import contextlib
from pathlib import Path

import torch
from groundingdino.util.inference import Model

from preparation_suite.module.Processor import Processor
from preparation_suite.util.Image import Compose, FillTransparent, LoadImageWrapper


class GroundingDINO(Processor):
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor: LoadImageWrapper = LoadImageWrapper(Compose([FillTransparent(0)]))

    def __init__(self, model_path: str | Path, device: str | None = None, cast: bool = True) -> None:
        super().__init__()
        if isinstance(model_path, str):
            model_path = Path(model_path)

        self._cast = cast
        self._device = device or self._device
        self._load_models(model_path)

    def __del__(self) -> None:
        del self._model
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def _load_models(self, model_path: Path):
        model_config_paths = {
            "groundingdino_swinb_cogcoor.pth": "GroundingDINO_SwinB_cfg.py",
            "groundingdino_swint_ogc.pth": "GroundingDINO_SwinT_OGC.py",
        }
        import inspect

        import groundingdino as gd

        config = Path(inspect.getfile(gd)).parent.joinpath("config")
        model_config_path = model_config_paths[model_path.name] or "groundingdino_swinb_cogcoor.pth"

        model = Model(
            str(config / model_config_path),
            str(model_path),
            device=self._device,
            cast=self._cast,
        )
        model.model.eval()

        self._model = model

    def process(
        self,
        image: str | Path,
        classes=None,
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
        max_workers: int | None = None,
    ):
        if classes is None:
            classes = ["Person"]
        image = [self._to_path(image)]
        return self.process_batch(
            images=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_workers=max_workers,
        )[0]

    def process_batch(
        self,
        images: list[Path],
        classes=None,
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
        max_workers: int | None = None,
    ):
        if classes is None:
            classes = ["Person"]
        results = []
        for image in self._iter(self._processor, images, max_workers=max_workers):
            detections = self._model.predict_with_classes(
                image=image,
                classes=classes,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            results.append(detections)

        return results
