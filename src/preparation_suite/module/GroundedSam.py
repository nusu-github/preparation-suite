from pathlib import Path

import numpy as np
import supervision as sv
import torch
from tqdm import tqdm

from preparation_suite import MODEL_DIR
from preparation_suite.module.Processor import Processor
from preparation_suite.util.Image import Compose, FillTransparent, LoadImageWrapper


class GroundedSam(Processor):
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor: LoadImageWrapper = LoadImageWrapper(Compose([FillTransparent(0)]))

    def __init__(
        self,
        sam_encoder: str,
        sam_checkpoint: Path | None = None,
        grounding_dino_config: Path | None = None,
        grounding_dino_checkpoint: Path | None = None,
        cast: bool = True,
    ) -> None:
        super().__init__()

        self._cast = cast

        self.grounding_dino_config = grounding_dino_config
        self.grounding_dino_checkpoint = grounding_dino_checkpoint or MODEL_DIR / "groundingdino_swinb_cogcoor.pth"
        self.sam_encoder = sam_encoder
        self.sam_checkpoint = sam_checkpoint

        self._initialize_grounding_dino()
        self._initialize_sam()

    def _initialize_grounding_dino(self):
        model_config_paths = {
            "groundingdino_swinb_cogcoor.pth": "GroundingDINO_SwinB_cfg.py",
            "groundingdino_swint_ogc.pth": "GroundingDINO_SwinT_OGC.py",
        }
        import inspect

        import groundingdino as gd
        from groundingdino.util.inference import Model as GroundingDino

        config = Path(inspect.getfile(gd)).parent.joinpath("config")

        model_config_path = (
            model_config_paths[Path(self.grounding_dino_checkpoint).name] or "groundingdino_swinb_cogcoor.pth"
        )

        model = GroundingDino(
            str(config / model_config_path),
            str(self.grounding_dino_checkpoint),
            device=self._device,
        )
        model.model.eval()

        self.dino_model = model

    def _initialize_sam(self):
        from segment_anything_hq import SamPredictor, sam_model_registry

        self.sam = sam_model_registry[self.sam_encoder](
            checkpoint=str(self.sam_checkpoint),
        )
        self.sam.eval().to(device=torch.device(self._device))
        self.sam_predictor = SamPredictor(self.sam)

    def _open_images(self, items: list[Path], max_workers: int | None = None):
        return self._iter(self._processor, items, max_workers=max_workers)

    def process(
        self,
        img: str | Path,
        classes: list[str],
        box_threshold: float = 0.3,
        nms_threshold: float = 0.8,
        max_workers: int | None = None,
    ):
        img = [self._to_path(img)]
        return self.process_batch(
            imgs=img,
            classes=classes,
            box_threshold=box_threshold,
            max_workers=max_workers,
        )

    def process_batch(
        self,
        imgs: list[Path],
        classes: list[str],
        box_threshold: float = 0.3,
        nms_threshold: float = 0.8,
        max_workers=None,
    ):
        results = []
        for image in tqdm(self._open_images(imgs, max_workers=max_workers), total=len(imgs)):
            detections = self._ground(
                img=image,
                classes=classes,
                box_threshold=box_threshold,
                nms_threshold=nms_threshold,
            )

            detections = self._segment(image, detections)

            results.append(detections)

        return results

    def _ground(self, img: np.ndarray, classes: list[str], box_threshold: float, nms_threshold: float):
        with torch.inference_mode(), torch.autocast(device_type=self._device, enabled=self._cast):
            detections = self.dino_model.predict_with_classes(img, classes, box_threshold, 0.3)
            return detections.with_nms(threshold=nms_threshold)

    def _segment(self, img: np.ndarray, detections: sv.Detections):
        self.sam_predictor.set_image(img, "BGR")

        H, W, _ = img.shape
        boxes_xyxy = torch.from_numpy(detections.xyxy)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, (H, W)).to(device=self._device)

        with torch.autocast(device_type=self._device, enabled=self._cast):
            masks, scores, _ = self.sam_predictor.predict_torch(
                point_labels=None,
                point_coords=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        masks = masks.detach().cpu().numpy().reshape(-1, H, W)
        scores = scores.detach().cpu().numpy().reshape(-1)

        return sv.Detections(
            xyxy=detections.xyxy,
            class_id=detections.class_id,
            mask=masks,
            confidence=scores,
        )

    def annotate(self, image: str | Path, detections: sv.Detections):
        mask_annotator = sv.MaskAnnotator()
        return mask_annotator.annotate(scene=self._processor(image), detections=detections)

    def annotate_batch(self, image: list[str | Path], detections: list[sv.Detections]):
        return self._iter(lambda x: self.annotate(*x), zip(image, detections, strict=True))
