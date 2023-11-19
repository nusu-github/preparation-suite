from os import cpu_count

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.core.utils as fou
import numpy as np
from fiftyone import ViewField as F
from torch import cuda, device


class Preprocessor:
    device_type: str = "cuda" if cuda.is_available() else "cpu"
    _embeddings: np.array = None

    def __init__(
        self,
        dataset: fo.Dataset,
        max_workers: int | None = None,
        batch_size: int = 1,
        gpu_id: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, (cpu_count() or 1) + 4)

        if gpu_id is None:
            gpu_id = cuda.current_device() if self.device_type == "cuda" else -1

        self.device = self._device(gpu_id)

    def _get_batch(self):
        return fou.iter_batches(
            self.dataset.iter_samples(progress=True),
            batch_size=self.batch_size,
        )

    def _device(self, gpu_id: int = -1):
        return device(
            f"cuda:{gpu_id}" if gpu_id != -1 else "cpu",
        )

    def _compute_embeddings(self):
        from preparation_suite.module.ImageEmbeddingDeiT import ImageEmbeddingDeit

        model = ImageEmbeddingDeit(device=self.device_type)

        embeddings_list: list[np.array] = []
        for batch in self._get_batch():
            embeddings = model.process_batch(
                [sample.filepath for sample in batch],
                max_workers=self.max_workers,
            )
            embeddings_list.append(embeddings)

        self._embeddings = np.concatenate(embeddings_list, axis=0)

    def uniqueness(self):
        if self._embeddings is None:
            self._compute_embeddings()

        fob.compute_uniqueness(self.dataset, embeddings=self._embeddings)

    def del_duplicate(self, fraction: float = 0.3):
        if self._embeddings is None:
            self._compute_embeddings()

        results = fob.compute_similarity(
            self.dataset,
            embeddings=self._embeddings,
            brain_key="emb_img_sim",
        )

        self._dup_del(fraction=fraction, results=results)

    def img_hash(self, threshold: float = 0.00001):
        import scipy

        from preparation_suite.module.ImageHash import ImageHash

        image_hash = ImageHash()

        hash_list = image_hash.process_batch(
            [sample.filepath for sample in self.dataset.iter_samples(progress=True)],
            max_workers=self.max_workers,
        )

        hash_list = scipy.stats.zscore(np.concatenate(hash_list, axis=0))
        results = fob.compute_similarity(self.dataset, embeddings=hash_list, brain_key="hash_img_sim")

        self._dup_del(threshold=threshold, results=results)

    def _dup_del(
        self,
        results,
        fraction: float | None = None,
        threshold: float | None = None,
    ):
        if fraction is None and threshold is None:
            msg = "fraction or threshold must be set"
            raise ValueError(msg)

        if fraction is not None:
            results.find_duplicates(fraction=fraction)
        else:
            results.find_duplicates(thresh=threshold)

        try:
            duplicates_view = results.duplicates_view(
                type_field="dup_type",
                id_field="dup_id",
                dist_field="dup_dist",
            )
        except ValueError:
            return

        duplicates_view = duplicates_view.match(F("dup_type") == "duplicate")
        self.dataset.delete_samples(duplicates_view.values("id"))
        self.dataset.reload()

        self.dataset.clear_sample_fields(["dup_type", "dup_id", "dup_dist"])

    def color(self):
        from preparation_suite.module.ImageColor import ImageColor

        image_color = ImageColor()

        color_list = image_color.process_batch(
            [sample.filepath for sample in self.dataset.iter_samples(progress=True)],
            max_workers=self.max_workers,
        )

        for sample, is_color in zip(self.dataset.select_fields().iter_samples(autosave=True), color_list, strict=True):
            sample["is_color"] = is_color

    def clip_aesthetic(self, state_name: str):
        from preparation_suite.module.AestheticScorer import AestheticScorer

        model = AestheticScorer(state_name, device=self.device_type)

        with self.dataset.save_context() as context:
            for batch in self._get_batch():
                scores = model.process_batch(
                    [sample.filepath for sample in batch],
                    max_workers=self.max_workers,
                )
                for sample, score in zip(batch, scores, strict=True):
                    sample["clip_aesthetic"] = float(score)
                    context.save(sample)

    def dino_object_detection(
        self,
        threshold: float = 0.2,
        texts: list[str] | None = None,
    ):
        from preparation_suite import MODEL_DIR
        from preparation_suite.module.GroundingDINO import GroundingDINO

        if texts is None:
            texts = ["person"]  # anime girl . anime boy

        model = GroundingDINO(MODEL_DIR / "groundingdino_swinb_cogcoor.pth", device=self.device_type)

        with self.dataset.save_context() as context:
            for batch in self._get_batch():
                results = model.process_batch(
                    [sample.filepath for sample in batch],
                    classes=texts,
                    box_threshold=threshold,
                    text_threshold=threshold,
                    max_workers=self.max_workers,
                )

                for sample, result in zip(batch, results, strict=True):
                    w, h = sample.metadata.width, sample.metadata.height

                    detections = []
                    for xyxy, _, confidence, class_id, _ in result:
                        x1, y1, x2, y2 = xyxy

                        bounding_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
                        detections.append(
                            fo.Detection(
                                label=texts[(class_id or 0)],
                                bounding_box=bounding_box,
                                confidence=round(confidence, 3),
                            ),
                        )

                    sample["objects"] = fo.Detections(detections=detections)
                    context.save(sample)

    def wd_tags_label(
        self,
        general_threshold=0.1,
        character_threshold=0.3,
    ):
        from preparation_suite.module.WDTags import WDTags

        wd_tags = WDTags()

        with self.dataset.save_context() as context:
            for batch in self._get_batch():
                results = wd_tags.process_batch(
                    [sample.filepath for sample in batch],
                    general_threshold=general_threshold,
                    character_threshold=character_threshold,
                    max_workers=self.max_workers,
                )

                for sample, result in zip(batch, results, strict=True):
                    rating = max(
                        result[0].tags,
                        key=lambda x: x.confidence,
                    )
                    sample["rating"] = fo.Classification(
                        label=rating.name,
                        confidence=rating.confidence,
                    )

                    general_tags = [
                        fo.Classification(
                            label=tag.name,
                            confidence=tag.confidence,
                        )
                        for tag in result[1].tags
                    ]
                    sample["general_tags"] = fo.Classifications(
                        classifications=general_tags,
                    )

                    character_tags = [
                        fo.Classification(
                            label=tag.name,
                            confidence=tag.confidence,
                        )
                        for tag in result[2].tags
                    ]
                    sample["character_tags"] = fo.Classifications(
                        classifications=character_tags,
                    )

                    context.save(sample)
