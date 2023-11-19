from dataclasses import dataclass, field
from pathlib import Path

import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd

from preparation_suite.module.Processor import Processor
from preparation_suite.util.Image import (
    Compose,
    FillTransparent,
    LoadImageWrapper,
    PadToSquare,
    Resize,
)


@dataclass
class Tag:
    name: str = ""
    confidence: float = 0.0

    def __str__(self) -> str:
        return self.name

    def clean(self):
        return self.name.replace("_", " ").replace("(", r"\(").replace(")", r"\)")


@dataclass
class TagGroup:
    name: str = ""
    tags: list[Tag] = field(default_factory=list)

    def __str__(self) -> str:
        return ", ".join(map(str, self.tags))

    def clean(self):
        return ", ".join(tag.clean() for tag in self.tags)


@dataclass
class Rating(TagGroup):
    name: str = "Rating"


@dataclass
class General(TagGroup):
    name: str = "General"


@dataclass
class Character(TagGroup):
    name: str = "Character"


class WDTags(Processor):
    # Specify providers to support different execution environments.
    providers: str | list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def __init__(
        self,
        repo: str = "SmilingWolf/wd-v1-4-moat-tagger-v2",
        model_filename: str = "model.onnx",
        labels_filename: str = "selected_tags.csv",
    ) -> None:
        super().__init__()
        self.repo = repo
        self.model_filename = model_filename
        self.labels_filename = labels_filename

        # Initialize model and labels.
        self.model = self._load_model()
        (
            self.tag_names,
            self.rating_indexes,
            self.general_indexes,
            self.character_indexes,
        ) = self._load_labels()

        # Extract input requirements from the model.
        _, self.img_input_size, _, _ = self.model.get_inputs()[0].shape
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

        self._pipeline = self._preprocess_compose()

    def process(
        self,
        image: str | Path,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        max_workers: int | None = None,
    ) -> tuple[Rating, General, Character]:
        """Classify image contents, returning formatted and raw tag strings along with individual tag dictionaries."""
        image = self._to_paths([image])
        return self.process_batch(
            image,
            general_threshold,
            character_threshold,
            max_workers=max_workers,
        )[0]

    def process_batch(
        self,
        images: list[str | Path],
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        max_workers: int | None = None,
    ) -> list[tuple[Rating, General, Character]]:
        """Classify image contents, returning formatted and raw tag strings along with individual tag dictionaries."""
        images = self._to_paths(images)

        results = []
        for image in self._iter(self._preprocess, images, max_workers=max_workers):
            # Run inference on the model.
            probabilities = self.model.run([self.output_name], {self.input_name: image})[0]
            labels_with_confidences = list(
                zip(self.tag_names, probabilities[0].astype(float), strict=True),
            )

            # Extract relevant information from the model's output.
            rating, general_tags, character_tags = self._postprocess(
                labels_with_confidences,
                general_threshold,
                character_threshold,
            )

            rating = Rating(
                tags=[Tag(name=k, confidence=v) for k, v in rating.items()],
            )

            general = General(
                tags=[Tag(name=k, confidence=v) for k, v in general_tags.items()],
            )

            character = Character(
                tags=[Tag(name=k, confidence=v) for k, v in character_tags.items()],
            )

            results.append((rating, general, character))

        return results

    def _load_model(self) -> rt.InferenceSession:
        """Load the ONNX model from the specified repository or local directory."""
        model_path = huggingface_hub.hf_hub_download(self.repo, self.model_filename)
        return rt.InferenceSession(model_path, providers=self.providers)

    def _load_labels(self) -> tuple[list[str], list[int], list[int], list[int]]:
        """Retrieve and categorize labels from the remote repository or local file."""
        path = huggingface_hub.hf_hub_download(self.repo, self.labels_filename)
        df = pd.read_csv(path).convert_dtypes()

        tag_names = df["name"].tolist()
        # Indices correspond to specific types of labels.
        rating_indexes = list(np.where(df["category"] == 9)[0])
        general_indexes = list(np.where(df["category"] == 0)[0])
        character_indexes = list(np.where(df["category"] == 4)[0])
        return tag_names, rating_indexes, general_indexes, character_indexes

    def _preprocess_compose(self) -> LoadImageWrapper:
        """Create a composition of preprocessing functions depending on the input type."""
        return LoadImageWrapper(
            Compose(
                [
                    FillTransparent(color=255),
                    PadToSquare(color=(255, 255, 255)),
                    Resize((self.img_input_size, self.img_input_size)),
                ],
            ),
        )

    def _preprocess(self, image: Path) -> np.ndarray:
        """Prepare the image for input into the model."""
        processed_image = self._pipeline(str(image))

        return np.expand_dims(np.asarray(processed_image, dtype=np.float32), 0)

    def _postprocess(
        self,
        labels: list[tuple[str, np.float16]],
        general_threshold: float,
        character_threshold: float,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Separate and threshold the model's output labels into categories."""
        # Separate labels based on pre-determined indices and apply confidence threshold.
        rating_labels = [labels[i] for i in self.rating_indexes]
        rating_dict = {k: float(v) for k, v in rating_labels}

        general_labels = [labels[i] for i in self.general_indexes]
        general_tags = {k: float(v) for k, v in general_labels if v > general_threshold}

        character_labels = [labels[i] for i in self.character_indexes]
        character_tags = {k: float(v) for k, v in character_labels if v > character_threshold}

        return rating_dict, general_tags, character_tags
