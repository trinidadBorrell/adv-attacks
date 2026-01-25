#!/usr/bin/env python3
"""
Streaming Dataset Utility

Downloads images on-the-fly from large datasets (ImageNet, COCO), processes them,
and optionally deletes them after use to manage storage constraints.

Supports:
- ImageNet (via HuggingFace datasets)
- COCO (via fiftyone or direct download)
- Local datasets (no download, just iteration)
"""

import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Iterator, Optional, Set

logger = logging.getLogger(__name__)


class StreamingDataset(ABC):
    """Abstract base class for streaming datasets."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        delete_after_use: bool = True,
        max_cache_size: int = 100,
    ):
        """
        Args:
            cache_dir: Directory to cache downloaded images. If None, uses temp dir.
            delete_after_use: Whether to delete images after they've been processed.
            max_cache_size: Maximum number of images to keep in cache before cleanup.
        """
        self.delete_after_use = delete_after_use
        self.max_cache_size = max_cache_size
        self.cache_dir = cache_dir or Path(
            tempfile.mkdtemp(prefix="streaming_dataset_")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached_files: list = []
        logger.info(f"Streaming dataset initialized. Cache dir: {self.cache_dir}")

    def cleanup_cache(self, force_all: bool = False):
        """Remove cached files to free up space."""
        if force_all:
            # Remove all cached files
            for f in self._cached_files:
                try:
                    if Path(f).exists():
                        Path(f).unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {f}: {e}")
            self._cached_files.clear()
        elif len(self._cached_files) > self.max_cache_size:
            # Remove oldest files to stay under limit
            files_to_remove = self._cached_files[
                : len(self._cached_files) - self.max_cache_size
            ]
            for f in files_to_remove:
                try:
                    if Path(f).exists():
                        Path(f).unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {f}: {e}")
            self._cached_files = self._cached_files[len(files_to_remove) :]

    def mark_processed(self, image_path: str):
        """Mark an image as processed. If delete_after_use is True, deletes it."""
        if self.delete_after_use:
            try:
                if Path(image_path).exists():
                    Path(image_path).unlink()
                if image_path in self._cached_files:
                    self._cached_files.remove(image_path)
            except Exception as e:
                logger.warning(f"Could not delete processed image {image_path}: {e}")

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iterate over image paths. Downloads on-demand if needed."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of images (may be approximate for streaming)."""
        pass

    def __del__(self):
        """Cleanup on deletion."""
        if self.delete_after_use:
            self.cleanup_cache(force_all=True)


class LocalDataset(StreamingDataset):
    """Iterate over a local directory of images without downloading."""

    def __init__(
        self,
        root_dir: Path,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".JPEG"),
        **kwargs,
    ):
        # For local datasets, we don't delete files
        kwargs["delete_after_use"] = False
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        self.extensions = extensions
        self._image_paths = self._discover_images()
        logger.info(
            f"LocalDataset: Found {len(self._image_paths)} images in {root_dir}"
        )

    def _discover_images(self) -> list:
        """Discover all image files in the directory."""
        images = []
        for ext in self.extensions:
            images.extend(self.root_dir.rglob(f"*{ext}"))
        return [str(p) for p in images]

    def __iter__(self) -> Iterator[str]:
        for path in self._image_paths:
            yield path

    def __len__(self) -> int:
        return len(self._image_paths)


class ImageNetStreaming(StreamingDataset):
    """Stream images from ImageNet using HuggingFace datasets."""

    def __init__(
        self,
        split: str = "validation",
        subset: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            split: Dataset split ('train', 'validation', 'test')
            subset: Optional subset name
        """
        super().__init__(**kwargs)
        self.split = split
        self.subset = subset
        self._dataset = None
        self._length = None

    def _load_dataset(self):
        """Lazy load the dataset."""
        if self._dataset is None:
            try:
                from datasets import load_dataset

                logger.info(
                    "Loading ImageNet dataset from HuggingFace (streaming mode)..."
                )
                self._dataset = load_dataset(
                    "ILSVRC/imagenet-1k",
                    split=self.split,
                    streaming=True,
                    trust_remote_code=True,
                )
                logger.info("ImageNet dataset loaded in streaming mode")
            except Exception as e:
                logger.error(f"Failed to load ImageNet: {e}")
                logger.info(
                    "Make sure you have accepted the dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k"
                )
                raise

    def __iter__(self) -> Iterator[str]:
        self._load_dataset()

        for i, example in enumerate(self._dataset):
            # Save image to cache
            image = example["image"]
            label = example.get("label", i)

            image_path = self.cache_dir / f"imagenet_{i}_{label}.jpg"
            image.save(str(image_path))
            self._cached_files.append(str(image_path))

            # Cleanup if cache is too large
            self.cleanup_cache()

            yield str(image_path)

    def __len__(self) -> int:
        # ImageNet validation has ~50k images, train has ~1.2M
        if self.split == "validation":
            return 50000
        elif self.split == "train":
            return 1281167
        return 0


class COCOStreaming(StreamingDataset):
    """Stream images from COCO dataset."""

    def __init__(
        self,
        split: str = "validation",
        year: str = "2017",
        **kwargs,
    ):
        """
        Args:
            split: Dataset split ('train', 'validation')
            year: COCO year ('2014', '2017')
        """
        super().__init__(**kwargs)
        self.split = split
        self.year = year
        self._dataset = None

    def _load_dataset(self):
        """Lazy load the dataset."""
        if self._dataset is None:
            try:
                from datasets import load_dataset

                logger.info(
                    f"Loading COCO {self.year} dataset from HuggingFace (streaming mode)..."
                )

                # Map split names
                hf_split = "train" if self.split == "train" else "validation"

                self._dataset = load_dataset(
                    "detection-datasets/coco",
                    split=hf_split,
                    streaming=True,
                )
                logger.info("COCO dataset loaded in streaming mode")
            except Exception as e:
                logger.error(f"Failed to load COCO: {e}")
                raise

    def __iter__(self) -> Iterator[str]:
        self._load_dataset()

        for i, example in enumerate(self._dataset):
            # Save image to cache
            image = example["image"]
            image_id = example.get("image_id", i)

            image_path = self.cache_dir / f"coco_{image_id}.jpg"
            image.save(str(image_path))
            self._cached_files.append(str(image_path))

            # Cleanup if cache is too large
            self.cleanup_cache()

            yield str(image_path)

    def __len__(self) -> int:
        # COCO 2017: train ~118k, val ~5k
        if self.split == "train":
            return 118287
        return 5000


class FilteredDatasetIterator:
    """
    Wraps a dataset and filters images on-the-fly based on a label filter function.

    This avoids pre-classifying all images by checking each image as we iterate.
    """

    def __init__(
        self,
        dataset: StreamingDataset,
        label_filter_fn,
        required_label: str,
        used_images: Optional[Set[str]] = None,
        max_images: Optional[int] = None,
    ):
        """
        Args:
            dataset: The underlying dataset to iterate over
            label_filter_fn: Function(image_path) -> str that returns the image's label
            required_label: Only yield images with this label
            used_images: Set of already-used image paths to skip
            max_images: Maximum number of matching images to yield (None for unlimited)
        """
        self.dataset = dataset
        self.label_filter_fn = label_filter_fn
        self.required_label = required_label
        self.used_images = used_images or set()
        self.max_images = max_images
        self._yielded_count = 0
        self._checked_count = 0

    def __iter__(self) -> Generator[str, None, None]:
        for image_path in self.dataset:
            # Skip already used images
            if image_path in self.used_images:
                continue

            self._checked_count += 1

            # Log progress periodically
            if self._checked_count % 50 == 0:
                logger.info(
                    f"Checked {self._checked_count} images, "
                    f"found {self._yielded_count} matching '{self.required_label}'"
                )

            # Check if image matches required label
            try:
                image_label = self.label_filter_fn(image_path)
                if image_label == self.required_label:
                    self._yielded_count += 1
                    yield image_path

                    # Check if we've reached max
                    if self.max_images and self._yielded_count >= self.max_images:
                        logger.info(f"Reached max_images limit ({self.max_images})")
                        return
                else:
                    # Mark non-matching images as processed (delete if streaming)
                    self.dataset.mark_processed(image_path)
            except Exception as e:
                logger.warning(f"Error classifying {image_path}: {e}")
                self.dataset.mark_processed(image_path)

    @property
    def stats(self) -> dict:
        """Return statistics about the filtering process."""
        return {
            "checked": self._checked_count,
            "matched": self._yielded_count,
            "match_rate": self._yielded_count / self._checked_count
            if self._checked_count > 0
            else 0,
        }


def create_dataset(
    source: str,
    path: Optional[str] = None,
    split: str = "validation",
    delete_after_use: bool = True,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> StreamingDataset:
    """
    Factory function to create a streaming dataset.

    Args:
        source: Dataset source ('local', 'imagenet', 'coco')
        path: Path for local datasets
        split: Split for remote datasets ('train', 'validation')
        delete_after_use: Whether to delete images after processing
        cache_dir: Directory to cache downloaded images

    Returns:
        StreamingDataset instance
    """
    cache_path = Path(cache_dir) if cache_dir else None

    if source == "local":
        if not path:
            raise ValueError("path is required for local datasets")
        return LocalDataset(Path(path), cache_dir=cache_path, **kwargs)
    elif source == "imagenet":
        return ImageNetStreaming(
            split=split,
            delete_after_use=delete_after_use,
            cache_dir=cache_path,
            **kwargs,
        )
    elif source == "coco":
        return COCOStreaming(
            split=split,
            delete_after_use=delete_after_use,
            cache_dir=cache_path,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset source: {source}")
