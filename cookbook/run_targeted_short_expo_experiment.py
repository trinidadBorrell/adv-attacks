#!/usr/bin/env python3
"""
Targeted Attack Pipeline with Original Label Constraint

This script runs targeted adversarial attacks but only on images that have a specific
original label (coarse or fine class). Images that don't match are skipped ON-THE-FLY
(no pre-classification of all images).

Supports both local datasets and streaming from large datasets (ImageNet, COCO).

Usage:
    # Local dataset:
    python run_targeted_short_expo_experiment.py <epsilons> <target> <original> <test_type> <output> --source local --path /path/to/images

    # Streaming ImageNet:
    python run_targeted_short_expo_experiment.py <epsilons> <target> <original> <test_type> <output> --source imagenet --split validation

    # Streaming COCO:
    python run_targeted_short_expo_experiment.py <epsilons> <target> <original> <test_type> <output> --source coco --split validation

Arguments:
    epsilons: Comma-separated epsilon values (e.g., '8.0,16.0')
    target_category: The target coarse class to attack towards
    original_category: The required original label (coarse or fine class)
    test_type: Test type (1 or 2)
    output: Output directory
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Set

import torch
import torch.nn.functional as F

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.streaming_dataset import create_dataset
from src.targeted.one_targets.batch_pipeline_multiprocessing import run_batch_attacks
from src.utils import (
    get_correct_coarse_mappings,
    get_ensemble_logits,
    get_normalize_transform,
    load_ensemble,
    load_image,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_imagenet_class_names() -> dict:
    """Load ImageNet class names from file."""
    class_names = {}
    try:
        with open("imagenet_classes/imagenet_classes.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and "," in line and not line.startswith("list"):
                    parts = line.split(", ", 1)
                    if len(parts) == 2:
                        class_id = int(parts[0])
                        class_name = parts[1]
                        class_names[class_id] = class_name
    except FileNotFoundError:
        logger.warning(
            "ImageNet class names file not found. Using class indices as names."
        )
        for i in range(1000):
            class_names[i] = f"class_{i}"
    return class_names


class OriginalLabelFilter:
    """Filter images based on their predicted original label (coarse or fine class)."""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.normalize = get_normalize_transform()
        self.models = load_ensemble(self.device)
        self.coarse_labels, self.coarse_indices = get_correct_coarse_mappings()
        self.fine_class_names = load_imagenet_class_names()
        logger.info(f"OriginalLabelFilter initialized on {device}")

    def get_image_label(self, image_path: str) -> str:
        """
        Get the predicted label for an image.

        Returns:
            - Coarse class name if the top prediction index is in a coarse class
            - Fine class name if the top prediction index is NOT in any coarse class
        """
        try:
            image = load_image(image_path, self.device)
            logits = get_ensemble_logits(self.normalize(image), self.models)
            probs = F.softmax(logits, dim=1)
            top_idx = torch.argmax(probs, dim=1).item()

            # First, check if this fine class belongs to any coarse class
            for i, indices in enumerate(self.coarse_indices):
                if top_idx in indices:
                    return self.coarse_labels[i]

            # If not in any coarse class, return the fine class name
            return self.fine_class_names.get(top_idx, f"class_{top_idx}")
        except Exception as e:
            logger.warning(f"Could not classify {image_path}: {e}")
            return "error"


def load_used_images(output_dir: Path) -> Set[str]:
    """Load the set of already used image paths."""
    used_images_file = output_dir / "used_images_short_expo.json"
    if used_images_file.exists():
        try:
            with open(used_images_file, "r") as f:
                used_images_list = json.load(f)
                return set(used_images_list)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load used images file: {e}")
            return set()
    return set()


def save_used_images(used_images: Set[str], output_dir: Path) -> None:
    """Save the set of used image paths to disk."""
    used_images_file = output_dir / "used_images_short_expo.json"
    try:
        with open(used_images_file, "w") as f:
            json.dump(list(used_images), f, indent=2)
        logger.info(f"Saved {len(used_images)} used image paths to {used_images_file}")
    except IOError as e:
        logger.error(f"Could not save used images file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run targeted adversarial attacks with original label constraint (on-the-fly filtering)"
    )
    parser.add_argument(
        "epsilons", help="Comma-separated epsilon values (e.g., '8.0,16.0')"
    )
    parser.add_argument("target_category", help="Target coarse class to attack towards")
    parser.add_argument(
        "original_category",
        help="Required original label (coarse or fine class, images not matching are skipped)",
    )
    parser.add_argument(
        "test_type", type=int, choices=[1, 2], help="Test type (1 or 2)"
    )
    parser.add_argument("output", help="Output directory")

    # Dataset source options
    parser.add_argument(
        "--source",
        type=str,
        choices=["local", "imagenet", "coco"],
        default="local",
        help="Dataset source: 'local' (default), 'imagenet' (streaming), or 'coco' (streaming)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to local dataset folder (required if --source=local)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split for streaming datasets: 'train' or 'validation' (default)",
    )
    parser.add_argument(
        "--delete_after_use",
        action="store_true",
        help="Delete downloaded images after processing (for streaming datasets)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--target_successes",
        type=int,
        default=35,
        help="Target number of successful attacks (default: 5)",
    )

    args = parser.parse_args()

    # Parse arguments
    epsilons = [float(eps.strip()) for eps in args.epsilons.split(",")]
    target_category = args.target_category.strip()
    original_category = args.original_category.strip()

    # Validate that original and target categories are different
    if target_category == original_category:
        logger.error("Target category and original category must be different!")
        return

    # Validate source/path combination
    if args.source == "local" and not args.path:
        logger.error("--path is required when using --source=local")
        return

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load used images
    used_images = load_used_images(output_dir)

    # Initialize the label filter (loads models once)
    logger.info("Initializing label filter...")
    label_filter = OriginalLabelFilter()

    # Create dataset based on source
    logger.info(f"Creating dataset from source: {args.source}")
    if args.source == "local":
        # For local, check if it's the old mini_imagenet structure
        local_path = Path(args.path)
        mini_imagenet_path = local_path / "mini_imagenet_dog"
        if mini_imagenet_path.exists():
            dataset = create_dataset("local", path=str(mini_imagenet_path))
        else:
            dataset = create_dataset("local", path=str(local_path))
    else:
        # Streaming dataset (ImageNet or COCO)
        cache_dir = output_dir / "image_cache"
        dataset = create_dataset(
            args.source,
            split=args.split,
            delete_after_use=args.delete_after_use,
            cache_dir=str(cache_dir),
        )

    # Run pipeline with on-the-fly filtering
    logger.info("Starting attack pipeline with ON-THE-FLY filtering...")
    logger.info(f"Epsilons: {epsilons}")
    logger.info(f"Target category: {target_category}")
    logger.info(f"Original category (required): {original_category}")
    logger.info(f"Test type: {args.test_type}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Target successful attacks: {args.target_successes}")
    logger.info(f"Dataset source: {args.source}")

    # Process until target successes are reached
    overall_summary = {
        "total_images": 0,
        "total_tasks": 0,
        "successful_attacks": 0,
        "failed_validations": 0,
        "failed_tests": 0,
        "success_rate": 0.0,
        "successful_results": [],
        "images_checked": 0,
        "images_matched_label": 0,
        "experiment_config": {
            "target_category": target_category,
            "original_category": original_category,
            "epsilons": epsilons,
            "test_type": args.test_type,
            "source": args.source,
        },
    }

    category_successes = 0
    total_iterations = 0
    images_checked = 0
    images_matched = 0
    batch_buffer = []  # Buffer to collect matching images for batch processing

    logger.info(f"Target: {args.target_successes} successful attacks")

    # ON-THE-FLY FILTERING: Iterate through dataset, classify each image,
    # and only process those matching the required original label
    for image_path in dataset:
        # Skip already used images
        if image_path in used_images:
            continue

        images_checked += 1

        # Log progress periodically
        if images_checked % 50 == 0:
            logger.info(
                f"Checked {images_checked} images, "
                f"found {images_matched} matching '{original_category}', "
                f"{category_successes} successful attacks so far"
            )

        # Check if image matches required original label (ON-THE-FLY)
        try:
            image_label = label_filter.get_image_label(image_path)
        except Exception as e:
            logger.warning(f"Error classifying {image_path}: {e}")
            dataset.mark_processed(image_path)
            continue

        if image_label != original_category:
            # Not a match - mark as processed (deletes if streaming)
            dataset.mark_processed(image_path)
            continue

        # Image matches! Add to batch buffer
        images_matched += 1
        batch_buffer.append(image_path)

        # Process batch when buffer is full
        if len(batch_buffer) >= args.batch_size:
            batch_image_paths = batch_buffer
            batch_buffer = []

            batch_fine_class_ids = [-1] * len(batch_image_paths)
            batch_target_coarse_classes = [target_category] * len(batch_image_paths)

            logger.info(
                f"Processing batch of {len(batch_image_paths)} matching images..."
            )

            # Run batch processing
            batch_results = run_batch_attacks(
                image_paths=batch_image_paths,
                fine_class_ids=batch_fine_class_ids,
                targeted_coarse_classes=batch_target_coarse_classes,
                epsilons=epsilons,
                test_types=[args.test_type],
                output_base_dir=output_dir,
                batch_size=args.batch_size,
            )

            # Update counters
            batch_successes = batch_results["successful_attacks"]
            category_successes += batch_successes

            # Add successful images to used_images set
            for success_result in batch_results["successful_results"]:
                used_images.add(success_result["image_path"])

            # Mark processed images for cleanup (streaming datasets)
            for img_path in batch_image_paths:
                dataset.mark_processed(img_path)

            # Update overall summary
            overall_summary["total_images"] += batch_results["total_images"]
            overall_summary["total_tasks"] += batch_results["total_tasks"]
            overall_summary["successful_attacks"] += batch_results["successful_attacks"]
            overall_summary["failed_validations"] += batch_results["failed_validations"]
            overall_summary["failed_tests"] += batch_results["failed_tests"]
            overall_summary["successful_results"].extend(
                batch_results["successful_results"]
            )

            # Periodic checkpoint
            total_iterations += 1
            if total_iterations % 5 == 0:
                save_used_images(used_images, output_dir)
                logger.info(
                    f"Checkpoint: Saved used images list at iteration {total_iterations}"
                )

            logger.info(
                f"Batch complete: {batch_successes} successes. "
                f"Total: {category_successes}/{args.target_successes}"
            )

            # Check if we've reached our target
            if category_successes >= args.target_successes:
                logger.info(
                    f"✅ Target reached: {category_successes} successful attacks"
                )
                break

    # Process any remaining images in buffer
    if batch_buffer and category_successes < args.target_successes:
        batch_image_paths = batch_buffer
        batch_fine_class_ids = [-1] * len(batch_image_paths)
        batch_target_coarse_classes = [target_category] * len(batch_image_paths)

        logger.info(f"Processing final batch of {len(batch_image_paths)} images...")

        batch_results = run_batch_attacks(
            image_paths=batch_image_paths,
            fine_class_ids=batch_fine_class_ids,
            targeted_coarse_classes=batch_target_coarse_classes,
            epsilons=epsilons,
            test_types=[args.test_type],
            output_base_dir=output_dir,
            batch_size=args.batch_size,
        )

        category_successes += batch_results["successful_attacks"]
        for success_result in batch_results["successful_results"]:
            used_images.add(success_result["image_path"])
        for img_path in batch_image_paths:
            dataset.mark_processed(img_path)

        overall_summary["total_images"] += batch_results["total_images"]
        overall_summary["total_tasks"] += batch_results["total_tasks"]
        overall_summary["successful_attacks"] += batch_results["successful_attacks"]
        overall_summary["failed_validations"] += batch_results["failed_validations"]
        overall_summary["failed_tests"] += batch_results["failed_tests"]
        overall_summary["successful_results"].extend(
            batch_results["successful_results"]
        )

    # Update final stats
    overall_summary["images_checked"] = images_checked
    overall_summary["images_matched_label"] = images_matched

    # Calculate overall success rate
    if overall_summary["total_tasks"] > 0:
        overall_summary["success_rate"] = (
            overall_summary["successful_attacks"] * 100
        ) / overall_summary["total_tasks"]

    results_summary = overall_summary

    # Final save of used images
    save_used_images(used_images, output_dir)

    # Save experiment summary
    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved experiment summary to {summary_file}")

    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"Original category constraint: {original_category}")
    logger.info(f"Target category: {target_category}")
    logger.info(f"Images checked: {images_checked}")
    logger.info(f"Images matching '{original_category}': {images_matched}")
    logger.info(f"Total images processed: {results_summary['total_images']}")
    logger.info(f"Total tasks: {results_summary['total_tasks']}")
    logger.info(f"Successful attacks: {results_summary['successful_attacks']}")
    logger.info(f"Failed validations: {results_summary['failed_validations']}")
    logger.info(f"Failed tests: {results_summary['failed_tests']}")
    logger.info(f"Overall success rate: {results_summary['success_rate']:.1f}%")

    if category_successes >= args.target_successes:
        logger.info(
            f"✅ COMPLETED: Achieved {category_successes}/{args.target_successes} successful attacks"
        )
    else:
        logger.warning(
            f"⚠️  INCOMPLETE: Only achieved {category_successes}/{args.target_successes} successful attacks "
            f"(checked {images_checked} images, {images_matched} matched label)"
        )


if __name__ == "__main__":
    main()
