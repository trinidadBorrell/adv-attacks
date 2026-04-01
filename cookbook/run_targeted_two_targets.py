#!/usr/bin/env python3
"""
Clean Untargeted Attack Pipeline

Usage:
    python run_targeted_two_targets.py <epsilons> <categories1> <categories2> [imagenet_folder] <test_type> <output>
    epsilons: Comma-separated epsilon values (e.g., '8.0,16.0')
    categories1: First category to target
    categories2: Second category to target
    the pair of categories will be considered sequentially (categories1[0], categories2[0]), (categories1[1], categories2[1]), ...
    imagenet_folder: Path to ImageNet folder
    test_type: Test type (1 or 2)
    output: Output directory
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Set

import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.targeted.two_targets.batch_pipeline_multiprocessing import run_batch_attacks

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_images(folder_path: str) -> List[str]:
    """Find all image files in the folder."""
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = []

    folder = Path(folder_path)
    if not folder.exists():
        return []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(str(Path(root) / file))

    return image_paths


def load_used_images(output_dir: Path) -> Set[str]:
    """Load the set of already used image paths."""
    used_images_file = output_dir / "used_images_two_targets.json"
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
    used_images_file = output_dir / "used_images_two_targets.json"
    try:
        with open(used_images_file, "w") as f:
            json.dump(list(used_images), f, indent=2)
        logger.info(f"Saved {len(used_images)} used image paths to {used_images_file}")
    except IOError as e:
        logger.error(f"Could not save used images file: {e}")


def filter_unused_images(all_images: List[str], used_images: Set[str]) -> List[str]:
    """Filter out already used images from the list."""
    unused_images = [img for img in all_images if img not in used_images]
    logger.info(
        f"Filtered images: {len(all_images)} total, {len(used_images)} used, {len(unused_images)} available"
    )
    return unused_images


def main():
    parser = argparse.ArgumentParser(description="Run untargeted adversarial attacks")
    parser.add_argument(
        "epsilons", help="Comma-separated epsilon values (e.g., '8.0,16.0')"
    )
    parser.add_argument(
        "categories1", help="Comma-separated categories (e.g., 'cat,dog')"
    )
    parser.add_argument(
        "categories2", help="Comma-separated categories (e.g., 'cat,dog')"
    )
    parser.add_argument("imagenet_folder", help="Path to ImageNet folder")
    parser.add_argument(
        "test_type", type=int, choices=[1, 2], help="Test type (1 or 2)"
    )
    parser.add_argument("output", help="Output directory")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--target_successes",
        type=int,
        default=5,
        help="Target number of successful attacks per category (default: 5)",
    )

    args = parser.parse_args()

    # Parse arguments
    epsilons = [float(eps.strip()) for eps in args.epsilons.split(",")]
    # generate a list of pairs of categories
    categories = [
        (x, y) for x, y in zip(args.categories1.split(","), args.categories2.split(","))
    ]

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Check if mini-ImageNet dataset exists
    if not args.imagenet_folder:
        logger.error(
            "Please provide a path to the ImageNet folder. If not downloaded, please run: python src/download_images.py"
        )
        return

    mini_imagenet_path = Path(args.imagenet_folder)

    if not mini_imagenet_path.exists():
        logger.error(
            f"Mini-ImageNet dataset not found at {mini_imagenet_path}. Please run: python src/download_images.py"
        )
        return

    synsets = [d.name for d in mini_imagenet_path.iterdir() if d.is_dir()]

    images = []

    for synset in synsets:
        synset_dir = mini_imagenet_path / synset
        if synset_dir.exists():
            synset_images = (
                list(synset_dir.glob("*.JPEG"))
                + list(synset_dir.glob("*.jpg"))
                + list(synset_dir.glob("*.png"))
            )
            images.extend([str(img) for img in synset_images])

    np.random.shuffle(images)
    logger.info(f"Total images: {len(images)}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load used images and filter out already processed ones
    used_images = load_used_images(output_dir)
    images = filter_unused_images(images, used_images)

    # Run pipeline
    logger.info("Starting attack pipeline...")
    logger.info(f"Epsilons: {epsilons}")
    logger.info(f"Categories: {categories}")
    logger.info(f"Test type: {args.test_type}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Target successful attacks per category: {args.target_successes}")

    # Process each category until target successes are reached
    overall_summary = {
        "total_images": 0,
        "total_tasks": 0,
        "successful_attacks": 0,
        "failed_validations": 0,
        "failed_tests": 0,
        "success_rate": 0.0,
        "successful_results": [],
        "category_results": {},
    }

    # Track total iterations for periodic saving
    total_iterations = 0

    for category1, category2 in categories:
        image_paths = images

        logger.info(
            f"Target: {args.target_successes} successful attacks for {category1}, {category2}"
        )

        categories_successes = 0
        processed_images = 0
        batch_start_idx = 0

        # Process images in batches until we reach target successes or run out of images
        while categories_successes < args.target_successes and batch_start_idx < len(
            image_paths
        ):
            # Determine batch size for this iteration
            remaining_images = len(image_paths) - batch_start_idx
            current_batch_size = min(args.batch_size, remaining_images)

            # Get current batch of images
            batch_image_paths = image_paths[
                batch_start_idx : batch_start_idx + current_batch_size
            ]
            batch_fine_class_ids = [-1] * len(batch_image_paths)
            batch_target_coarse_classes_1 = [category1] * len(batch_image_paths)
            batch_target_coarse_classes_2 = [category2] * len(batch_image_paths)

            logger.info(
                f"Processing batch of {len(batch_image_paths)} images for {category1}, {category2} "
                f"(images {batch_start_idx + 1}-{batch_start_idx + len(batch_image_paths)} of {len(image_paths)})"
            )

            # Run batch processing for this subset
            batch_results = run_batch_attacks(
                image_paths=batch_image_paths,
                fine_class_ids=batch_fine_class_ids,
                targeted_coarse_classes_1=batch_target_coarse_classes_1,
                targeted_coarse_classes_2=batch_target_coarse_classes_2,
                epsilons=epsilons,
                test_types=[args.test_type],
                output_base_dir=output_dir,
                batch_size=args.batch_size,
            )

            # Update counters
            batch_successes = batch_results["successful_attacks"]
            categories_successes += batch_successes
            processed_images += len(batch_image_paths)

            # Add processed images to used_images set
            for img_path in batch_image_paths:
                used_images.add(img_path)

            # Update overall summary
            overall_summary["total_images"] += batch_results["total_images"]
            overall_summary["total_tasks"] += batch_results["total_tasks"]
            overall_summary["successful_attacks"] += batch_results["successful_attacks"]
            overall_summary["failed_validations"] += batch_results["failed_validations"]
            overall_summary["failed_tests"] += batch_results["failed_tests"]
            overall_summary["successful_results"].extend(
                batch_results["successful_results"]
            )

            # Increment total iterations and save used images every 100 iterations
            total_iterations += 1
            if total_iterations % 100 == 0:
                save_used_images(used_images, output_dir)
                logger.info(
                    f"Checkpoint: Saved used images list at iteration {total_iterations}"
                )

            logger.info(
                f"Batch complete: {batch_successes} successes. "
                f"Category total: {categories_successes}/{args.target_successes}"
            )

            # Move to next batch
            batch_start_idx += current_batch_size

            # Check if we've reached our target
            if categories_successes >= args.target_successes:
                logger.info(
                    f"✅ Target reached for {category1}, {category2}: {categories_successes} successful attacks"
                )
                break

        # Store category results
        overall_summary["category_results"][f"{category1}-{category2}"] = {
            "target": args.target_successes,
            "achieved": categories_successes,
            "processed_images": processed_images,
            "total_available": len(image_paths),
            "success_rate": (categories_successes / processed_images * 100)
            if processed_images > 0
            else 0,
        }

        if categories_successes < args.target_successes:
            logger.warning(
                f"⚠️  Could not reach target for {category1}, {category2}: "
                f"{categories_successes}/{args.target_successes} successful attacks "
                f"(processed all {len(image_paths)} available images)"
            )

    # Calculate overall success rate
    if overall_summary["total_tasks"] > 0:
        overall_summary["success_rate"] = (
            overall_summary["successful_attacks"] * 100
        ) / overall_summary["total_tasks"]

    results_summary = overall_summary

    # Final save of used images
    save_used_images(used_images, output_dir)

    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"Total images processed: {results_summary['total_images']}")
    logger.info(f"Total tasks: {results_summary['total_tasks']}")
    logger.info(f"Successful attacks: {results_summary['successful_attacks']}")
    logger.info(f"Failed validations: {results_summary['failed_validations']}")
    logger.info(f"Failed tests: {results_summary['failed_tests']}")
    logger.info(f"Overall success rate: {results_summary['success_rate']:.1f}%")

    logger.info("\n=== CATEGORY RESULTS ===")
    for category, cat_results in results_summary["category_results"].items():
        status = (
            "✅ COMPLETED"
            if cat_results["achieved"] >= cat_results["target"]
            else "⚠️  INCOMPLETE"
        )
        logger.info(f"{category}: {status}")
        logger.info(f"  Target: {cat_results['target']} successful attacks")
        logger.info(f"  Achieved: {cat_results['achieved']} successful attacks")
        logger.info(
            f"  Processed: {cat_results['processed_images']}/{cat_results['total_available']} images"
        )
        logger.info(f"  Category success rate: {cat_results['success_rate']:.1f}%")


if __name__ == "__main__":
    main()
