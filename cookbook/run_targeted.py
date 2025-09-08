#!/usr/bin/env python3
"""
Clean Untargeted Attack Pipeline

Usage:
    python run_targeted.py <epsilons> <categories> [imagenet_folder] <test_type> <output>
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.targeted.one_targets.batch_pipeline_multiprocessing import run_batch_attacks

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


def main():
    parser = argparse.ArgumentParser(description="Run untargeted adversarial attacks")
    parser.add_argument(
        "epsilons", help="Comma-separated epsilon values (e.g., '8.0,16.0')"
    )
    parser.add_argument(
        "categories", help="Comma-separated categories (e.g., 'cat,dog')"
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
    categories = [cat.strip() for cat in args.categories.split(",")]

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Check if mini-ImageNet dataset exists
    if not args.imagenet_folder:
        logger.error(
            "Please provide a path to the ImageNet folder. If not downloaded, please run: python src/download_images.py"
        )
        return

    mini_imagenet_path = Path(args.imagenet_folder) / "mini_imagenet"

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

    logger.info(f"Total images: {len(images)}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    for category in categories:
        image_paths = images

        logger.info(
            f"Target: {args.target_successes} successful attacks for {category}"
        )

        category_successes = 0
        processed_images = 0
        batch_start_idx = 0

        # Process images in batches until we reach target successes or run out of images
        while category_successes < args.target_successes and batch_start_idx < len(
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
            batch_target_coarse_classes = [category] * len(batch_image_paths)

            logger.info(
                f"Processing batch of {len(batch_image_paths)} images for {category} "
                f"(images {batch_start_idx + 1}-{batch_start_idx + len(batch_image_paths)} of {len(image_paths)})"
            )

            # Run batch processing for this subset
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
            processed_images += len(batch_image_paths)

            # Update overall summary
            overall_summary["total_images"] += batch_results["total_images"]
            overall_summary["total_tasks"] += batch_results["total_tasks"]
            overall_summary["successful_attacks"] += batch_results["successful_attacks"]
            overall_summary["failed_validations"] += batch_results["failed_validations"]
            overall_summary["failed_tests"] += batch_results["failed_tests"]
            overall_summary["successful_results"].extend(
                batch_results["successful_results"]
            )

            logger.info(
                f"Batch complete: {batch_successes} successes. "
                f"Category total: {category_successes}/{args.target_successes}"
            )

            # Move to next batch
            batch_start_idx += current_batch_size

            # Check if we've reached our target
            if category_successes >= args.target_successes:
                logger.info(
                    f"✅ Target reached for {category}: {category_successes} successful attacks"
                )
                break

        # Store category results
        overall_summary["category_results"][category] = {
            "target": args.target_successes,
            "achieved": category_successes,
            "processed_images": processed_images,
            "total_available": len(image_paths),
            "success_rate": (category_successes / processed_images * 100)
            if processed_images > 0
            else 0,
        }

        if category_successes < args.target_successes:
            logger.warning(
                f"⚠️  Could not reach target for {category}: "
                f"{category_successes}/{args.target_successes} successful attacks "
                f"(processed all {len(image_paths)} available images)"
            )

    # Calculate overall success rate
    if overall_summary["total_tasks"] > 0:
        overall_summary["success_rate"] = (
            overall_summary["successful_attacks"] * 100
        ) / overall_summary["total_tasks"]

    results_summary = overall_summary

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
