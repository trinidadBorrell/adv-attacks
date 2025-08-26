#!/usr/bin/env python3
"""
Untargeted Attack Runner Script

This script runs untargeted adversarial attacks on a set of images with different
epsilons and categories. It validates each image first, then generates attacks
and tests them if validation passes.

Usage:
    python run_untargeted.py <epsilons> <categories> <imagenet_folder_path> <test_type> <output>

Arguments:
    epsilons: Comma-separated list of epsilon values (e.g., "8.0,16.0,32.0")
    categories: Comma-separated list of coarse categories (e.g., "fish,bird,mammal")
    imagenet_folder_path: Path to the miniImageNet folder (structure: miniImageNet/{folder}/{image_files..})
    test_type: Test type to use (1 or 2)
    output: Output base directory
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def log_message(message: str):
    """Log message with timestamp."""
    logger.info(message)


def find_images(folder_path: str) -> List[str]:
    """Find all image files in the folder and its subfolders."""
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = []

    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        return []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(folder):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(str(Path(root) / file))

    logger.info(f"Found {len(image_paths)} images in {folder_path}")
    return image_paths


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    try:
        log_message(f"Running: {description}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log_message(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"✗ {description} failed: {e}")
        if e.stdout:
            log_message(f"stdout: {e.stdout}")
        if e.stderr:
            log_message(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        log_message(f"✗ {description} failed with exception: {e}")
        return False


def main():
    """Main function to run untargeted attacks."""

    parser = argparse.ArgumentParser(description="Run untargeted adversarial attacks")
    parser.add_argument(
        "epsilons", help="Comma-separated list of epsilon values (e.g., '8.0,16.0')"
    )
    parser.add_argument(
        "categories",
        help="Comma-separated list of coarse categories (e.g., 'fish,bird')",
    )
    parser.add_argument("imagenet_folder_path", help="Path to the miniImageNet folder")
    parser.add_argument(
        "test_type", type=int, choices=[1, 2], help="Test type to use (1 or 2)"
    )
    parser.add_argument("output", help="Output base directory")

    args = parser.parse_args()

    # Parse comma-separated arguments
    epsilons = [float(eps.strip()) for eps in args.epsilons.split(",")]
    categories = [cat.strip() for cat in args.categories.split(",")]

    # Get project root
    project_root = Path(__file__).parent.parent

    # Check if required files exist
    val_script = project_root / "src" / "untargeted" / "val.py"
    gen_script = project_root / "src" / "untargeted" / "gen.py"
    test_script = project_root / "src" / "untargeted" / "test.py"

    for script_path in [val_script, gen_script, test_script]:
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            sys.exit(1)

    # Check if miniImageNet folder exists
    if not Path(args.imagenet_folder_path).exists():
        logger.error(f"miniImageNet folder not found: {args.imagenet_folder_path}")
        sys.exit(1)

    # Create output directory
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    # Log start
    log_message("Starting untargeted attack runner")
    log_message(f"Epsilons: {epsilons}")
    log_message(f"Categories: {categories}")
    log_message(f"miniImageNet folder: {args.imagenet_folder_path}")
    log_message(f"Test type: {args.test_type}")
    log_message(f"Output base: {args.output}")

    # Statistics
    total_images = 0
    valid_images = 0
    successful_attacks = 0

    # Find all images
    image_paths = find_images(args.imagenet_folder_path)

    for eps in epsilons:
        log_message(f"Processing epsilon: {eps}")

        for category in categories:
            log_message(f"Processing category: {category}")

            # Create category-specific output directory
            category_output = output_base / "exp2" / category / f"epsilon_{eps}"
            category_output.mkdir(parents=True, exist_ok=True)

            for image_path in image_paths:
                total_images += 1

                # Get image name without extension
                image_name = Path(image_path).stem

                log_message(f"Processing image: {image_name}")

                # Step 1: Validate image
                val_cmd = [sys.executable, str(val_script), image_path, "0", category]

                if run_command(val_cmd, f"Validating {image_name}"):
                    valid_images += 1
                    log_message(f"Validation PASSED for {image_name}")

                    # Create image-specific output directory ONLY after validation succeeds
                    image_output = category_output / image_name
                    image_output.mkdir(exist_ok=True)

                    # Step 2: Generate untargeted attack
                    gen_cmd = [
                        sys.executable,
                        str(gen_script),
                        image_path,
                        "0",
                        category,
                        str(eps),
                    ]

                    if run_command(
                        gen_cmd, f"Generating untargeted attack for {image_name}"
                    ):
                        log_message("Untargeted attack generated successfully")

                        # Step 3: Generate targeted attack
                        gen_targeted_cmd = [
                            sys.executable,
                            str(gen_script),
                            image_path,
                            "0",
                            category,
                            str(eps),
                            "--targeted",
                        ]

                        if run_command(
                            gen_targeted_cmd,
                            f"Generating targeted attack for {image_name}",
                        ):
                            log_message("Targeted attack generated successfully")

                            # Step 4: Test attacks
                            test_cmd = [
                                sys.executable,
                                str(test_script),
                                str(args.test_type),
                                image_path,
                                "untargeted.png",
                                "targeted.png",
                                "0",
                                category,
                                str(eps),
                            ]

                            if run_command(
                                test_cmd, f"Testing attacks for {image_name}"
                            ):
                                successful_attacks += 1
                                log_message(f"Test PASSED for {image_name}")
                            else:
                                log_message(f"Test FAILED for {image_name}")
                        else:
                            log_message(
                                f"Targeted attack generation FAILED for {image_name}"
                            )
                    else:
                        log_message(
                            f"Untargeted attack generation FAILED for {image_name}"
                        )
                else:
                    log_message(
                        f"Validation FAILED for {image_name} - skipping to next image"
                    )

                log_message(f"Completed processing {image_name}")
                log_message("---")

    # Final summary
    log_message("=== FINAL SUMMARY ===")
    log_message(f"Total images processed: {total_images}")
    log_message(f"Valid images: {valid_images}")
    log_message(f"Successful attacks: {successful_attacks}")
    if total_images > 0:
        success_rate = (successful_attacks * 100) / total_images
        log_message(f"Success rate: {success_rate:.2f}%")
    log_message(f"Results saved to: {output_base}/exp2/")

    log_message("Untargeted attack runner completed successfully!")


if __name__ == "__main__":
    main()
