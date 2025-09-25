#!/usr/bin/env python3
"""
Pipeline Integration Module

This module provides integrated functions that handle the complete adversarial attack pipeline
using tensors internally and only saving files after successful tests.
"""

import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import torch

from .simple_pipeline import run_complete_attack_pipeline

# Setup logging
logger = logging.getLogger(__name__)


def _process_single_attack(
    args: tuple[str, int, str, float, int, Path, str],
) -> tuple[bool, dict, str, dict]:
    """
    Worker function for parallel processing of single attack.
    Returns (success, results, output_folder, metadata) where metadata contains input parameters.
    """
    (
        image_path,
        fine_class_id,
        targeted_coarse_class,
        epsilon,
        test_type,
        output_base_dir,
        device,
    ) = args

    # Create metadata for tracking
    metadata = {
        "image_path": image_path,
        "fine_class_id": fine_class_id,
        "targeted_coarse_class": targeted_coarse_class,
        "epsilon": epsilon,
        "test_type": test_type,
    }

    try:
        success, results, output_folder = run_complete_attack_pipeline(
            image_path,
            fine_class_id,
            targeted_coarse_class,
            epsilon,
            test_type,
            output_base_dir,
            device,
        )
        return success, results, output_folder, metadata
    except Exception as e:
        logger.error(f"Error processing {Path(image_path).name}: {e}")
        return False, {"error": str(e)}, "", metadata


def _process_image_all_combinations(
    args: tuple[str, int, str, list, list, Path, str],
) -> tuple[bool, dict, str, list]:
    """
    Worker function for parallel processing of a single image with all epsilon/test_type combinations.
    Returns (overall_success, overall_results, last_output_folder, metadata_list)
    """
    (
        image_path,
        fine_class_id,
        targeted_coarse_class,
        epsilons,
        test_types,
        output_base_dir,
        device,
    ) = args

    metadata_list = []
    overall_success = False
    overall_results = {}
    last_output_folder = ""

    # Process all combinations for this image
    for epsilon in epsilons:
        for test_type in test_types:
            try:
                success, results, output_folder = run_complete_attack_pipeline(
                    image_path,
                    fine_class_id,
                    targeted_coarse_class,
                    epsilon,
                    test_type,
                    output_base_dir,
                    device,
                )

                # Create metadata for this combination
                metadata = {
                    "image_path": image_path,
                    "fine_class_id": fine_class_id,
                    "targeted_coarse_class": targeted_coarse_class,
                    "epsilon": epsilon,
                    "test_type": test_type,
                    "success": success,
                    "output_folder": output_folder if success else "",
                    "validation_failed": False,
                }

                # Check if validation failed
                if (
                    not success
                    and "validation_results" in results
                    and not results.get("overall_success", True)
                ):
                    metadata["validation_failed"] = True

                metadata_list.append(metadata)

                if success:
                    overall_success = True
                    last_output_folder = output_folder

            except Exception as e:
                logger.error(
                    f"Error processing {Path(image_path).name} with eps={epsilon}, test={test_type}: {e}"
                )
                metadata = {
                    "image_path": image_path,
                    "fine_class_id": fine_class_id,
                    "targeted_coarse_class": targeted_coarse_class,
                    "epsilon": epsilon,
                    "test_type": test_type,
                    "success": False,
                    "output_folder": "",
                    "validation_failed": False,
                    "error": str(e),
                }
                metadata_list.append(metadata)

    return overall_success, overall_results, last_output_folder, metadata_list


def run_batch_attacks(
    image_paths: list[str],
    fine_class_ids: list[int],
    targeted_coarse_classes: list[str],
    epsilons: list[float],
    test_types: list[int],
    output_base_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    Run adversarial attacks on a batch of images.

    Args:
        batch_size: Number of images to process in parallel (default 1 for sequential processing)

    Returns summary statistics.
    """

    # Prepare unique image tasks (one task per unique image)
    # Each task will process ALL epsilon/test_type combinations for that image
    unique_image_tasks = []
    for i, image_path in enumerate(image_paths):
        unique_image_tasks.append(
            (
                image_path,
                fine_class_ids[i],
                targeted_coarse_classes[i],
                epsilons,
                test_types,
                output_base_dir,
                device,
            )
        )

    total_images = len(unique_image_tasks)
    total_tasks = total_images * len(epsilons) * len(test_types)
    successful_attacks = 0
    results_summary = {
        "total_images": total_images,
        "total_tasks": total_tasks,
        "successful_attacks": 0,
        "failed_validations": 0,
        "failed_tests": 0,
        "success_rate": 0.0,
        "successful_results": [],
    }

    logger.info(
        f"Processing {total_images} unique images with {total_tasks} total tasks, batch_size={batch_size}"
    )

    if batch_size == 1:
        # Sequential processing (original behavior)
        for i, task_args in enumerate(unique_image_tasks):
            logger.info(
                f"Processing image {i + 1}/{total_images}: {Path(task_args[0]).name}"
            )
            success, results, output_folder, metadata_list = (
                _process_image_all_combinations(task_args)
            )

            # Process results from all combinations for this image
            for metadata in metadata_list:
                if metadata["success"]:
                    successful_attacks += 1
                    results_summary["successful_results"].append(
                        {
                            "image_path": metadata["image_path"],
                            "fine_class_id": metadata["fine_class_id"],
                            "coarse_class": metadata["targeted_coarse_class"],
                            "epsilon": metadata["epsilon"],
                            "test_type": metadata["test_type"],
                            "output_folder": metadata["output_folder"],
                        }
                    )
                elif metadata.get("validation_failed", False):
                    results_summary["failed_validations"] += 1
                else:
                    results_summary["failed_tests"] += 1
    else:
        # Parallel processing - each worker gets a unique image
        num_processes = min(batch_size, cpu_count(), total_images)
        logger.info(
            f"Using {num_processes} parallel processes for {total_images} unique images"
        )

        with Pool(processes=num_processes) as pool:
            # Process unique images in parallel
            results = pool.map(_process_image_all_combinations, unique_image_tasks)

            # Process results
            for i, (
                image_success,
                image_results,
                image_output_folder,
                metadata_list,
            ) in enumerate(results):
                if i % 5 == 0:  # Log progress every 5 images
                    logger.info(f"Processed {i + 1}/{total_images} images")

                # Process results from all combinations for this image
                for metadata in metadata_list:
                    if metadata["success"]:
                        successful_attacks += 1
                        results_summary["successful_results"].append(
                            {
                                "image_path": metadata["image_path"],
                                "fine_class_id": metadata["fine_class_id"],
                                "coarse_class": metadata["targeted_coarse_class"],
                                "epsilon": metadata["epsilon"],
                                "test_type": metadata["test_type"],
                                "output_folder": metadata["output_folder"],
                            }
                        )
                    elif metadata.get("validation_failed", False):
                        results_summary["failed_validations"] += 1
                    else:
                        results_summary["failed_tests"] += 1

    results_summary["successful_attacks"] = successful_attacks
    if total_tasks > 0:
        results_summary["success_rate"] = (successful_attacks * 100) / total_tasks

    logger.info(
        f"Batch processing complete: {successful_attacks}/{total_tasks} successful attacks"
    )
    return results_summary
