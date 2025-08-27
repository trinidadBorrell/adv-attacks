#!/usr/bin/env python3
"""
Pipeline Integration Module

This module provides integrated functions that handle the complete adversarial attack pipeline
using tensors internally and only saving files after successful tests.
"""

import logging
from pathlib import Path
from typing import Any, Tuple

import torch
import numpy as np
from PIL import Image

# Import from other modules
from .untargeted.val import ImageValidator
from .untargeted.gen import AdversarialGenerator
from .untargeted.test import test_adversarial_tensors
from .utils import load_image

# Setup logging
logger = logging.getLogger(__name__)


def save_tensor_as_image(tensor: torch.Tensor, save_path: str):
    """Save tensor as image file."""
    # Convert tensor to numpy array
    image_np = tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    # Denormalize and convert to uint8
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    # Save as image
    Image.fromarray(image_np).save(save_path)
    logger.info(f"Saved image to: {save_path}")


def run_complete_attack_pipeline(
    image_path: str,
    fine_class_id: int,
    coarse_class: str,
    epsilon: float,
    test_type: int,
    output_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[bool, dict[str, Any], str]:
    """
    Run the complete adversarial attack pipeline with tensor-based processing.
    Only saves files if the attack is successful.
    
    Returns:
        - success: Whether the attack was successful
        - results: Test results dictionary
        - output_folder: Path to saved results (empty if unsuccessful)
    """
    
    logger.info(f"Starting complete pipeline for {Path(image_path).name}")
    
    try:
        # Step 1: Validate image
        logger.info("Step 1: Validating image...")
        validator = ImageValidator(device)
        val_success, val_results = validator.validate_image(
            image_path, fine_class_id, coarse_class
        )
        
        if not val_success:
            logger.info("Validation FAILED - skipping attack generation")
            return False, val_results, ""
        
        logger.info("Validation PASSED - proceeding with attack generation")
        
        # Step 2: Generate adversarial attacks (returns tensors)
        logger.info("Step 2: Generating adversarial attacks...")
        generator = AdversarialGenerator(device)
        original_image = load_image(image_path, device)
        
        untargeted_image = generator.generate_untargeted_attack(
            original_image, fine_class_id, epsilon
        )
        targeted_image = generator.generate_targeted_attack(
            original_image, coarse_class, epsilon
        )
        
        logger.info("Attack generation completed")
        
        # Step 3: Test attacks (using tensors directly)
        logger.info("Step 3: Testing attacks...")
        test_success, test_results = test_adversarial_tensors(
            test_type, original_image, untargeted_image, targeted_image, coarse_class, device
        )
        
        if not test_success:
            logger.info("Attack test FAILED - not saving images")
            return False, test_results, ""
        
        logger.info("Attack test PASSED - saving successful attack images")
        
        # Step 4: Save images only after successful test
        image_name = Path(image_path).stem
        attack_folder = output_dir / f"{image_name}_eps{epsilon}_test{test_type}"
        attack_folder.mkdir(parents=True, exist_ok=True)
        
        # Save original image (copy)
        original_save_path = attack_folder / "original.png"
        save_tensor_as_image(original_image, str(original_save_path))
        
        # Save adversarial images
        untargeted_save_path = attack_folder / "untargeted.png"
        targeted_save_path = attack_folder / "targeted.png"
        
        save_tensor_as_image(untargeted_image, str(untargeted_save_path))
        save_tensor_as_image(targeted_image, str(targeted_save_path))
        
        # Calculate and save metadata
        untargeted_norm = torch.norm(untargeted_image - original_image, p=float("inf")).item()
        targeted_norm = torch.norm(targeted_image - original_image, p=float("inf")).item()
        
        metadata = {
            "image_path": image_path,
            "fine_class_id": fine_class_id,
            "coarse_class": coarse_class,
            "epsilon": epsilon,
            "test_type": test_type,
            "untargeted_norm": untargeted_norm,
            "targeted_norm": targeted_norm,
            "validation_results": val_results,
            "test_results": test_results
        }
        
        import json
        with open(attack_folder / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successful attack saved to: {attack_folder}")
        return True, test_results, str(attack_folder)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return False, {"error": str(e)}, ""


def run_batch_attacks(
    image_paths: list[str],
    fine_class_ids: list[int],
    coarse_classes: list[str],
    epsilons: list[float],
    test_types: list[int],
    output_base_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict[str, Any]:
    """
    Run adversarial attacks on a batch of images.
    
    Returns summary statistics.
    """
    
    total_images = len(image_paths)
    successful_attacks = 0
    results_summary = {
        "total_images": total_images,
        "successful_attacks": 0,
        "failed_validations": 0,
        "failed_tests": 0,
        "success_rate": 0.0,
        "successful_results": []
    }
    
    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing image {i+1}/{total_images}: {Path(image_path).name}")
        
        for epsilon in epsilons:
            for test_type in test_types:
                for j, (fine_class_id, coarse_class) in enumerate(zip(fine_class_ids, coarse_classes)):
                    
                    success, results, output_folder = run_complete_attack_pipeline(
                        image_path, fine_class_id, coarse_class, epsilon, test_type, 
                        output_base_dir, device
                    )
                    
                    if success:
                        successful_attacks += 1
                        results_summary["successful_results"].append({
                            "image_path": image_path,
                            "fine_class_id": fine_class_id,
                            "coarse_class": coarse_class,
                            "epsilon": epsilon,
                            "test_type": test_type,
                            "output_folder": output_folder
                        })
                    elif "validation_results" in results and not results.get("overall_success", True):
                        results_summary["failed_validations"] += 1
                    else:
                        results_summary["failed_tests"] += 1
    
    results_summary["successful_attacks"] = successful_attacks
    if total_images > 0:
        results_summary["success_rate"] = (successful_attacks * 100) / total_images
    
    return results_summary
