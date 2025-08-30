#!/usr/bin/env python3
"""
Simple Pipeline Integration

Runs validation -> generation -> testing -> save (only if successful)
"""

import logging
import json
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image

from .untargeted.val import ImageValidator
from .untargeted.gen import AdversarialGenerator
from .untargeted.test import test_adversarial_tensors
from .utils import load_image

logger = logging.getLogger(__name__)


def save_tensor_as_image(tensor: torch.Tensor, save_path: str):
    """Save tensor as image file."""
    image_np = tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    Image.fromarray(image_np).save(save_path)


def run_complete_attack_pipeline(
    image_path: str,
    fine_class_id: int,
    coarse_class: str,
    epsilon: float,
    test_type: int,
    output_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple[bool, dict[str, Any], str]:
    """
    Run complete pipeline: validate -> generate -> test -> save (only if successful)
    """
    
    # Step 1: Validate
    validator = ImageValidator(device)
    val_success, val_results = validator.validate_image(image_path, fine_class_id, coarse_class, test_type)
    
    if not val_success:
        return False, val_results, ""
    
    # Step 2: Generate attacks
    generator = AdversarialGenerator(device)
    original_image = load_image(image_path, device)
    
    untargeted_image = generator.generate_untargeted_attack(original_image, fine_class_id, epsilon)
    targeted_image = generator.generate_targeted_attack(original_image, coarse_class, epsilon)
    
    # Step 3: Test attacks
    test_success, test_results = test_adversarial_tensors(
        test_type, original_image, untargeted_image, targeted_image, coarse_class, device
    )
    
    if not test_success:
        return False, test_results, ""
    
    # Step 4: Save results (only if attack succeeded)
    image_name = Path(image_path).stem
    attack_folder = output_dir / f"{image_name}_eps{epsilon}_test{test_type}"
    attack_folder.mkdir(parents=True, exist_ok=True)
    
    # Save images
    save_tensor_as_image(original_image, str(attack_folder / "original.png"))
    save_tensor_as_image(untargeted_image, str(attack_folder / "untargeted.png"))
    save_tensor_as_image(targeted_image, str(attack_folder / "targeted.png"))
    
    # Save metadata
    metadata = {
        "image_path": image_path,
        "fine_class_id": fine_class_id,
        "coarse_class": coarse_class,
        "epsilon": epsilon,
        "test_type": test_type,
        "validation_results": val_results,
        "test_results": test_results
    }
    
    with open(attack_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return True, test_results, str(attack_folder)
