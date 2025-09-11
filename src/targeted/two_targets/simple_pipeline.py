#!/usr/bin/env python3
"""
Simple Pipeline Integration

Runs validation -> generation -> testing -> save (only if successful)
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ...utils import load_image
from .gen import AdversarialGenerator
from .test import test_adversarial_tensors_multiple
from .val import ImageValidator

logger = logging.getLogger(__name__)


def save_tensor_as_image(tensor: torch.Tensor, save_path: str):
    """Save tensor as image file."""
    image_np = tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    Image.fromarray(image_np).save(save_path)


def run_complete_attack_pipeline(
    image_path: str,
    fine_class_id: int,
    targeted_coarse_class_1: str,
    targeted_coarse_class_2: str,
    epsilon: float,
    test_type: int,
    output_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[bool, dict[str, Any], str]:
    """
    Run complete pipeline: validate -> generate -> test -> save (only if successful)
    """

    # Step 1: Validate
    validator = ImageValidator(device)
    val_success, val_results = validator.validate_image(
        image_path,
        fine_class_id,
        targeted_coarse_class_1,
        targeted_coarse_class_2,
        test_type,
    )

    if not val_success:
        return False, val_results, ""

    # Step 2: Generate attacks (3 variants for each target using top 3 categories)
    generator = AdversarialGenerator(device)
    original_image = load_image(image_path, device)

    targeted_images_1 = generator.generate_targeted_attacks_top3(
        original_image, targeted_coarse_class_1, epsilon
    )
    targeted_images_2 = generator.generate_targeted_attacks_top3(
        original_image, targeted_coarse_class_2, epsilon
    )

    # Step 3: Test attacks (any successful combination passes)
    test_success, test_results = test_adversarial_tensors_multiple(
        test_type,
        targeted_images_1,
        targeted_images_2,
        targeted_coarse_class_1,
        targeted_coarse_class_2,
        device,
    )

    if not test_success:
        return False, test_results, ""

    # Step 4: Save results (only if attack succeeded)
    image_name = Path(image_path).stem
    attack_folder = output_dir / f"{image_name}_eps{epsilon}_test{test_type}"
    attack_folder.mkdir(parents=True, exist_ok=True)

    # Save images (save the successful combination)
    save_tensor_as_image(original_image, str(attack_folder / "original.png"))

    if (
        "targeted_image_1_used" in test_results
        and "targeted_image_2_used" in test_results
    ):
        # Save the successful combination
        save_tensor_as_image(
            test_results["targeted_image_1_used"], str(attack_folder / "targeted_1.png")
        )
        save_tensor_as_image(
            test_results["targeted_image_2_used"], str(attack_folder / "targeted_2.png")
        )

        # Also save all variants for analysis
        for i, targeted_img_1 in enumerate(targeted_images_1):
            save_tensor_as_image(
                targeted_img_1, str(attack_folder / f"targeted_1_variant_{i}.png")
            )
        for i, targeted_img_2 in enumerate(targeted_images_2):
            save_tensor_as_image(
                targeted_img_2, str(attack_folder / f"targeted_2_variant_{i}.png")
            )
    else:
        # Fallback for legacy results
        save_tensor_as_image(
            targeted_images_1[0], str(attack_folder / "targeted_1.png")
        )
        save_tensor_as_image(
            targeted_images_2[0], str(attack_folder / "targeted_2.png")
        )

    # Save metadata
    metadata = {
        "image_path": image_path,
        "fine_class_id": fine_class_id,
        "targeted_coarse_class_1": targeted_coarse_class_1,
        "targeted_coarse_class_2": targeted_coarse_class_2,
        "epsilon": epsilon,
        "test_type": test_type,
        "validation_results": val_results,
        "test_results": test_results,
    }

    with open(attack_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return True, test_results, str(attack_folder)
