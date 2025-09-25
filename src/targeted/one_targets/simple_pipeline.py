#!/usr/bin/env python3
"""
Simple Pipeline Integration

Runs validation -> generation -> testing -> save (only if successful)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ...utils import get_ensemble_logits, load_image
from .gen import AdversarialGenerator
from .test import AdversarialTester, test_adversarial_tensors_multiple
from .val import ImageValidator

logger = logging.getLogger(__name__)


def save_tensor_as_image(tensor: torch.Tensor, save_path: str):
    """Save tensor as image file."""
    image_np = tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    Image.fromarray(image_np).save(save_path)


def _save_comprehensive_results(
    attack_folder: Path,
    test_type: int,
    original_image: torch.Tensor,
    targeted_image: torch.Tensor,
    control_image: torch.Tensor,
    targeted_logits: torch.Tensor,
    control_logits: torch.Tensor,
    test_results: dict,
    image_path: str,
    targeted_coarse_class: str,
    epsilon: float,
    tester: AdversarialTester,
):
    """Save comprehensive results including all ensemble predictions."""

    # Save images
    save_tensor_as_image(original_image, str(attack_folder / "original.png"))
    save_tensor_as_image(targeted_image, str(attack_folder / "targeted.png"))
    save_tensor_as_image(control_image, str(attack_folder / "control.png"))

    # Get original image logits and predictions
    original_logits = get_ensemble_logits(
        tester.normalize(original_image), tester.models
    )
    original_top_indices, original_top_probs = tester.get_top_predictions(
        original_logits, top_k=1000
    )

    # Get targeted and control predictions
    targeted_top_indices, targeted_top_probs = tester.get_top_predictions(
        targeted_logits, top_k=1000
    )
    control_top_indices, control_top_probs = tester.get_top_predictions(
        control_logits, top_k=1000
    )

    # Create class name mappings
    original_class_probs = {
        tester.class_names.get(idx, f"class_{idx}"): prob
        for idx, prob in zip(original_top_indices, original_top_probs)
    }
    targeted_class_probs = {
        tester.class_names.get(idx, f"class_{idx}"): prob
        for idx, prob in zip(targeted_top_indices, targeted_top_probs)
    }
    control_class_probs = {
        tester.class_names.get(idx, f"class_{idx}"): prob
        for idx, prob in zip(control_top_indices, control_top_probs)
    }

    # Create output dictionaries
    original_output = {
        "logits": original_logits.cpu().numpy().tolist(),
        "class_probabilities": original_class_probs,
        "top_indices": original_top_indices,
        "top_probs": original_top_probs,
    }

    targeted_output = {
        "logits": targeted_logits.cpu().numpy().tolist(),
        "class_probabilities": targeted_class_probs,
        "top_indices": targeted_top_indices,
        "top_probs": targeted_top_probs,
    }

    control_output = {
        "logits": control_logits.cpu().numpy().tolist(),
        "class_probabilities": control_class_probs,
        "top_indices": control_top_indices,
        "top_probs": control_top_probs,
    }

    # Add test-specific information to outputs
    if test_type == 1:
        if "targeted_top_indices" in test_results:
            targeted_output.update(
                {
                    "top_prediction": test_results.get("targeted_top_prediction"),
                    "targeted_coarse_indices": test_results.get(
                        "targeted_coarse_indices"
                    ),
                }
            )
        if "control_top_indices" in test_results:
            control_output.update(
                {
                    "top_prediction": test_results.get("control_top_prediction"),
                    "targeted_coarse_indices": test_results.get(
                        "targeted_coarse_indices"
                    ),
                }
            )
    elif test_type == 2:
        if "targeted_coarse_score (Sc)" in test_results:
            targeted_output.update(
                {
                    "coarse_score_Sc": test_results["targeted_coarse_score (Sc)"],
                    "top_prediction": test_results.get("targeted_top_prediction"),
                    "targeted_coarse_indices": test_results.get(
                        "targeted_coarse_indices"
                    ),
                }
            )
        if "control_coarse_score (Sc)" in test_results:
            control_output.update(
                {
                    "coarse_score_Sc": test_results["control_coarse_score (Sc)"],
                    "top_prediction": test_results.get("control_top_prediction"),
                    "targeted_coarse_indices": test_results.get(
                        "targeted_coarse_indices"
                    ),
                }
            )

    # Save detailed ensemble prediction JSON files
    with open(attack_folder / "original_ensemble_output.json", "w") as f:
        json.dump(original_output, f, indent=2)

    with open(attack_folder / "targeted_ensemble_output.json", "w") as f:
        json.dump(targeted_output, f, indent=2)

    with open(attack_folder / "control_ensemble_output.json", "w") as f:
        json.dump(control_output, f, indent=2)

    # Create comprehensive metadata
    metadata = {
        "image_path": image_path,
        "targeted_coarse_class": targeted_coarse_class,
        "epsilon": epsilon,
        "test_type": test_type,
        "creation_date": datetime.now().isoformat(),
        "successful_variant_used": test_results.get("successful_variant_used"),
        "total_variants_tested": test_results.get("total_variants_tested"),
        "successful_variants": test_results.get("successful_variants", []),
    }

    with open(attack_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Saved comprehensive results with ensemble predictions to: {attack_folder}"
    )


def run_complete_attack_pipeline(
    image_path: str,
    fine_class_id: int,
    targeted_coarse_class: str,
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
        image_path, fine_class_id, targeted_coarse_class, test_type
    )

    if not val_success:
        return False, val_results, ""

    # Step 2: Generate attacks (3 variants using top 3 categories)
    generator = AdversarialGenerator(device)
    original_image = load_image(image_path, device)

    targeted_images = generator.generate_targeted_attacks_top3(
        original_image, targeted_coarse_class, epsilon
    )

    control_images = []
    for targeted_image in targeted_images:
        control_image = generator.generate_control_image_from_targeted_attack(
            original_image, targeted_image, epsilon
        )
        control_images.append(control_image)

    # Step 3: Test attacks (any successful variant passes)
    test_success, test_results = test_adversarial_tensors_multiple(
        test_type, targeted_images, control_images, targeted_coarse_class, device
    )

    if not test_success:
        return False, test_results, ""

    # Step 4: Save results with detailed ensemble predictions (only if attack succeeded)
    # Create the attack folder with our naming convention
    image_name = Path(image_path).stem
    attack_folder = output_dir / f"{image_name}_eps{epsilon}_test{test_type}"
    attack_folder.mkdir(parents=True, exist_ok=True)

    # Initialize tester for comprehensive saving
    tester = AdversarialTester(device)

    # Get the successful variant images and corresponding logits
    if "targeted_image_used" in test_results and "control_image_used" in test_results:
        # Use the successful variant
        targeted_image_to_save = test_results["targeted_image_used"]
        control_image_to_save = test_results["control_image_used"]

        # Compute logits for the successful variant
        targeted_logits = get_ensemble_logits(
            tester.normalize(targeted_image_to_save), tester.models
        )
        control_logits = get_ensemble_logits(
            tester.normalize(control_image_to_save), tester.models
        )
    else:
        # Fallback for legacy results - use first variant
        targeted_image_to_save = targeted_images[0]
        control_image_to_save = control_images[0]

        targeted_logits = get_ensemble_logits(
            tester.normalize(targeted_image_to_save), tester.models
        )
        control_logits = get_ensemble_logits(
            tester.normalize(control_image_to_save), tester.models
        )

    # Save all ensemble prediction files directly to our attack folder
    _save_comprehensive_results(
        attack_folder,
        test_type,
        original_image,
        targeted_image_to_save,
        control_image_to_save,
        targeted_logits,
        control_logits,
        test_results,
        image_path,
        targeted_coarse_class,
        epsilon,
        tester,
    )

    return True, test_results, str(attack_folder)
