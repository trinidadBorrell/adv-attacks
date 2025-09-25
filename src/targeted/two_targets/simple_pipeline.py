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
    targeted_image_1: torch.Tensor,
    targeted_image_2: torch.Tensor,
    targeted_logits_1: torch.Tensor,
    targeted_logits_2: torch.Tensor,
    test_results: dict,
    image_path: str,
    targeted_coarse_class_1: str,
    targeted_coarse_class_2: str,
    epsilon: float,
    tester: AdversarialTester,
):
    """Save comprehensive results including all ensemble predictions."""

    # Save images
    save_tensor_as_image(original_image, str(attack_folder / "original.png"))
    save_tensor_as_image(targeted_image_1, str(attack_folder / "targeted_1.png"))
    save_tensor_as_image(targeted_image_2, str(attack_folder / "targeted_2.png"))

    # Get original image logits and predictions
    original_logits = get_ensemble_logits(
        tester.normalize(original_image), tester.models
    )
    original_top_indices, original_top_probs = tester.get_top_predictions(
        original_logits, top_k=1000
    )

    # Get targeted predictions
    targeted_top_indices_1, targeted_top_probs_1 = tester.get_top_predictions(
        targeted_logits_1, top_k=1000
    )
    targeted_top_indices_2, targeted_top_probs_2 = tester.get_top_predictions(
        targeted_logits_2, top_k=1000
    )

    # Create class name mappings
    original_class_probs = {
        tester.class_names.get(idx, f"class_{idx}"): prob
        for idx, prob in zip(original_top_indices, original_top_probs)
    }
    targeted_class_probs_1 = {
        tester.class_names.get(idx, f"class_{idx}"): prob
        for idx, prob in zip(targeted_top_indices_1, targeted_top_probs_1)
    }
    targeted_class_probs_2 = {
        tester.class_names.get(idx, f"class_{idx}"): prob
        for idx, prob in zip(targeted_top_indices_2, targeted_top_probs_2)
    }

    # Create output dictionaries
    original_output = {
        "logits": original_logits.cpu().numpy().tolist(),
        "class_probabilities": original_class_probs,
        "top_indices": original_top_indices,
        "top_probs": original_top_probs,
    }

    targeted_output_1 = {
        "logits": targeted_logits_1.cpu().numpy().tolist(),
        "class_probabilities": targeted_class_probs_1,
        "top_indices": targeted_top_indices_1,
        "top_probs": targeted_top_probs_1,
    }

    targeted_output_2 = {
        "logits": targeted_logits_2.cpu().numpy().tolist(),
        "class_probabilities": targeted_class_probs_2,
        "top_indices": targeted_top_indices_2,
        "top_probs": targeted_top_probs_2,
    }

    # Add test-specific information to outputs
    if test_type == 1:
        if "targeted_prediction_1" in test_results:
            targeted_output_1.update(
                {
                    "top_prediction": test_results["targeted_prediction_1"],
                }
            )
        if "targeted_prediction_2" in test_results:
            targeted_output_2.update(
                {
                    "top_prediction": test_results["targeted_prediction_2"],
                }
            )
    elif test_type == 2:
        if "targeted_prediction_1" in test_results:
            targeted_output_1.update(
                {
                    "top_prediction": test_results["targeted_prediction_1"],
                }
            )
        if "targeted_prediction_2" in test_results:
            targeted_output_2.update(
                {
                    "top_prediction": test_results["targeted_prediction_2"],
                }
            )

    # Save detailed ensemble prediction JSON files
    with open(attack_folder / "original_ensemble_output.json", "w") as f:
        json.dump(original_output, f, indent=2)

    with open(attack_folder / "targeted_1_ensemble_output.json", "w") as f:
        json.dump(targeted_output_1, f, indent=2)

    with open(attack_folder / "targeted_2_ensemble_output.json", "w") as f:
        json.dump(targeted_output_2, f, indent=2)

    # Create comprehensive metadata
    metadata = {
        "image_path": image_path,
        "targeted_coarse_class_1": targeted_coarse_class_1,
        "targeted_coarse_class_2": targeted_coarse_class_2,
        "epsilon": epsilon,
        "test_type": test_type,
        "creation_date": datetime.now().isoformat(),
        "successful_combination_used": test_results.get("successful_combination_used"),
        "total_combinations_tested": test_results.get("total_combinations_tested"),
        "successful_combinations": test_results.get("successful_combinations", []),
    }

    with open(attack_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Saved comprehensive results with ensemble predictions to: {attack_folder}"
    )


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

    # Step 4: Save results with detailed ensemble predictions (only if attack succeeded)
    # Create the attack folder with our naming convention
    image_name = Path(image_path).stem
    attack_folder = output_dir / f"{image_name}_eps{epsilon}_test{test_type}"
    attack_folder.mkdir(parents=True, exist_ok=True)

    # Initialize tester for comprehensive saving
    tester = AdversarialTester(device)

    # Get the successful variant images and corresponding logits
    if (
        "targeted_image_1_used" in test_results
        and "targeted_image_2_used" in test_results
    ):
        # Use the successful combination
        targeted_image_1_to_save = test_results["targeted_image_1_used"]
        targeted_image_2_to_save = test_results["targeted_image_2_used"]

        # Compute logits for the successful combination
        targeted_logits_1 = get_ensemble_logits(
            tester.normalize(targeted_image_1_to_save), tester.models
        )
        targeted_logits_2 = get_ensemble_logits(
            tester.normalize(targeted_image_2_to_save), tester.models
        )
    else:
        # Fallback for legacy results - use first variant
        targeted_image_1_to_save = targeted_images_1[0]
        targeted_image_2_to_save = targeted_images_2[0]

        targeted_logits_1 = get_ensemble_logits(
            tester.normalize(targeted_image_1_to_save), tester.models
        )
        targeted_logits_2 = get_ensemble_logits(
            tester.normalize(targeted_image_2_to_save), tester.models
        )

    # Save all ensemble prediction files directly to our attack folder
    _save_comprehensive_results(
        attack_folder,
        test_type,
        original_image,
        targeted_image_1_to_save,
        targeted_image_2_to_save,
        targeted_logits_1,
        targeted_logits_2,
        test_results,
        image_path,
        targeted_coarse_class_1,
        targeted_coarse_class_2,
        epsilon,
        tester,
    )

    return True, test_results, str(attack_folder)
