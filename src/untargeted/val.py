#!/usr/bin/env python3
"""
Image Validation Script for Untargeted Attacks

Validates if an original image is suitable for untargeted adversarial attacks.
Performs two validation tests to ensure the image meets the criteria.

Usage:
    python val.py <original_image_path> <original_fine_class> <original_coarse_class>

    original_image_path: Path to the original image
    original_fine_class: ImageNet class ID (0-999)
    original_coarse_class: Coarse class label
"""

import logging
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

# Import functions from utils module
from ..utils import (
    compute_coarse_score,
    get_correct_coarse_mappings,
    get_ensemble_logits,
    get_normalize_transform,
    get_preprocess_transform,
    load_ensemble,
    load_image,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def load_imagenet_class_names() -> Dict[int, str]:
    """Load ImageNet class names from file."""
    class_names = {}
    try:
        with open("imagenet_classes/imagenet_classes.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and "," in line and not line.startswith("List"):
                    parts = line.split(", ", 1)
                    if len(parts) == 2:
                        class_id = int(parts[0])
                        class_name = parts[1]
                        class_names[class_id] = class_name
    except FileNotFoundError:
        logger.warning(
            "ImageNet class names file not found. Using class indices as names."
        )
        # Fallback: use class indices as names
        for i in range(1000):
            class_names[i] = f"class_{i}"

    return class_names


class ImageValidator:
    """Validate images for untargeted adversarial attacks."""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        logger.info(f"Using device: {self.device}")

        # Standard ImageNet normalization
        self.normalize = get_normalize_transform()

        # Image preprocessing
        self.preprocess = get_preprocess_transform()

        # Load ensemble models
        self.models = load_ensemble(self.device)

        # Load 16-class coarse mappings
        self.coarse_labels, self.coarse_indices = get_correct_coarse_mappings()

        # Load ImageNet class names
        self.class_names = load_imagenet_class_names()

        logger.info("Image validator ready")

    def get_top_predictions(
        self, logits: torch.Tensor, top_k: int = 1000
    ) -> Tuple[List[int], List[float]]:
        """Get top-k predictions from logits."""
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
        return top_indices[0].cpu().numpy().tolist(), top_probs[
            0
        ].cpu().numpy().tolist()

    def test_1_top_category_check(
        self, original_logits: torch.Tensor, original_coarse_class: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Test 1: Check if top category in original image contains the category of interest.
        """
        logger.info("Performing Test 1: Top category check")

        # Get top predictions for original image
        original_top_indices, original_top_probs = self.get_top_predictions(
            original_logits
        )

        logger.info(f"Original top predictions: {original_top_indices[:5]}")

        # Get original coarse class indices
        original_coarse_indices = self.coarse_indices[
            self.coarse_labels.index(original_coarse_class)
        ]

        logger.info(
            f"Original coarse class '{original_coarse_class}' indices: {original_coarse_indices}"
        )

        # Check if TOP (first) category contains original coarse category
        original_top_prediction = original_top_indices[0]
        success = original_top_prediction in original_coarse_indices

        # Get class name for logging
        top_class_name = self.class_names.get(
            original_top_prediction, f"class_{original_top_prediction}"
        )

        results = {
            "test_type": 1,
            "original_top_indices": original_top_indices,
            "original_top_probs": original_top_probs,
            "original_top_prediction": original_top_prediction,
            "original_top_class_name": top_class_name,
            "original_coarse_indices": original_coarse_indices,
            "success": success,
        }

        logger.info(
            f"Test 1 Results: Success={success}, Top prediction: {top_class_name} (class {original_top_prediction})"
        )

        return success, results

    def test_2_coarse_score_check(
        self, original_logits: torch.Tensor, original_coarse_class: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Test 2: Check if coarse score in original image is > 0.
        """
        logger.info("Performing Test 2: Coarse score check")

        if original_coarse_class not in self.coarse_labels:
            logger.warning(
                f"Original coarse class '{original_coarse_class}' not found in coarse labels. Test 2 failed."
            )
            return False, {
                "test_type": 2,
                "success": False,
                "error": "Coarse class not found",
            }

        # Get coarse class index
        coarse_class_idx = self.coarse_labels.index(original_coarse_class)

        # Get top predictions for original image
        original_top_indices, original_top_probs = self.get_top_predictions(
            original_logits
        )

        original_top_prediction = original_top_indices[0]

        # Compute coarse score
        original_coarse_score = compute_coarse_score(
            original_logits, coarse_class_idx, self.coarse_indices
        ).item()

        logger.info(f"Original coarse score: {original_coarse_score}")

        # Check condition: coarse score > 0
        success = original_coarse_score > 0

        results = {
            "test_type": 2,
            "original_coarse_score_Sc": original_coarse_score,
            "original_top_indices": original_top_indices,
            "original_top_probs": original_top_probs,
            "original_top_prediction": original_top_prediction,
            "original_coarse_class": original_coarse_class,
            "success": success,
        }

        logger.info(
            f"Test 2 Results: Success={success}, Coarse score: {original_coarse_score}"
        )

        return success, results

    def validate_image(
        self,
        original_image_path: str,
        original_fine_class: int,
        original_coarse_class: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate an image using both test criteria.
        Returns True if both tests pass, False otherwise.
        """
        logger.info(f"Validating image: {original_image_path}")

        # Load and preprocess image
        original_image = load_image(original_image_path, self.device)
        original_logits = get_ensemble_logits(
            self.normalize(original_image), self.models
        )

        # Perform both tests
        test1_success, test1_results = self.test_1_top_category_check(
            original_logits, original_coarse_class
        )
        test2_success, test2_results = self.test_2_coarse_score_check(
            original_logits, original_coarse_class
        )

        # Both tests must pass
        overall_success = test1_success and test2_success

        # Create comprehensive results
        validation_results = {
            "original_image_path": original_image_path,
            "original_fine_class": original_fine_class,
            "original_coarse_class": original_coarse_class,
            "overall_success": overall_success,
            "test1_success": test1_success,
            "test2_success": test2_success,
            "test1_results": test1_results,
            "test2_results": test2_results,
            "validation_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Validation complete: Overall success={overall_success}, "
            f"Test1={test1_success}, Test2={test2_success}"
        )

        return overall_success, validation_results


def main():
    """Main function to validate an image for untargeted attacks."""

    if len(sys.argv) != 4:
        print(
            "Usage: python val.py <original_image_path> <original_fine_class> <original_coarse_class>"
        )
        print("  original_image_path: Path to the original image")
        print("  original_fine_class: ImageNet class ID (0-999)")
        print("  original_coarse_class: Coarse class label")
        sys.exit(1)

    original_image_path = sys.argv[1]
    original_fine_class = int(sys.argv[2])
    original_coarse_class = str(sys.argv[3])

    # Validate inputs
    if not (0 <= original_fine_class <= 999):
        raise ValueError("Original fine class must be between 0 and 999")

    try:
        # Initialize validator
        logger.info("Initializing image validator...")
        validator = ImageValidator()

        # Validate image
        success, results = validator.validate_image(
            original_image_path, original_fine_class, original_coarse_class
        )

        # Print results
        if success:
            print("\nVALIDATION PASSED! Image is suitable for untargeted attacks.")
        else:
            print("\nVALIDATION FAILED! Image is not suitable for untargeted attacks.")
            print(
                f"Test 1 (top category): {'PASS' if results['test1_success'] else 'FAIL'}"
            )
            print(
                f"Test 2 (coarse score): {'PASS' if results['test2_success'] else 'FAIL'}"
            )

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
