#!/usr/bin/env python3
"""
Adversarial Attack Generator using iFGSM

Generates targeted adversarial attack 1 and 2 following the paper:
"Subtle adversarial image manipulations influence both human and machine perception"

Usage:
    python gen.py <image_path> <original_fine_class> <original_coarse_class> <target_coarse_class 1> <target_coarse_class 2> <epsilon>

    image_path: Path to input image
    original_fine_class: ImageNet class ID (0-999)
    original_coarse_class: ImageNet coarse class label
    target_coarse_class 1: ImageNet coarse class label
    target_coarse_class 2: ImageNet coarse class label
    epsilon: Perturbation magnitude (e.g., 8.0 for 8/255)

Output:
    - Targeted adversarial image tensor 1: maximizes confidence in target class 1
    - Targeted adversarial image tensor 2: maximizes confidence in target class 2
"""

import logging
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Import functions from utils module
from ...utils import (
    get_correct_coarse_mappings,
    get_ensemble_logits,
    get_normalize_transform,
    get_preprocess_transform,
    ifgsm_attack,
    load_ensemble,
    load_image,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class AdversarialGenerator:
    """Adversarial attack generator following paper methodology exactly."""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        logger.info(f"Using device: {self.device}")

        # Standard ImageNet normalization
        self.normalize = get_normalize_transform()

        # Image preprocessing
        self.preprocess = get_preprocess_transform()

        # Load ensemble models
        self.models = load_ensemble(self.device)

        # Load 16-class coarse mappings from paper using correct WNID mapping
        self.coarse_labels, self.coarse_indices = get_correct_coarse_mappings()

        logger.info("Adversarial generator ready")

    def generate_targeted_attack(
        self, image: torch.Tensor, target_class: str, epsilon: float
    ) -> torch.Tensor:
        """
        Generate targeted attack to maximize confidence in target coarse class.
        """

        # Prepare image for gradient computation
        image_var = image.clone().detach().requires_grad_(True)
        normalized_image = self.normalize(image_var)

        # Step 0: Get ensemble logits
        ensemble_logits = get_ensemble_logits(normalized_image, self.models)

        # Step 1: Get probabilities from ensemble logits using softmax (P_ens(y|X))
        probs = F.softmax(ensemble_logits, dim=1)  # Shape: [batch_size, num_classes]

        # Step 2: Get coarse indices of target class
        coarse_indices = self.coarse_indices[self.coarse_labels.index(target_class)]

        # Step 3: Sum the actual gradients, then take the sign
        loss = -torch.log(
            probs[0, coarse_indices].sum()
        )  # Total coarse class probability
        gradient = torch.autograd.grad(loss, image_var)[0]

        # Step 4: Apply iFGSM
        adversarial_image = ifgsm_attack(image, epsilon, gradient)

        return adversarial_image

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        return load_image(image_path, self.device)


def save_tensor_as_image(tensor: torch.Tensor, save_path: str):
    """Save tensor as image file."""
    # Convert tensor to numpy array
    image_np = tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    # Denormalize and convert to uint8
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    # Save as image
    Image.fromarray(image_np).save(save_path)
    logger.info(f"Saved image to: {save_path}")


def main():
    """Main function following exact specifications."""

    if len(sys.argv) != 7:
        print(
            "Usage: python gen.py <image_path> <original_fine_class> <original_coarse_class> <target_coarse_class 1> <target_coarse_class 2> <epsilon>"
        )
        print("  image_path: Path to input image")
        print("  original_fine_class: ImageNet class ID (0-999)")
        print("  original_coarse_class: Coarse class label")
        print("  target_coarse_class 1: Coarse class label")
        print("  target_coarse_class 2: Coarse class label")
        print("  epsilon: Perturbation magnitude (e.g., 8.0)")
        sys.exit(1)

    image_path = sys.argv[1]
    original_fine_class = int(sys.argv[2])
    original_coarse_class = str(sys.argv[3])
    target_coarse_class_1 = str(sys.argv[4])
    target_coarse_class_2 = str(sys.argv[5])
    epsilon = float(sys.argv[6])

    # Validate inputs
    if not (0 <= original_fine_class <= 999):
        raise ValueError("Original class must be between 0 and 999")
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")

    try:
        # Initialize generator
        logger.info("Initializing adversarial generator...")
        generator = AdversarialGenerator()

        # Load image
        logger.info(f"Loading image: {image_path}")
        image = generator.load_image(image_path)

        # Generate targeted attack (towards original class)
        logger.info(f"Generating targeted attack towards {target_coarse_class_1}...")

        targeted_image_1 = generator.generate_targeted_attack(
            image, target_coarse_class_1, epsilon
        )

        # Generate control image (same magnitude of perturbation as targeted adversarial image)
        logger.info("Generating targeted attack towards {target_coarse_class_2}...")

        targeted_image_2 = generator.generate_targeted_attack(
            image, target_coarse_class_2, epsilon
        )

        # Return tensors for testing - will only be saved if tests pass
        return (
            targeted_image_1,
            targeted_image_2,
            original_fine_class,
            original_coarse_class,
            target_coarse_class_1,
            target_coarse_class_2,
        )

    except Exception as e:
        logger.error(f"Attack generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
