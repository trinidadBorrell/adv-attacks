#!/usr/bin/env python3
"""
Adversarial Attack Generator using iFGSM

Generates untargeted and targeted adversarial attacks following the paper:
"Subtle adversarial image manipulations influence both human and machine perception"

Usage:
    python gen.py <image_path> <original_fine_class> <original_coarse_class> <epsilon>

    image_path: Path to input image
    original_fine_class: ImageNet class ID (0-999)
    original_coarse_class: ImageNet coarse class label
    epsilon: Perturbation magnitude (e.g., 8.0 for 8/255)

Output:
    - Untargeted adversarial image tensor: reduces confidence in original class
    - Targeted adversarial image tensor: maximizes confidence in original class
"""

import logging
import sys
import warnings

import torch
import torch.nn.functional as F

# Import functions from utils module
from ..utils import (
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

    def generate_untargeted_attack(
        self, image: torch.Tensor, original_class: int, epsilon: float
    ) -> torch.Tensor:
        """
        Generate untargeted attack to reduce confidence in original class.

        Following paper: "the adversarial objective function corresponding to an
        untargeted adversarial attack reducing the prediction confidence of class y
        for input X is the binary cross entropy loss with label y_hat (aka, anything except y)"
        """

        # Prepare image for gradient computation
        image_var = image.clone().detach().requires_grad_(True)
        normalized_image = self.normalize(image_var)  # Why is this needed?

        # Step 0: Get ensemble logits
        ensemble_logits = get_ensemble_logits(normalized_image, self.models)

        # Step 1: Get probabilities from ensemble logits using softmax
        probs = F.softmax(ensemble_logits, dim=1)  # Shape: [batch_size, num_classes]

        # Step 2: Get probability of the original predicted class (y)
        prob_original_class = probs[0, original_class]  # P_ens(y|X)

        # Step 3: Calculate probability of "not y" (ȳ) - all other classes combined
        prob_not_original_class = 1 - prob_original_class  # P_ens(ȳ|X) = 1 - P_ens(y|X)

        # Step 4: Calculate untargeted loss according to equation (3)
        loss = -torch.log(prob_not_original_class)  # -log(P_ens(ȳ|X))

        # Compute gradient
        gradient = torch.autograd.grad(loss, image_var, retain_graph=False)[0]

        # Apply iFGSM
        adversarial_image = ifgsm_attack(image, epsilon, gradient)

        return adversarial_image

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

        gradients = torch.zeros_like(image_var)
        for class_idx in coarse_indices:
            # Step 2.1: Get probability of fine-grained class of coarse class (P_ens(y|X))
            prob_target_class = probs[0, class_idx]  # P_ens(y|X)
            # Step 2.2: Get loss of fine-grained class of coarse class (-log(P_ens(y|X)))
            loss = -torch.log(prob_target_class)
            # Step 2.3: Get gradient of loss
            gradient = torch.autograd.grad(loss, image_var, retain_graph=False)[0]
            # Step 2.4: Get sign of gradient
            sign_grad = gradient.sign()
            # Step 2.5: Add sign of gradient to total gradient
            gradients += sign_grad

        # Step 3: Apply iFGSM
        adversarial_image = ifgsm_attack(image, epsilon, gradients)

        return adversarial_image

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        return load_image(image_path, self.device)


def main():
    """Main function following exact specifications."""

    if len(sys.argv) != 4:
        print("Usage: python gen.py <image_path> <original_class> <epsilon>")
        print("  image_path: Path to input image")
        print("  original_class: ImageNet class ID (0-999)")
        print("  epsilon: Perturbation magnitude (e.g., 8.0)")
        sys.exit(1)

    image_path = sys.argv[1]
    original_fine_class = int(sys.argv[2])
    original_coarse_class = str(sys.argv[3])
    epsilon = float(sys.argv[4])

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

        # Generate untargeted attack
        logger.info(
            f"Generating untargeted attack (class {original_fine_class}, ε={epsilon})..."
        )
        untargeted_image = generator.generate_untargeted_attack(
            image, original_fine_class, epsilon
        )

        # Generate targeted attack (towards original class)
        logger.info(
            f"Generating targeted attack towards original class {original_coarse_class}..."
        )

        if original_coarse_class is not None:
            targeted_image = generator.generate_targeted_attack(
                image, original_coarse_class, epsilon
            )
        else:
            targeted_image = generator.generate_targeted_attack(
                image, original_fine_class, epsilon
            )

        # Calculate perturbation norms
        untargeted_norm = torch.norm(untargeted_image - image, p=float("inf")).item()
        targeted_norm = torch.norm(targeted_image - image, p=float("inf")).item()

        print("\nAttack generation completed!")
        print(f"Untargeted perturbation L∞ norm: {untargeted_norm:.6f}")
        print(f"Targeted perturbation L∞ norm: {targeted_norm:.6f}")
        print("Returning adversarial image tensors (not saving to files)")

        # Return tensors for use by test.py
        return (
            untargeted_image,
            targeted_image,
            original_fine_class,
            original_coarse_class,
        )

    except Exception as e:
        logger.error(f"Attack generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
