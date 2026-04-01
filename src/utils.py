#!/usr/bin/env python3
"""
Utility functions for adversarial attack generation and testing.

This module contains common functions used across different adversarial attack scripts.
"""

import logging
import os
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Setup logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def load_ensemble(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[nn.Module]:
    """
    Load the 6-model ensemble specified in the original code.

    Args:
        device: Device to load models on ('cuda' or 'cpu')

    Returns:
        List of loaded and prepared PyTorch models
    """
    weights = [
        models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        models.EfficientNet_B5_Weights.IMAGENET1K_V1,
        models.ResNet101_Weights.IMAGENET1K_V1,
        models.ResNet152_Weights.IMAGENET1K_V1,
        models.Inception_V3_Weights.IMAGENET1K_V1,
        models.ResNet50_Weights.IMAGENET1K_V1,
    ]

    model_list = [
        models.efficientnet_b4(weights=weights[0]),
        models.efficientnet_b5(weights=weights[1]),
        models.resnet101(weights=weights[2]),
        models.resnet152(weights=weights[3]),
        models.inception_v3(weights=weights[4]),
        models.resnet50(weights=weights[5]),
    ]

    # Prepare models
    prepared_models = []
    for i, model in enumerate(model_list):
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        prepared_models.append(model)
        # Model loaded silently

    return prepared_models


def get_correct_coarse_mappings() -> Tuple[List[str], List[List[int]]]:
    """
    Get 16-class coarse mappings from paper using correct WNID to ImageNet index mapping.
    Based on synset_to_name.txt file.

    Returns:
        Tuple of (coarse_labels, coarse_indices) where:
        - coarse_labels: List of coarse class names
        - coarse_indices: List of lists containing ImageNet class indices for each coarse class
    """

    # 16 coarse categories from paper + 4 additional categories
    coarse_labels = [
        "knife",
        "keyboard",
        "elephant",
        "bicycle",
        "airplane",
        "clock",
        "oven",
        "chair",
        "bear",
        "boat",
        "cat",
        "bottle",
        "truck",
        "car",
        "bird",
        "dog",
        "head_cabbage",
        "broccoli",
        "snake",
        "spider",
        "sheep",
    ]

    # Correctly mapped ImageNet class indices based on synset_to_name.txt
    # WNID -> ImageNet index mapping:

    # knife = [n03041632] -> cleaver, meat cleaver, chopper
    knife = [499]  # n03041632 cleaver, meat cleaver, chopper

    # keyboard = [n03085013, n04505470] -> computer keyboard, typewriter keyboard
    keyboard = [508, 878]  # n03085013 computer keyboard, n04505470 typewriter keyboard

    # elephant = [n02504013, n02504458] -> Indian elephant, African elephant
    elephant = [385, 386]  # n02504013 Indian elephant, n02504458 African elephant

    # bicycle = [n02835271, n03792782] -> bicycle-built-for-two, mountain bike
    bicycle = [444, 671]  # n02835271 bicycle-built-for-two, n03792782 mountain bike

    # airplane = [n02690373, n03955296, n13861050, n13941806] -> airliner (n03955296 not found)
    airplane = [404]  # n02690373 airliner

    # clock = [n02708093, n03196217, n04548280] -> analog clock, digital clock, wall clock
    clock = [
        409,
        530,
        892,
    ]  # n02708093 analog clock, n03196217 digital clock, n04548280 wall clock

    # oven = [n03259401, n04111414, n04111531] -> rotisserie (only n04111531 found)
    oven = [766]  # n04111531 rotisserie

    # chair = [n02791124, n03376595, n04099969, n00605023, n04429376] -> various chairs
    chair = [
        423,
        559,
        765,
        857,
    ]  # n02791124 barber chair, n03376595 folding chair, n04099969 rocking chair, n04429376 throne

    # bear = [n02132136, n02133161, n02134084, n02134418] -> various bears
    bear = [
        294,
        295,
        296,
        297,
    ]  # n02132136 brown bear, n02133161 American black bear, n02134084 ice bear, n02134418 sloth bear

    # boat = [n02951358, n03344393, n03662601, n04273569, n04612373, n04612504] -> various boats
    boat = [
        472,
        554,
        625,
        814,
        914,
    ]  # n02951358 canoe, n03344393 fireboat, n03662601 lifeboat, n04273569 speedboat, n04612504 yawl

    # cat = [n02122878, n02123045, n02123159, n02126465, n02123394, n02123597, n02124075, n02125311]
    cat = [
        281,
        282,
        283,
        284,
        285,
        286,
    ]  # n02123045 tabby, n02123159 tiger cat, n02123394 Persian cat, n02123597 Siamese cat, n02124075 Egyptian cat, n02125311 cougar

    # bottle = [n02823428, n03937543, n03983396, n04557648, n04560804, n04579145, n04591713]
    bottle = [
        440,
        720,
        737,
        898,
        899,
        901,
        907,
    ]  # n02823428 beer bottle, n03937543 pill bottle, n03983396 pop bottle, n04557648 water bottle, n04560804 water jug, n04579145 whiskey jug, n04591713 wine bottle

    # truck = [n03345487, n03417042, n03770679, n03796401, n00319176, n01016201, n03930630, n03930777, n05061003, n06547832, n10432053, n03977966, n04461696, n04467665]
    truck = [
        555,
        569,
        656,
        675,
        717,
        734,
        864,
        867,
    ]  # n03345487 fire engine, n03417042 garbage truck, n03770679 minivan, n03796401 moving van, n03930630 pickup, n03977966 police van, n04461696 tow truck, n04467665 trailer truck

    # car = [n02814533, n03100240, n03100346, n13419325, n04285008]
    car = [
        436,
        511,
        817,
    ]  # n02814533 beach wagon, n03100240 convertible, n04285008 sports car

    # bird = [extensive list] -> many bird species (mapped from WNIDs)
    bird = [
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        18,
        19,
        20,
        22,
        23,
        24,
        80,
        81,
        82,
        83,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        98,
        99,
        100,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
    ]

    # dog = [extensive list] -> many dog breeds (mapped from WNIDs)
    dog = [
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        193,
        194,
        195,
        196,
        197,
        198,
        199,
        200,
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        228,
        229,
        230,
        231,
        232,
        233,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
        242,
        243,
        244,
        245,
        246,
        247,
        248,
        249,
        250,
        251,
        252,
        253,
    ]

    # Additional categories (not from original paper)
    head_cabbage = [936]  # head cabbage
    broccoli = [937]  # broccoli
    snake = [52, 53, 54, 55, 56, 57, 58, 59, 60]  # various snake species
    spider = [72, 73, 74, 75, 76, 77, 78]  # various spider species
    sheep = [348, 349, 350]  # various sheep species

    coarse_indices = [
        knife,
        keyboard,
        elephant,
        bicycle,
        airplane,
        clock,
        oven,
        chair,
        bear,
        boat,
        cat,
        bottle,
        truck,
        car,
        bird,
        dog,
        head_cabbage,
        broccoli,
        snake,
        spider,
        sheep,
    ]

    # Filter non-empty categories
    valid_labels = []
    valid_indices = []
    for label, indices in zip(coarse_labels, coarse_indices):
        if indices:
            valid_labels.append(label)
            valid_indices.append(indices)
            # Coarse class mapping loaded silently

    return valid_labels, valid_indices


def get_ensemble_logits(image: torch.Tensor, models: List[nn.Module]) -> torch.Tensor:
    """
    Get ensemble logits using arithmetic mean as specified in paper:
    "taking an average of the unnormalized predictions (aka logits)"

    Args:
        image: Input image tensor
        models: List of ensemble models

    Returns:
        Ensemble logits tensor
    """
    ensemble_outputs = []

    for model in models:
        output = model(image)
        ensemble_outputs.append(
            output
        )  # Keep gradients for adversarial attack generation

    # Arithmetic mean of logits
    ensemble_logits = torch.mean(torch.stack(ensemble_outputs), dim=0)
    return ensemble_logits


def compute_coarse_score(
    logits: torch.Tensor, coarse_class_idx: int, coarse_indices: List[List[int]]
) -> torch.Tensor:
    """
    Compute coarse class score using Equation (1) from paper:
    Sc = log(∑_{i∈c} exp(Si)) - log(∑_{j∉c} exp(Sj))

    Args:
        logits: Model logits tensor
        coarse_class_idx: Index of the coarse class
        coarse_indices: List of lists containing ImageNet class indices for each coarse class

    Returns:
        Coarse score tensor
    """
    coarse_class_indices = coarse_indices[coarse_class_idx]

    # Logits for classes IN coarse category
    coarse_logits = logits[:, coarse_class_indices]

    # Logits for classes NOT in coarse category
    all_indices = set(range(1000))
    non_coarse_indices = list(all_indices - set(coarse_class_indices))
    non_coarse_logits = logits[:, non_coarse_indices]

    # Apply Equation (1) with numerical stability
    sum_exp_coarse = torch.logsumexp(coarse_logits, dim=1)
    sum_exp_non_coarse = torch.logsumexp(non_coarse_logits, dim=1)

    coarse_score = sum_exp_coarse - sum_exp_non_coarse
    return coarse_score


def load_image(
    image_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Load and preprocess image.

    Args:
        image_path: Path to the image file
        device: Device to load the tensor on

    Returns:
        Preprocessed image tensor

    Raises:
        FileNotFoundError: If the image file doesn't exist
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Image preprocessing
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return image_tensor


def ifgsm_attack(
    image: torch.Tensor, epsilon: float, gradient: torch.Tensor
) -> torch.Tensor:
    """
    iFGSM attack implementation following paper's Equations (4) and (5).

    Args:
        image: Original image tensor
        epsilon: Perturbation magnitude
        gradient: Computed gradient tensor

    Returns:
        Perturbed image tensor
    """
    max_iterations = int(min([epsilon + 4, epsilon * 1.25]))
    alpha = 1.0
    perturbed_image = image.clone().detach()

    for i in range(max_iterations):
        # Equation (4): Xi = Xi-1 + α × sign(∇X(J(X, y)))
        perturbed_image = perturbed_image + (alpha / 255.0) * gradient.sign()

        # Equation (5): Xi = clip(Xi, X - ε, X + ε)
        perturbed_image = torch.clamp(
            perturbed_image, image - epsilon / 255.0, image + epsilon / 255.0
        )
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def get_normalize_transform():
    """
    Get standard ImageNet normalization transform.

    Returns:
        Normalization transform
    """
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_preprocess_transform():
    """
    Get standard image preprocessing transform.

    Returns:
        Preprocessing transform
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
