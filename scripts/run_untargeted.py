#!/usr/bin/env python3
"""
Clean Untargeted Attack Pipeline

Usage:
    python run_untargeted_clean.py <epsilons> <categories> [imagenet_folder] <test_type> <output>
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.mapping import get_representative_class_for_category
from src.simple_pipeline import run_complete_attack_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_images(folder_path: str) -> List[str]:
    """Find all image files in the folder."""
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = []
    
    folder = Path(folder_path)
    if not folder.exists():
        return []
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(str(Path(root) / file))
    
    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Run untargeted adversarial attacks")
    parser.add_argument("epsilons", help="Comma-separated epsilon values (e.g., '8.0,16.0')")
    parser.add_argument("categories", help="Comma-separated categories (e.g., 'cat,dog')")
    parser.add_argument("imagenet_folder", help="Path to ImageNet folder")
    parser.add_argument("test_type", type=int, choices=[1, 2], help="Test type (1 or 2)")
    parser.add_argument("output", help="Output directory")
    
    args = parser.parse_args()
    
    # Parse arguments
    epsilons = [float(eps.strip()) for eps in args.epsilons.split(",")]
    categories = [cat.strip() for cat in args.categories.split(",")]
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Check if mini-ImageNet dataset exists
    if not args.imagenet_folder:
        logger.error("Please provide a path to the ImageNet folder. If not downloaded, please run: python src/download_images.py")
        return
    
    mini_imagenet_path = Path(args.imagenet_folder) / "mini_imagenet"
    
    if not mini_imagenet_path.exists():
        logger.error(f"Mini-ImageNet dataset not found at {mini_imagenet_path}. Please run: python src/download_images.py")
        return
    
    # Load 16 class mappings and find which ones have available synsets
    def load_available_class_mappings():
        import re
        
        # Get available synsets in dataset
        available_synsets = [d.name for d in mini_imagenet_path.iterdir() if d.is_dir()]
        
        # Load 16 class mapping
        with open("imagenet_classes/16_class_mapping.txt", "r") as f:
            content = f.read()
        
        # Parse mapping to find which classes have available synsets
        class_synsets = {}
        for line in content.split('\n'):
            if '=' in line and '[' in line:
                category = line.split('=')[0].strip()
                synsets_match = re.findall(r'n\d{8}', line)
                if synsets_match:
                    # Only keep synsets that are actually available in our dataset
                    available_for_category = [s for s in synsets_match if s in available_synsets]
                    if available_for_category:
                        class_synsets[category] = available_for_category
        
        return class_synsets
    
    class_mappings = load_available_class_mappings()
    logger.info(f"Available classes with synsets: {[(k, len(v)) for k, v in class_mappings.items()]}")
    
    # Get images for each requested category
    image_paths_by_category = {}
    
    for category in categories:
        if category not in class_mappings:
            logger.warning(f"Category '{category}' not found in class mappings")
            image_paths_by_category[category] = []
            continue
            
        category_images = []
        synsets = class_mappings[category]
        
        for synset in synsets:
            synset_dir = mini_imagenet_path / synset
            if synset_dir.exists():
                images = list(synset_dir.glob("*.JPEG")) + list(synset_dir.glob("*.jpg")) + list(synset_dir.glob("*.png"))
                category_images.extend([str(img) for img in images])
                logger.info(f"Found {len(images)} images in synset {synset} for category {category}")
        
        image_paths_by_category[category] = category_images
        logger.info(f"Total images for {category}: {len(category_images)}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    total_images = 0
    successful_attacks = 0
    
    logger.info("Starting attack pipeline...")
    logger.info(f"Epsilons: {epsilons}")
    logger.info(f"Categories: {categories}")
    logger.info(f"Test type: {args.test_type}")
    
    for eps in epsilons:
        for category in categories:
            fine_class_id = get_representative_class_for_category(category)
            image_paths = image_paths_by_category.get(category, [])
            
            if not image_paths:
                logger.warning(f"No images found for category {category}")
                continue
            
            logger.info(f"Processing {len(image_paths)} images for {category} (eps={eps})")
            aimed_successful_attacks = 5
            for image_path in image_paths:  # Process 20 images per category
                if successful_attacks >= aimed_successful_attacks:
                    break
                total_images += 1
                image_name = Path(image_path).stem
                
                try:
                    success, results, output_folder = run_complete_attack_pipeline(
                        image_path=image_path,
                        fine_class_id=fine_class_id,
                        coarse_class=category,
                        epsilon=eps,
                        test_type=args.test_type,
                        output_dir=output_dir
                    )
                    
                    if success:
                        successful_attacks += 1
                        logger.info(f"SUCCESS: {image_name} -> {output_folder}")
                    else:
                        logger.info(f"FAILED: {image_name}")
                        
                except Exception as e:
                    logger.error(f"ERROR processing {image_name}: {e}")
    
    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Successful attacks: {successful_attacks}")
    if total_images > 0:
        logger.info(f"Success rate: {(successful_attacks/total_images)*100:.1f}%")


if __name__ == "__main__":
    main()
