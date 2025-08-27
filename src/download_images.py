#!/usr/bin/env python3
"""
Simplified mini-ImageNet dataset downloader.
Downloads the mini-ImageNet dataset from Kaggle using kagglehub.
"""

from pathlib import Path
import kagglehub

def download_sample_images():
    """
    Download mini-ImageNet dataset from Kaggle.
    """
    print("Downloading mini-ImageNet dataset from Kaggle...")
    
    # Download latest version
    path = kagglehub.dataset_download("arjunashok33/miniimagenet")
    
    print("Path to dataset files:", path)
    
    # Create a symlink in our results directory for easy access
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_link = results_dir / "mini_imagenet"
    if dataset_link.exists():
        dataset_link.unlink()
    
    dataset_link.symlink_to(Path(path))
    print(f"Dataset linked to: {dataset_link}")
    
    # Explore the dataset structure
    print("\nDataset structure:")
    for item in Path(path).iterdir():
        if item.is_dir():
            print(f"  Directory: {item.name} ({len(list(item.iterdir()))} items)")
        else:
            print(f"  File: {item.name}")

if __name__ == "__main__":
    download_sample_images()
