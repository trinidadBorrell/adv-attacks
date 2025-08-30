#!/usr/bin/env python3
"""
Simple mapping utilities for ImageNet classes and coarse categories.
"""

from pathlib import Path


def load_16_class_mapping() -> dict[str, list[str]]:
    """Load the 16 coarse class to WNID mapping from file."""
    mapping_file = Path("imagenet_classes/16_class_mapping.txt")
    mapping = {}
    
    with open(mapping_file, 'r') as f:
        current_category = None
        current_wnids = []
        
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                # Save previous category if exists
                if current_category:
                    mapping[current_category] = current_wnids
                
                # Parse new category
                parts = line.split('=')
                current_category = parts[0].strip()
                wnids_part = parts[1].strip()
                
                # Extract WNIDs from brackets
                if '[' in wnids_part and ']' in wnids_part:
                    wnids_str = wnids_part.split('[')[1].split(']')[0]
                    current_wnids = [wnid.strip() for wnid in wnids_str.split(',')]
                else:
                    current_wnids = []
            elif current_category and line and not line.startswith('#'):
                # Continue reading WNIDs from next lines
                if '[' in line or ']' in line:
                    wnids_str = line.replace('[', '').replace(']', '')
                    additional_wnids = [wnid.strip() for wnid in wnids_str.split(',') if wnid.strip()]
                    current_wnids.extend(additional_wnids)
        
        # Save last category
        if current_category:
            mapping[current_category] = current_wnids
    
    return mapping


def load_wnid_to_imagenet_mapping() -> dict[str, int]:
    """Load WNID to ImageNet class index mapping from synset_to_name.txt."""
    synset_file = Path("imagenet_classes/synset_to_name.txt")
    mapping = {}
    
    with open(synset_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                wnid = line.split()[0]
                mapping[wnid] = i
    
    return mapping


def get_coarse_to_imagenet_mapping() -> dict[str, list[int]]:
    """Get mapping from coarse categories to ImageNet class indices."""
    coarse_mapping = load_16_class_mapping()
    wnid_mapping = load_wnid_to_imagenet_mapping()
    
    result = {}
    for category, wnids in coarse_mapping.items():
        imagenet_indices = []
        for wnid in wnids:
            if wnid in wnid_mapping:
                imagenet_indices.append(wnid_mapping[wnid])
        result[category] = imagenet_indices
    
    return result


def get_representative_class_for_category(category: str) -> int:
    """Get a representative ImageNet class ID for a coarse category."""
    mapping = get_coarse_to_imagenet_mapping()
    if category in mapping and mapping[category]:
        return mapping[category][0]  # Return first class as representative
    raise ValueError(f"Unknown category: {category}")
