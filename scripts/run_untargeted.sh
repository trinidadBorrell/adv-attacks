#!/bin/bash

# Untargeted Attack Runner Script
# 
# This script runs untargeted adversarial attacks on a set of images with different
# epsilons and categories. It validates each image first, then generates attacks
# and tests them if validation passes.
#
# Usage: ./run_untargeted.sh <epsilons> <categories> <imagenet_folder_path> <test_type> <output>
#
# Arguments:
#   epsilons: Comma-separated list of epsilon values (e.g., "8.0,16.0,32.0")
#   categories: Comma-separated list of coarse categories (e.g., "fish,bird,mammal")
#   imagenet_folder_path: Path to the miniImageNet folder (structure: miniImageNet/{folder}/{image_files..})
#   test_type: Test type to use (1 or 2)
#   output: Output base directory

set -e  # Exit on any error

# Function to print usage
print_usage() {
    echo "Usage: $0 <epsilons> <categories> <imagenet_folder_path> <test_type> <output>"
    echo ""
    echo "Arguments:"
    echo "  epsilons: Comma-separated list of epsilon values (e.g., '8.0,16.0,32.0')"
    echo "  categories: Comma-separated list of coarse categories (e.g., 'fish,bird,mammal')"
    echo "  imagenet_folder_path: Path to the miniImageNet folder (structure: miniImageNet/{folder}/{image_files..})"
    echo "  test_type: Test type to use (1 or 2)"
    echo "  output: Output base directory"
    echo ""
    echo "Example:"
    echo "  $0 '8.0,16.0' 'fish,bird' '/path/to/miniImageNet' 1 '/path/to/results'"
}

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check number of arguments
if [ $# -ne 5 ]; then
    echo "Error: Incorrect number of arguments"
    print_usage
    exit 1
fi

# Parse arguments
EPSILONS_STR="$1"
CATEGORIES_STR="$2"
IMAGENET_FOLDER_PATH="$3"
TEST_TYPE="$4"
OUTPUT_BASE="$5"

# Validate test type
if [ "$TEST_TYPE" != "1" ] && [ "$TEST_TYPE" != "2" ]; then
    echo "Error: test_type must be 1 or 2"
    exit 1
fi

# Convert comma-separated strings to arrays
IFS=',' read -ra EPSILONS <<< "$EPSILONS_STR"
IFS=',' read -ra CATEGORIES <<< "$CATEGORIES_STR"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if required commands exist
if ! command_exists python3; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "$PROJECT_ROOT/src/untargeted/val.py" ]; then
    echo "Error: val.py not found at $PROJECT_ROOT/src/untargeted/val.py"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/src/untargeted/gen.py" ]; then
    echo "Error: gen.py not found at $PROJECT_ROOT/src/untargeted/gen.py"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/src/untargeted/test.py" ]; then
    echo "Error: test.py not found at $PROJECT_ROOT/src/untargeted/test.py"
    exit 1
fi

# Check if miniImageNet folder exists
if [ ! -d "$IMAGENET_FOLDER_PATH" ]; then
    echo "Error: miniImageNet folder not found at $IMAGENET_FOLDER_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Log start
log_message "Starting untargeted attack runner"
log_message "Epsilons: ${EPSILONS[*]}"
log_message "Categories: ${CATEGORIES[*]}"
log_message "miniImageNet folder: $IMAGENET_FOLDER_PATH"
log_message "Test type: $TEST_TYPE"
log_message "Output base: $OUTPUT_BASE"



# Main loop
total_images=0
valid_images=0
successful_attacks=0

for eps in "${EPSILONS[@]}"; do
    log_message "Processing epsilon: $eps"
    
    for category in "${CATEGORIES[@]}"; do
        log_message "Processing category: $category"
        
        # Create category-specific output directory
        category_output="$OUTPUT_BASE/exp2/$category/epsilon_$eps"
        mkdir -p "$category_output"
        
        # Find all images in subfolders of the miniImageNet folder
        while IFS= read -r -d '' image_path; do
            total_images=$((total_images + 1))
            
            # Get image name without extension
            image_name=$(basename "$image_path" | sed 's/\.[^.]*$//')
            
            log_message "Processing image: $image_name"
            
            # Use category as coarse class directly
            coarse_class="$category"
            
            # Step 1: Validate image
            log_message "Validating image..."
            if python3 "$PROJECT_ROOT/src/untargeted/val.py" "$image_path" "0" "$coarse_class"; then
                valid_images=$((valid_images + 1))
                log_message "Validation PASSED for $image_name"
                
                # Create image-specific output directory ONLY after validation succeeds
                image_output="$category_output/$image_name"
                mkdir -p "$image_output"
                
                # Step 2: Generate untargeted attack
                log_message "Generating untargeted attack..."
                if python3 "$PROJECT_ROOT/src/untargeted/gen.py" "$image_path" "0" "$coarse_class" "$eps"; then
                    log_message "Untargeted attack generated successfully"
                    
                    # Step 3: Generate targeted attack
                    log_message "Generating targeted attack..."
                    if python3 "$PROJECT_ROOT/src/untargeted/gen.py" "$image_path" "0" "$coarse_class" "$eps" --targeted; then
                        log_message "Targeted attack generated successfully"
                        
                        # Step 4: Test attacks
                        log_message "Testing attacks..."
                        if python3 "$PROJECT_ROOT/src/untargeted/test.py" "$TEST_TYPE" "$image_path" "untargeted.png" "targeted.png" "0" "$coarse_class" "$eps"; then
                            successful_attacks=$((successful_attacks + 1))
                            log_message "Test PASSED for $image_name"
                        else
                            log_message "Test FAILED for $image_name"
                        fi
                    else
                        log_message "Targeted attack generation FAILED for $image_name"
                    fi
                else
                    log_message "Untargeted attack generation FAILED for $image_name"
                fi
            else
                log_message "Validation FAILED for $image_name - skipping to next image"
            fi
            
            log_message "Completed processing $image_name"
            echo "---"
        done < <(find "$IMAGENET_FOLDER_PATH" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) -print0)
    done
done

# Final summary
log_message "=== FINAL SUMMARY ==="
log_message "Total images processed: $total_images"
log_message "Valid images: $valid_images"
log_message "Successful attacks: $successful_attacks"
log_message "Success rate: $((successful_attacks * 100 / total_images))%"
log_message "Results saved to: $OUTPUT_BASE/exp2/"

log_message "Untargeted attack runner completed successfully!" 