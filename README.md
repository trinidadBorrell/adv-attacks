# AdvAttacks

Research framework for generating subtle adversarial perturbations on ImageNet images that fool both human and machine perception. Attacks are validated against an ensemble of 6 pretrained ImageNet classifiers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the mini-ImageNet dataset (requires Kaggle credentials):

```bash
python src/download_images.py
```

## Attack Types

Three attack modes are supported, each producing adversarial images that pass a success test before being saved:

| Mode | Script | Description |
|---|---|---|
| Untargeted | `cookbook/run_untargeted.py` | Reduces model confidence in the original class |
| Targeted (one) | `cookbook/run_targeted_one_target.py` | Shifts perception to a single target coarse category |
| Targeted (two) | `cookbook/run_targeted_two_targets.py` | Simultaneously targets two coarse categories |

### Coarse Categories

Attacks operate over 21 coarse categories (e.g., `cat`, `dog`, `vehicle`) that each aggregate multiple fine-grained ImageNet classes. Mappings are defined in `imagenet_classes/`.

## Usage

All scripts are run from the repo root.

### Untargeted

```bash
python cookbook/run_untargeted.py \
    "8.0,16.0" \          # epsilons (L-inf perturbation budget)
    "cat,dog" \           # source categories to attack
    data/mini_imagenet \  # path to ImageNet image folder
    1 \                   # test type (1 or 2)
    results/untargeted    # output directory
    --batch_size 4        # parallel workers (default: 1)
    --target_successes 25 # successful attacks to collect per category (default: 25)
```

### Targeted — One Target

```bash
python cookbook/run_targeted_one_target.py \
    "8.0,16.0" \
    "cat,dog" \
    data/mini_imagenet \
    1 \
    results/targeted_one
```

### Targeted — Two Targets

```bash
python cookbook/run_targeted_two_targets.py \
    "8.0,16.0" \
    "cat,dog" \           # source categories
    "vehicle,bird" \      # target categories
    data/mini_imagenet \
    1 \
    results/targeted_two
```

Each run saves a `used_images_*.json` file in the output directory to avoid reprocessing images across runs.

## Algorithm

1. **Validate** — confirm the model ensemble correctly classifies the original image
2. **Generate** — apply iFGSM to produce a perturbation within the epsilon L-inf budget
3. **Test** — verify the adversarial image achieves the desired misclassification
4. **Save** — write only successful adversarial images to the output directory

For targeted attacks, the algorithm identifies the top-3 most probable fine-grained classes within the target coarse category and runs separate attacks in each direction, plus one attack on the aggregated coarse-category score. The best successful result is kept.

## Model Ensemble

Six pretrained torchvision models are used in combination:

- EfficientNet B4 & B5
- ResNet 101 & 152
- Inception V3
- ResNet 50

Decisions are based on the arithmetic mean of their logits.

## Project Structure

```
src/
  utils.py                          # ensemble loading, iFGSM, coarse scoring
  mapping.py                        # coarse ↔ fine class mappings
  streaming_dataset.py              # local and streaming dataset iterators
  untargeted/                       # untargeted attack pipeline
  targeted/
    one_targets/                    # single-target pipeline
    two_targets/                    # two-target pipeline
  sanitycheck/                      # post-experiment verification scripts
cookbook/                           # high-level experiment runners
imagenet_classes/                   # WNID-to-category mapping files
data/                               # image datasets
results/                            # output adversarial images
```

## Linting

```bash
ruff check src/ cookbook/
```
