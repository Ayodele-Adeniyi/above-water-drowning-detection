## Training Scripts

This folder contains scripts used to train YOLOv11 models under different conditions.

- `baseline.py`: Baseline model trained on the original dataset.
- `turbidity.py`: Model trained with turbidity augmentation.
- `haze.py`: Model trained with haze augmentation.
- `caustics.py`: Model trained with glare/caustics augmentation.
- `combined_model.py`: Model trained using combined augmentations.

All scripts use the same dataset split and training hyperparameters unless stated otherwise.
