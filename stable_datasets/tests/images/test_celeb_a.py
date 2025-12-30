import numpy as np
from PIL import Image

from stable_datasets.images.celeb_a import CelebA


def test_celeb_a_dataset():
    # Load training split
    celeb_a_train = CelebA(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 162770
    assert len(celeb_a_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(celeb_a_train)}."
    )

    # Test 2: Check sample keys
    sample = celeb_a_train[0]
    expected_keys = {"image", "attributes"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."

    # Convert to numpy for basic sanity checks
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Celeb A images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"Celeb A images should have 3 channels, got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    attributes = sample["attributes"]
    assert isinstance(attributes, list | np.ndarray), f"Attributes should be list of ints, got {type(attributes)}."
    assert len(attributes) == 40, "Length of attributes should be 40."
    assert all(attr in [-1, 1] for attr in attributes), "Attributes should be binary (-1 or 1)."

    # Test 5: Load and validate test split
    celeb_a_test = CelebA(split="test")
    expected_num_test_samples = 19962
    assert len(celeb_a_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(celeb_a_test)}."
    )

    # Test 6: Load and validate validation split
    celeb_a_val = CelebA(split="valid")
    expected_num_val_samples = 19867
    assert len(celeb_a_val) == expected_num_val_samples, (
        f"Expected {expected_num_val_samples} validation samples, got {len(celeb_a_val)}."
    )

    print("All Celeb A dataset tests passed successfully!")
