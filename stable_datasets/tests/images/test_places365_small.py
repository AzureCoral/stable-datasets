import numpy as np
from PIL import Image

from stable_datasets.images.places365_small import Places365Small


def test_places365_small_dataset():
    # Load training split
    places365_train = Places365Small(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 1803460
    assert len(places365_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(places365_train)}."
    )

    # Test 2: Check sample keys
    sample = places365_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."

    # Convert to numpy for basic sanity checks
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Places365 images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"Places365 images should have 3 channels, got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an int, got {type(label)}."
    assert 0 <= label < 365, f"Label should be in range [0, 364], got {label}."

    # Test 5: Load and validate test split
    places365_test = Places365Small(split="test")
    expected_num_test_samples = 382500
    assert len(places365_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(places365_test)}."
    )

    # Test 6: Load and validate validation split
    places365_val = Places365Small(split="validation")
    expected_num_val_samples = 36500
    assert len(places365_val) == expected_num_val_samples, (
        f"Expected {expected_num_val_samples} validation samples, got {len(places365_val)}."
    )

    print("All Places365 dataset tests passed successfully!")
