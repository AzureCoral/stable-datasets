from io import BytesIO
import tarfile
from zipfile import ZipFile
import numpy as np
from ..utils import Dataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from PIL import Image


class AWA2(Dataset):
    """
    Tiny Imagenet has 200 classes. Each class has 500 training images, 50
    validation images, and 50 test images. We have released the training and
    validation sets with images and annotations. We provide both class labels an
    bounding boxes as annotations; however, you are asked only to predict the
    class label of each image without localizing the objects. The test set is
    released without labels. You can download the whole tiny ImageNet dataset
    here.
    """

    @property
    def urls(self):
        return {
            "AwA2-base.zip": "https://cvml.ista.ac.at/AwA2/AwA2-base.zip",
            "AwA2-data.zip": "https://cvml.ista.ac.at/AwA2/AwA2-data.zip",
        }

    @property
    def md5(self):
        return {
            "AwA2-base.zip": "90528d7ca1a48142e341f4ef8d21d0de",
            "AwA2-data.zip": "90528d7ca1a48142e341f4ef8d21d0de",
        }

    @property
    def num_classes(self):
        return 50

    def load(self):
        asdf
        # Loading the file
        f = ZipFile(self.path / self.name / "tiny-imagenet-200.zip", "r")
        names = [name for name in f.namelist() if name.endswith("JPEG")]
        val_classes = np.loadtxt(
            f.open("tiny-imagenet-200/val/val_annotations.txt"),
            dtype=str,
            delimiter="\t",
        )
        val_classes = dict(
            [(a, b) for a, b in zip(val_classes[:, 0], val_classes[:, 1])]
        )
        x_train, x_test, x_valid, y_train, y_test, y_valid = [], [], [], [], [], []
        for name in tqdm(names, desc=f"Loading {self.name}"):
            im = Image.open(f.open(name)).convert("RGB")
            if "train" in name:
                classe = name.split("/")[-1].split("_")[0]
                x_train.append(im)
                y_train.append(classe)
            if "val" in name:
                x_valid.append(im)
                arg = name.split("/")[-1]
                y_valid.append(val_classes[arg])
            if "test" in name:
                x_test.append(im)
        labels = LabelEncoder().fit(y_train)
        self["train_X"] = x_train
        self["train_y"] = labels.transform(y_train)
        self["test_X"] = x_valid
        self["test_y"] = labels.transform(y_valid)
