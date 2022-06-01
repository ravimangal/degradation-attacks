import os
from zipfile import ZipFile
import urllib.request
import numpy as np
import matplotlib.pyplot

# from PIL import Image


_urls = {"http://cs231n.stanford.edu/tiny-imagenet-200.zip": "tiny-imagenet-200.zip"}
_name = "tinyimagenet"

def load():
    """
    Tiny Imagenet has 200 classes. Each class has 500 training images, 50
    validation images, and 50 test images. We have released the training and
    validation sets with images and annotations. We provide both class labels an
    bounding boxes as annotations; however, you are asked only to predict the
    class label of each image without localizing the objects. The test set is
    released without labels. You can download the whole tiny ImageNet dataset
    here.
    """

    path="./tensorflow_datasets"
    # Loading the file
    f = ZipFile(os.path.join(path, "tiny-imagenet-200.zip"), "r")
    names = [name for name in f.namelist() if name.endswith("JPEG")]
    val_classes = np.loadtxt(
        f.open("tiny-imagenet-200/val/val_annotations.txt"),
        dtype=str,
        delimiter="\t",
    )
    val_classes = dict([(a, b) for a, b in zip(val_classes[:, 0], val_classes[:, 1])])
    x_train, x_test, x_valid, y_train, y_test, y_valid = [], [], [], [], [], []
    for name in names:
        if "train" in name:
            classe = name.split("/")[-1].split("_")[0]
            x_train.append(
                matplotlib.pyplot.imread(f.open(name))
            )
            y_train.append(classe)
        if "val" in name:
            x_valid.append(
                matplotlib.pyplot.imread(f.open(name))
            )
            arg = name.split("/")[-1]
            print(val_classes[arg])
            y_valid.append(val_classes[arg])
        if "test" in name:
            x_test.append(
                matplotlib.pyplot.imread(f.open(name))
            )

    print(np.array(x_train).shape)
    print(np.array(x_valid).shape)
    print(np.array(y_train).shape)
    print(np.array(y_valid).shape)

    np.save(f'{path}/TinyImagenet_train_x.npy', np.array(x_train))
    np.save(f'{path}/TinyImagenet_test_x.npy', np.array(x_valid))
    np.save(f'{path}/TinyImagenet_train_y.npy', np.array(y_train))
    np.save(f'{path}/TinyImagenet_test_y.npy', np.array(x_valid))


if __name__ == '__main__':
    load()