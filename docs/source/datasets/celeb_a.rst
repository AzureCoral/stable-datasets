Celeb-A
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Attributes-40-green" alt="Attributes: 40">
   <img src="https://img.shields.io/badge/Size-HxWx3-orange" alt="Image Size: HxWx3">
   </p>

Overview
--------

The CelebA dataset (CelebFaces Attributes Dataset) contains over 200,000 celebrity images, with each image annotated with 40 different attributes. The dataset covers large variations in pose, background, and identity, and is commonly used for face attribute recognition and representation learning.

- **Train**: 162,770 images
- **Validation**: 19,867 images
- **Test**: 19,962 images

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - H×W×3 RGB image
   * - ``attributes``
     - ``Dict[str, int]``
     - Binary attributes [-1, 1]

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.celeb_a import CelebA

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = CelebA(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = CelebA(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

References
----------

- Official dataset page: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Citation
--------

.. code-block:: bibtex

    @inproceedings{liu2015faceattributes,
        title     = {Deep Learning Face Attributes in the Wild},
        author    = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
        booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
        month     = {December},
        year      = {2015}
    }

