# Image Classification SavedModel format example

## Requirements

-   Tensorflow 2.0+ C API (use [Nightly build](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/c) if no official release is available)
-   Tensorflow 2.0+ for Python (You can use pip and do something like `pip install --upgrade tensorflow==2.2.0`, check the latest version [here](https://pypi.org/project/tensorflow/#history))

## Dataset

Download and exatract the [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification/data) dataset into the `Dataset/` folder.

## Full SavedModel Format Support

I didn't add support for [signature definitions](https://www.tensorflow.org/guide/saved_model#specifying_signatures_during_export) on the C++ side because there is no support for it on the C API headers, even on the nightly build file.

However, once they release Tensorflow 2 C API, it should be possible to add such options. The code submitted here is a first step to using it, and uses default signature and the default tag (`serving` tag).
