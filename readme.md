## Implementation of Resnet50

### Architecture

<p align="center">
    <img src="assets/resnet50.png" alt="drawing" width="700"/>
</p>

### How does it work ?

I used tensorflow for this implementation.

You can find examples of how to use it in the `Resnet_Example.ipynb` file.

You can find the weights of the model trained on an hand signs dataset in the `weights` folder.

There is also pretrained weights that I didn't used in the example in `.h5` format in the weights folder named `resnet50.h5`

### Files

The model implementation is in the `resnet50.py` file. There is also several useful functions for Layers and datasets related import functions in the resnets_utils.py

### Ressources

Original paper : https://arxiv.org/abs/1512.03385v1

I used the implementation I did for an exercices from Andrew Ng's course from Coursera : https://www.coursera.org/specializations/deep-learning