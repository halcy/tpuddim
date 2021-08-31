# tpuddim

Denoising Diffusion Implicit Models JAX TPU implementation. Based on the network architecture in https://github.com/openai/guided-diffusion , pretrained weights compatible.

Still quite WIP, though less so now, there is code for inference and basic training, weights for the MNIST example, an initial training attempt on Danbooru2019Faces and a [convenient colab inference notebook](https://colab.research.google.com/github/halcy/tpuddim/blob/main/colab/Danbooru2019Portraits_Inference_Colab.ipynb) for the same.

There's also an attempt at making a mixer work as a diffusion model, that doesn't quite work yet. Patches extremely welcome, especially if they adress one of the things marked TODO in the code somewhere.

MNIST example output:

![generated mnist digits](https://github.com/halcy/tpuddim/blob/main/mnist_example.png?raw=true)

Danbooru2019Faces example output:

![generated mnist digits](https://github.com/halcy/tpuddim/blob/main/danbooru_sample.png?raw=true)
