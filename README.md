# <img src="images/icon.png" style="width: 1em; height: 1em; vertical-align: middle;"> Anime Face Generator

This project applies a **Vanilla DCGAN to generate Anime girl faces**, utilizing TensorFlow to build and train the model. The goal is to create high-quality anime faces by leveraging the power of convolutional layers in both the generator and discriminator.

<p align="center">
    <img src='images/dcgan.gif'/>
</p>

This is an academic project to conclude **LAMIA's Machine Learning Bootcamp**, which involves applying a simple DCGAN architecture to generate Anime girl faces.  
DCGAN (Deep Convolutional Generative Adversarial Network) is a GAN architecture that uses deep convolutional layers in both the generator and discriminator. In a GAN, the discriminator tries to identify whether an image is real or generated, while the generator tries to deceive the discriminator into classifying the generated images as real. The DCGAN works in the same way but uses convolutional layers.
This architecture was chosen for its ability to learn complex data patterns and **generate high-quality images**.

## Framework

**Tensorflow** was chosen for its optimization and compatibility with a wide range of techniques and technologies. Since a GAN cannot be trained using the `Sequential.fit()` method, Tensorflow is very useful for customizing the model training step and process, including dataset handling, optimizations, and others tools.

## Dataset

The dataset used for this project is the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset), containing **63,632 images** with a **256x256** resolution, all scraped from [Getchu](www.getchu.com).  

### Pre-processing

The dataset was created for the goal *"I just want to generate perfect waifus"*. So, the images don't have labels and are stored in a single directory.  
The method `tensorflow.keras.utils.image_dataset_from_directory()` was used to create a dataset with the following properties:
- Images resized to **64x64**, balancing quality and computational efficiency during training.
- **128 images per batch**.
- The batches are shuffled every step.
- Images **normalized to the interval [-1, 1]**.
- 2 batches preloaded to optimize reading time.

## Architecture

The architecture used here is a simple DCGAN.
- The generator have **4 deconvolution layers**, and the discriminator have **4 convolution layers**. Both use a `stride = 2` instead of pooling layers to optimize the training time.
- **Batch normalization** is aplied in each layer of the generator and discriminator (except the output layer). This helps to stabilize and accelerate the training by normalizing the activations from the previous layer. This reduces the internal covariate shift, improving the model’s ability to learn and generalize better, and prevents overfitting.
- `LeakyReLu` is used on Deconvolution, Convolution and input layers to aviod *dead units* (commom problem in deep networks).
- `tanh` is used as the final activation of the generator because it return values between 1 and -1, which is consistent with the normalization of the input images.
- `sigmoid` used as the final activation of the discriminator. (return values between 0 and 1).
- 25% of **Dropout** is added to the two middle convolutional layers of the discriminator. This helps prevent overfitting by randomly disabling 25% of the neurons during each training step, forcing the model to learn more robust and generalized features.
### Aditional features
To maximize the learning of both the generator and the discriminator, two additional features were added to the training process:
1. The **discriminator runs twice on the real images per batch** in every epoch (except when the discriminator becomes too strong).  
2. In every x epochs, the learning rate of both is adjusted based on their loss mean.  

## Results

In GANs, **visual results are often more important than traditional metrics**. However, monitoring the losses during training is still crucial to evaluate the convergence between the discriminator and the generator.

Typically, during the initial epochs, the difference in loss between the two models can be quite large. Over time, they tend to balance out, with both losses stabilizing around 1. However, this is not a strict rule, as numerous factors can influence this behavior, such as the architecture, dataset, and training parameters.

Accuracy is also used as an additional metric to monitor the interaction between the models. An ideal discriminator accuracy hovers around 50%, indicating a well-balanced GAN where the generator produces realistic images that successfully confuse the discriminator.
### **Metrics Graph**
<p align="center">
    <img src='images/anime_face_dcgan_metrics.png'/>
</p>

### **Generated Images**
<p align="center">
    <img src='images/generated_images.png'/>
</p>
