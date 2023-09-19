# Fader Networks with TensorFlow

This repository presents our machine learning project, which is based on the Fader Networks concept introduced by Lample et al. (2017). Fader Networks utilize an encoder-decoder architecture to modify attributes of real images, such as gender, age, and adding glasses, while preserving the underlying character of the image.

## Introduction

Fader Networks enable the manipulation of natural images by controlling specific attributes, generating various realistic versions of images by altering attributes like gender or age group. These networks can smoothly transition between different attribute values and are built upon an encoder-decoder architecture trained to reconstruct images.

## Authors

- Youcef CHORFI
- Gontran GILLES
- Sara NAIT ATMAN
- Nathaniel DAHAN

## Dependencies

- Python 2/3
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [TensorFlow](https://www.tensorflow.org/)
- OpenCV
- CUDA

## Architecture

### Fader Architecture
![Fader Architecture](./results/FaderNetworks1.png)

### Encoder Architecture
![Encoder Architecture](./results/Encoder_Diagram.png)

### Decoder Architecture
![Decoder Architecture](./results/Decoder_Diagram.png)

### Discriminator Architecture
![Discriminator Architecture](./results/Discriminateur_Diagram1.png)

## Model Results
![Model Results](./results/results.jpg)

## Model Stats
![Model Stats](./results/loss.png)

## Data

The dataset used for this project is CelebA, which can be downloaded from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Classifier

Training the classifier is essential for the training of the Fader Network. To train the classifier, run the following command:
```bash
./classifier.py
```

## Train Fadernetwork 
Specify the name of the trained classifier folder in the params.yaml file, then train the Fader Network using the following command:
```bash
./fader_train.py
```

## Inference
Specify the name of the trained Fader Network folder in the params.yaml file, then run the following command to see the results:
```bash
./main.py #to see results
```
