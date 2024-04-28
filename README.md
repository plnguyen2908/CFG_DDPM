<div align="center">    
 
# Classifier-Free Guidance Diffusion Model

<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->

<!--
Conference
-->
</div>
 
## Description   
### Introduction

There has been an explosion in current trend of generative model ranging from GPT, VAE, GAN, diffusion model, and their variations. This is our project exploring the variation of diffusion model where you do not need a classifier jointly trained.

#### Goals

The goal of this project is to train a diffusion model with linear noise scheduling and without a classifier guidance. Also, we include a guide on how to play around with our model.

## How to run

First, upload all files and folder into a folder on google drive.

## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

## Reference
