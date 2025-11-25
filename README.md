This repository contains materials for a hands-on lab session that is part of a course on **Data Assimilation**. The goal is to discover how neural networks can be designed to emulate and learn the dynamics of chaotic systems—especially when informed by physical structure or numerical integration methods.

We experiment with three types of models:

* **NaiveNetwork**: a basic multi-layer perceptron (MLP)
* **ConvolutionalNetwork**: a spatially-aware model using Conv1D
* **HybridNetwork**: a physics-informed architecture with Runge-Kutta integration

## Contents

* `lorenz96_surrogate.ipynb`: Jupyter notebook guiding the lab. You can run it in Google Colab by clicking the link below:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/relmonta/lorenz96/blob/main/lorenz96_surrogate.ipynb)
* `lorenz96_surrogate_student.ipynb`: Jupyter notebook for students to complete. You can run it in Google Colab by clicking the link below:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/relmonta/lorenz96/blob/main/lorenz96_surrogate_student.ipynb)
* `lorenz96_models.py`: neural network model definitions
* `lorenz96_utils.py`: helper functions and data generation process
* `lorenz96_trainer.py`: training and evaluation functions