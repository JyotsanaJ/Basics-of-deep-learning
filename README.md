# Deep dive into Deep Learning 
This repository provides Jupyter notebooks aimed at deepening your understanding of Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers, three fundamental architectures in deep learning.

## Notebooks Overview
------------------

1.  **Understanding CNNs**
    
    *   This notebook covers the basics of Convolutional Neural Networks, including convolutional layers, pooling layers, and fully connected layers.
    *   Topics include:
        *   Convolutional operation
        *   ReLU activation
        *   Pooling layers (MaxPooling, AveragePooling)
        *   Training and evaluation of CNNs on image datasets
        *   Visualizing learned filters and feature maps

2.  **Exploring RNNs**
    
    *   This notebook delves into Recurrent Neural Networks, focusing on their sequential nature and ability to handle sequential data such as text and time series.
    *   Topics include:
        *   Vanilla RNNs
        *   Long Short-Term Memory (LSTM) networks
        *   Gated Recurrent Units (GRUs)
        *   Training and evaluation of RNNs on text datasets
        *   Text generation using RNNs

3.  **Unraveling Transformers**
    
    *   This notebook introduces Transformers, a revolutionary architecture in deep learning known for its success in natural language processing (NLP) and computer vision tasks.
    *   Topics include:
        *   Self-attention mechanism
        *   Multi-head attention
        *   Positional encoding
        *   Transformer encoder and decoder architectures
        *   Training and evaluation of Transformers on language translation tasks
        *   Fine-tuning pre-trained Transformers models for downstream tasks

## Requirements
------------

*   Python 3.x
*   Jupyter Notebook
*   PyTorch
*   TensorFlow/Keras (for CNN and RNN notebooks)
*   Hugging Face Transformers library (for Transformer notebook)

Install the required dependencies using pip:

bash
`pip install jupyter torch torchvision tensorflow transformers`

Usage
-----

1.  Clone the repository:

bash
`git clone https://github.com/your-username/Deep-Dive-into-Deep-Learning.git cd Deep-Dive-into-Deep-Learning`

2.  Launch Jupyter Notebook server:

bash
`jupyter notebook`

3. Open the desired notebook :

    3.a. *customer_churn.ipynb* : Training a basic CNN model using tensorflow to predict customer churn

    3.b. *finetune_gpt2_using_pytorch.ipynb* : Finetuning a gpt2 model using pytorch to translate english to french

    3.c. *finetune_t5_using_pytorch.ipynb* : Finetuning a gpt2 model using pytorch to generate text summary

    3.d. *image-pixel.ipynb* : Basic libraries and usage to read image as input

    3.e. *mnist_classification.ipynb* : Training a basic CNN model using tensorflow to predict digits in mnist

    3.f. *multilayer_perceptron.ipynb* : Having fun with multilayer perceptron MLP

    3.g. *perceptron.ipynb* : understand the basic working on a perceptron

    3.h. *pytorch_finetune_CNN.ipynb* : Finetuning a CNN model using pytorch

    3.i. *rnn.ipynb* : Basics of RNN and LSTM 

    3.j. *Transformers_.ipynb* : Understanding of Transformers

    3.k. *vgg_tensor.ipynb* : Architecture of a VGG model using tensorflow