Road Sign Recognition (GTSRB) using Transfer Learning
 Project Overview

This project builds a Road Sign Recognition system using the GTSRB (German Traffic Sign Recognition Benchmark) dataset.
The model is trained in Google Colab using MobileNetV2 transfer learning for high accuracy and fast inference.

Dataset we use 
Source: Kaggle – GTSRB: German Traffic Sign Recognition
Structure after extraction:



The dataset contains 43 traffic sign classes.

Model Architecture

The system uses MobileNetV2 (pretrained on ImageNet) with:

Frozen convolutional layers (initial training)

Global Average Pooling

Dropout regularization

Dense softmax classification layer

Fine-tuning entire model at low learning rate

 Training Pipeline

Install dependencies and set up Kaggle API

Download and extract dataset

Load images with image_dataset_from_directory

Create Train / Validation / Test splits

Apply data augmentation

Train MobileNetV2 (frozen)

Fine-tune entire model

Evaluate on test set

Save model

Run an interactive Gradio demo

Evaluation Metrics

The model evaluation includes:

Overall Accuracy

Precision / Recall / F1-Score per class

Confusion Matrix
Demo

A simple Gradio interface allows users to upload an image and receive:

Predicted sign class

Confidence score

How to Run (Colab)

Upload kaggle.json to Colab

Run the notebook cells in order

Train the model

Test using the Gradio demo
 Output Files

gtsrb_mobilenetv2_model/ — saved TensorFlow model

Classification report (printed in notebook)

Gradio interface for real-time testing
 Requirements

Python 3.9+

TensorFlow 2.x

Scikit-learn

Gradio

Kaggle API access
 Notes

Training time depends on Colab GPU availability

Additional fine-tuning can increase accuracy

The model can be exported to TFLite for mobile applications