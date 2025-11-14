Road Sign Recognition (GTSRB) using Transfer Learning

This project builds a high-accuracy Road Sign Recognition System using MobileNetV2 Transfer Learning, trained on the GTSRB dataset. It classifies 43 traffic sign types and includes a real-time Gradio demo.

Project Overview

Framework: TensorFlow + Keras

Model: MobileNetV2 (ImageNet pretrained)

Dataset: GTSRB (via Kaggle)

Interface: Gradio

Training:

Phase 1 – Feature extraction (freeze base)

Phase 2 – Fine-tuning (low LR)

Steps to Run in Google Colab

Upload your kaggle.json API token.

Run all cells in order.

Train the model (feature extraction → fine-tuning).

Launch the Gradio web demo to test with images.

Dataset & Model

GTSRB: 43 classes of traffic signs.

Model Structure:

Frozen MobileNetV2 base

Global Average Pooling

Dropout

Dense softmax layer (43 units)

Fine-tuning: Unfreeze MobileNetV2 and train with low LR for improved accuracy.

Evaluation

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

Real-time prediction via Gradio

Team Members

Hailye H/Giworgis — DBUE/0746/13

Adamu Abebaw — DBUE/0701/13

Tsige Tilahun — DBUE/0788/13

Yemisrach Girma — DBUE/0792/13

Leul Aschenaki — DBUE/0754/13

Requirements

Python 3.9+

TensorFlow 2.x

Gradio

Scikit-learn

Kaggle API
