Multi-Class Lung Disease Classification from Chest X-rays using Deep Learning

Deep learning system for multi-class lung disease classification from chest X-ray images using transfer learning and ensemble learning.

The project evaluates multiple CNN architectures and builds an ensemble model to improve robustness and diagnostic performance.

Project Overview

Chest X-ray analysis is a critical tool for diagnosing lung diseases such as pneumonia, tuberculosis, and COVID-19. However, distinguishing between these diseases is challenging due to high inter-class similarity and subtle radiographic patterns.

This project develops an end-to-end deep learning pipeline to classify lung diseases from chest X-rays into five categories.

Classes:

Normal

Viral Pneumonia

Bacterial Pneumonia

Tuberculosis

COVID-19

Model Architecture

The system evaluates three convolutional neural network architectures:

ResNet-50

DenseNet-121

EfficientNet-B0

All models are initialized with ImageNet pretrained weights and fine-tuned for the medical imaging task.

Final predictions are generated using a soft-voting ensemble model combining the best-performing architectures.

Training Pipeline

The training process follows a two-stage transfer learning approach.

Stage 1: Frozen Backbone

CNN backbone frozen

Train classification head

Stage 2: Fine-Tuning

Unfreeze backbone

Train entire network with low learning rate

Optimization techniques used:

AdamW optimizer

Cosine annealing learning rate scheduler

Automatic Mixed Precision (AMP)

Test-Time Augmentation (TTA)

Early stopping and checkpointing

Dataset

The dataset consists of chest X-ray images belonging to five disease categories.

Dataset distribution:

Class	Samples
Tuberculosis	      1220
COVID-19	          1218
Normal	            1207
Bacterial Pneumonia	1205
Viral Pneumonia	    1204

Images were resized to 224×224 resolution before training.

Results
Model	Test Accuracy	ROC-AUC	Parameters
EfficientNet-B0	84.71%	0.9765	5.29M
DenseNet-121	  83.03%	0.9654	7.98M
ResNet-50	      76.81%	0.9500	25.56M
Ensemble Model	85.40%	  —	      —

The ensemble model achieved the best performance by combining predictions from EfficientNet-B0 and DenseNet-121.


Tech Stack

Python

PyTorch

timm

NumPy

Pandas

Matplotlib

scikit-learn

Future Work

Potential improvements include:

Vision Transformer architectures

Larger medical datasets

Model explainability techniques

Clinical validation studies

Author
Eshan Puri
M.E. Artificial Intelligence
Thapar Institute of Engineering and Technology

Dataset Link- https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types

<img width="2752" height="1317" alt="Gemini_Generated_Image_mowtenmowtenmowt" src="https://github.com/user-attachments/assets/5da5fe44-39f3-402a-af75-2d59e5566a2b" />
