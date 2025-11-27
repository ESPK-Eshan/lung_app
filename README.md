Lung Disease Classification — Streamlit App

Author: Eshan Puri (8025340013)

A compact, deployable Python application that provides inference and a lightweight UI for classifying chest X-ray images into five categories. The app loads pre-trained PyTorch models and offers single-image and batch prediction, image-quality checks, probability visualization, simple interpretability, and a downloadable medical-style PDF report.

Key features

Single-image inference (DenseNet121 / EfficientNet-B0 / ResNet50 / Ensemble)

Batch prediction with CSV export

Image quality checks (blur, brightness, contrast) and optional enhancement

Test-time augmentation (TTA) option

Probability bar chart and per-class probability table

Model evaluation page (confusion matrix / ROC images from results/)

PDF report generator for each prediction

Dark theme via .streamlit/config.toml

Dataset Link- https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types
