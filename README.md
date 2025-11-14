# Face-Forgery-Detection
A hybrid deepfake detection model that combines spatial CNN features and frequency-domain FFT features to classify real vs fake facial images.

# Hybrid Face Forgery Detection  
*A Deep Learning Model for Detecting Real vs Fake Facial Images Using Spatial + Frequency Features*

---

## Overview
This repository contains the implementation of a **Hybrid Face Forgery Detection Model** that combines:

- **Spatial features** extracted using a ResNet-based **FaceXRayHRNet** CNN  
- **Frequency features** extracted using **Fast Fourier Transform (FFT)**  

By fusing both domains, the model improves robustness against modern deepfake generation methods and identifies subtle manipulation artifacts that spatial-only or frequency-only detectors may miss.

---

## Key Features
- Hybrid deepfake detection (Spatial + Frequency)  
- FaceXRayHRNet backbone for spatial artifact extraction  
- FFT-based frequency feature extraction  
- Feature fusion with fully connected classification  
- Training + evaluation pipeline included  
- Clean modular code structure  
- Supports custom datasets of real and fake images

---

## Model Architecture
The proposed architecture consists of three main components:

1. **Spatial Feature Extractor**  
   - ResNet-based FaceXRayHRNet  
   - Extracts pixel-level inconsistencies

2. **Frequency Feature Extractor**  
   - Computes FFT magnitude spectrum  
   - Captures high-frequency abnormalities

3. **Fusion + Classifier**  
   - Concatenates spatial & frequency representations  
   - Fully connected layer outputs real/fake prediction

A visual diagram of the architecture is included in the repository.

---

## Project Structure


---

## Dataset
A custom dataset of **10,000 images** was used:

- **5,000 real images**
- **5,000 fake images**

Images were preprocessed to **256 × 256 × 3** and normalized.  
Dataset split: **80% training** and **20% testing**.

> *(Note: Due to size and copyright, the dataset is not included in this repository.)*

---

## Training
Run the training script:

```bash
python train.py

