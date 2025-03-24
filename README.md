# BayesHydroCNN

**Bayesian Calibration of a 2D Hydraulic Model using CNN-based Emulation with Spatial Weighting**

This repository implements a novel approach for the Bayesian calibration of Manning’s roughness coefficients in a 2D hydraulic model using a convolutional neural network (CNN) emulator. The model uses spatial correlation maps to weight learning and performs posterior sampling using PyMC and JAX/Flax.

---

## 🧠 Highlights

- **CNN-based emulator** for hydraulic simulation
- **Bayesian inference** using PyMC v5 and custom PyTensor `Op`
- **Spatial correlation weighting** during training
- Application to real flood events with shapefile validation

---

## 📁 Folder Overview

| Folder         | Description                                       |
|----------------|---------------------------------------------------|
| `data/`        | Raw, processed, and sample data inputs            |
| `notebooks/`   | Jupyter Notebooks for step-by-step workflows      |
| `src/`         | All core code: preprocessing, modeling, calibration |
| `tests/`       | Unit tests for key functions                      |

---

## 🚀 Quickstart

1. Clone the repo:

```bash
git clone https://github.com/your-username/BayesHydroCNN.git
cd BayesHydroCNN

conda create -n bayeshydrocnn python=3.10
conda activate bayeshydrocnn
pip install -r requirements.txt

jupyter notebook notebooks/01_data_exploration.ipynb

