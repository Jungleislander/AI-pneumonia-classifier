# 🩺 Doctor AI Will See You Now – Pneumonia Detection from Chest X-Rays

## 📌 Project Overview

This project explores the application of **machine learning and computer vision** techniques to assist in the **automated diagnosis of pneumonia** from chest X-ray images.

Using a dataset sourced from Kaggle, our objective is to build a model that can **accurately classify** whether a patient has pneumonia based on their X-ray, with a particular emphasis on **reducing false positives**—a key challenge noted in prior work.

We are building on the approach introduced by Amy Jang in her [TensorFlow Pneumonia Classification](https://www.kaggle.com/code/amyjang/tensorflow-pneumonia-classification-on-x-rays) notebook, and aim to improve performance by applying advanced modeling techniques and better handling of overfitting.


## 📂 Dataset

- **Name:** Chest X-Ray Images (Pneumonia)  
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Description:**  
  - 5,863 chest X-ray images (JPEG format)  
  - Two classes: `PNEUMONIA` and `NORMAL`  
  - Class imbalance (more pneumonia cases than normal)  
  - Divided into `train`, `test`, and `val` folders  


## 🎯 Objectives

- Reproduce Amy Jang’s baseline model
- Experiment with data preprocessing and augmentation strategies
- Evaluate multiple CNN architectures and transfer learning techniques
- Focus on reducing **false positives** while maintaining high accuracy
- Deliver results via a technical notebook, business presentation, and recorded demo


## 🧪 Methods & Tools

- **Frameworks:** TensorFlow, Keras
- **Techniques:** CNNs, Transfer Learning, Data Augmentation, Regularization
- **Metrics:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC-AUC
- **Tools:** Python, Jupyter/Colab, GitHub, Google Drive, Zoom

## 💻 Collaboration & Environment

This project is being developed using **Google Colab** for ease of collaboration, GPU acceleration, and seamless integration with GitHub.

### Why Google Colab?
- No local setup required
- Supports GPU/TPU for faster model training
- Easy to share and co-edit notebooks in real-time
- Direct integration with Google Drive and GitHub

### How to Access Our Notebooks
1. Open the desired notebook from the `/notebooks` directory in this repo.
2. Click the **"Open in Colab"** badge at the top of the notebook *(or use the button below)*:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jungleislander/AAI-510-01-Team-7/)

> ⚠️ Note: Make sure to copy the dataset to your own Google Drive or mount it in the notebook if running locally. We recommend working with a shared Drive folder linked to Colab for persistent access.

### Version Control
We are using GitHub for versioning and team collaboration:
- All team members push and pull from the main repository.
- Code changes are committed with clear messages and organized by notebook sections.
- We periodically export `.ipynb` files from Colab and sync them back to the repo.


## 📁 Project Structure
```
AI-pneumonia-classifier/
    ├── chest_xray/ # Local copy of the dataset (if used outside Kaggle/Colab)
    │ ├── train/
    │ ├── test/
    │ └── val/
    ├── notebooks/ # Jupyter/Colab notebooks for EDA, modeling, etc.
    │ └── preprocess.ipynb
    ├── models/ # Trained model files (e.g., .h5, .pkl)
    ├── outputs/ # Plots, confusion matrices, logs
    ├── presentation/ # Slide deck and recorded video (PDF, MP4)
    ├── report/ # Final overall report
    ├── utils/ # Utility scripts (e.g., preprocessing functions)
    └── .README.md # Project summary and documentation
```

## 🚀 Deployment Plan (To Be Finalized)

TODO

## 📌 Reference Notebooks

- Amy Jang’s original work: [TensorFlow Pneumonia Classification on X-rays](https://www.kaggle.com/code/amyjang/tensorflow-pneumonia-classification-on-x-rays)


## Contributors
<table>
  <tr>
    <td>
        <a href="https://github.com/Jungleislander.png">
          <img src="https://github.com/Jungleislander.png" width="100" height="100" alt="Steve Farmer "/><br />
          <sub><b>Steve Farmer</b></sub>
        </a>
      </td>
      <td>
        <a href="https://github.com/AtulAneja.png">
          <img src="https://github.com/AtulAneja.png" width="100" height="100" alt="Atul Aneja "/><br />
          <sub><b>Atul Aneja </b></sub>
        </a>
      </td>
     <td>
      <a href="https://github.com/omarsagoo.png">
        <img src="https://github.com/omarsagoo.png" width="100" height="100" alt="Omar Sagoo"/><br />
        <sub><b>Omar Sagoo</b></sub>
      </a>
    </td>
    <td>
      <a href="https://github.com/rdhanase.png">
        <img src="https://github.com/rdhanase.png" width="100" height="100" alt="Ramesh Dhanasekaran"/><br />
        <sub><b>Ramesh Dhanasekaran  </b></sub>
      </a>
    </td>
  </tr>
</table>