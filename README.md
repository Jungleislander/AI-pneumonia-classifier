[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omarsagoo/AI-pneumonia-classifier/)

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
2. Click the **"Open in Colab"** badge at the top of the notebook *(or use the badge at the top of the readme)*:

> ⚠️ Note: Make sure to copy the dataset to your own Google Drive or mount it in the notebook if running locally. We recommend working with a shared Drive folder linked to Colab for persistent access.

### Version Control
We are using GitHub for versioning and team collaboration:
- All team members push and pull from the main repository.
- Code changes are committed with clear messages and organized by notebook sections.
- We periodically export `.ipynb` files from Colab and sync them back to the repo.


## 📁 Project Structure

**AI-pneumonia-classifier/**  

├── chest_xray/ – Local copy of the dataset (if used outside Kaggle/Colab)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── test/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── val/  
├── notebooks/ – Jupyter/Colab notebooks for EDA, modeling, etc.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── utils/ – Utility scripts (e.g., preprocessing functions)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── preprocessor.py - Adds utility functions for data preprocessing.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── [preprocessing.ipynb](https://colab.research.google.com/github/omarsagoo/AI-pneumonia-classifier/blob/main/notebooks/preprocessing.ipynb)  
├── models/ – Trained model files (e.g., `.h5`, `.pkl`)  
├── outputs/ – Plots, confusion matrices, logs  
├── presentation/ – Slide deck and recorded video (PDF, MP4)  
├── report/ – Final overall report  
└── README.md – Project summary and documentation

## Prerequisites: 

### Python Version
For this project we are using Python version 3.12.2, conda automatically will install and set the correct python version for the project so there is nothing that needs to be done.

### 1. Install Miniconda

If you are already using Anaconda or any other conda distribution, feel free to skip this step.

Miniconda is a minimal installer for `conda`, which we will use for managing environments and dependencies in this project. Follow these steps to install Miniconda or go [here](https://docs.anaconda.com/miniconda/install/) to reference the documentation: 

1. Open your terminal and run the following commands:
```bash
   $ mkdir -p ~/miniconda3

   <!-- If using Apple Silicon chip M1/M2/M3 -->
   $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
   <!-- If using intel chip -->
   $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda3/miniconda.sh

   $ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   $ rm ~/miniconda3/miniconda.sh
```

2. After installing and removing the installer, refresh your terminal by either closing and reopening or running the following command.
```bash
$ source ~/miniconda3/bin/activate
```

3. Initialize conda on all available shells.
```bash
$ conda init --all
```

You know conda is installed and working if you see (base) in your terminal. Next, we want to actually use the correct environments and packages.

### 2. Install Make

Make is a build automation tool that executes commands defined in a Makefile to streamline tasks like compiling code, setting up environments, and running scripts. [more information here](https://formulae.brew.sh/formula/make)

#### Installation

`make` is often pre-installed on Unix-based systems (macOS and Linux). To check if it's installed, open a terminal and type:
```bash
make -v
```

If it is not installed, simply use brew:
```bash
$ brew install make
```

#### Available Commands

The following commands are available in this project’s `Makefile`:

- **Set up the environment**:

    This will create the environment from the environment.yml file in the root directory of the project.

    ```bash
      $ make create
    ```

- **Update the environment**:

    This will update the environment from the environment.yml file in the root directory of the project. Useful if pulling in new changes that have updated the environment.yml file.

    ```bash
      $ make update
    ```

- **Remove the environment**:

    This will remove the environment from your shell. You will need to recreate and reinstall the environment with the setup command above.

    ```bash
    $ make clean
    ```

- **Activate the environment**:

    This will activate the environment in your shell. Keep in mind that make will not be able to actually activate the environment, this command will just tell you what conda command you need to run in order to start the environment.

    Please make sure to activate the environment before you start any development, we want to ensure that all packages that we use are the same for each of us.

    ```bash
    $ make activate
    ```

    Command you actually need to run in your terminal:
    ```bash
    $ conda activate ai-pneumonia-classifier
    ```

- **Deactivate the environment**:

    This will Deactivate the environment in your shell.

    ```bash
    $ make deactivate
    ```

- **run jupyter notebook**:

    This command will run jupyter notebook from within the conda environment. This is important so that we can make sure the package versions are the same for all of us! Please make sure that you have activated your environment before you run the notebook.

    ```bash
    $ make notebook
    ```

- **Export packages to env file**:

    This command will export any packages you install with either `conda install ` or `pip install` to the environment.yml file. This is important because if you add any packages we want to make sure that everyones machine knows to install it.

    ```bash
    $ make freeze
    ```

- **Verify conda environment**:

    This command will list all of your conda envs, the environment with the asterick next to it is the currently activated one. Ensure it is correct.

    ```bash
    $ make verify
    ```

#### Example workflows:

To simplify knowing which commands you need to run and when you can follow these instructions:

- **First time running, no env installed**:

    In the scenario where you just cloned this repo, or this is your first time using conda. These are the commands you will run to set up your environment.

    ```bash
    <!-- Make sure that conda is initialized -->
    $ conda init --all

    <!-- Next create the env from the env file in the root directory. -->
    $ make create

    <!-- After the environment was successfully created, activate the environment. -->
    $ conda activate ai-pneumonia-classifier

    <!-- verify the conda environment -->
    $ make verify

    <!-- verify the python version you are using. This should automatically be updated to the correct version 3.12.2 when you enter the environment. -->
    $ python --version

    <!-- Run jupyter notebook and have some fun! -->
    $ make notebook
    ```

- **Installing a new package**:

    While we are developing, we are going to need to install certain packages that we can utilize. Here is a sample workflow for installing packages. The first thing we do is verify the conda environment we are in to ensure that only the required packages get saved to the environment. We do not want to save all of the python packages that are saved onto our system to the `environment.yml` file. 

    Another thing to note is that if the package is not found in the conda distribution of packages you will get a `PackagesNotFoundError`. This is okay, just use pip instead of conda to install that specific package. Conda thankfully adds them to the environment properly.

    ```bash
    <!-- verify the conda environment -->
    $ make verify

    <!-- Install the package using conda -->
    $ conda install <package_name>

    <!-- If the package is not found in the conda channels, install the package with pip. -->
    $ pip install <package_name>

    <!-- If removing a package. -->
    $ conda remove <package_name>
    $ pip remove <package_name>

    <!-- Export the package names and versions that you downloaded to the environment.yml file -->
    make freeze
    ```

- **Daily commands to run before starting development**:

    Here is a sample workflow for the commands to run before starting development on any given day. We want to first pull all the changes from github into our local repository, 

    ```brew
    <!-- Pull changes from git -->
    $ git pull origin main

    <!-- Update env based off of the env file. It is best to deactivate the conda env before you do this step-->
    $ conda deactivate
    $ make update
    $ conda activate ai-pneumonia-classifier

    $ make notebook
    ```

- **Daily commands to run after finishing development**:

    Here is a sample workflow for the commands to run after finishing development for any given day.

    ```brew
    $ conda deactivate

    <!-- If you updated any of the existing packages, freeze to the environment.yml file. -->
    $ make freeze

    <!-- Commit changes to git -->
    $ git add .
    $ git commit -m "This is my commit message!"
    $ git push origin <branch_name>
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