# TripletFusionNet

**A Hybrid Graph–Spectral Framework for Predicting Molecular Triplet Lifetimes and Screening TTA-UC Systems**

This repository contains the source code and datasets for the paper:
> **TripletFusionNet: A Hybrid Graph–Spectral Framework for Predicting Molecular Triplet Lifetimes and Screening TTA-UC Systems**
>
> *Xiaoyu Yue, Jun Jang, Song Wang\**
>
> State Key Laboratory of Precision and Intelligent Chemistry, University of Science and Technology of China (USTC).

---

## 📖 Overview

**TripletFusionNet** is a machine-learning framework designed to predict molecular triplet lifetimes ($\tau_T$) and accelerate the discovery of high-performance sensitizers for Triplet–Triplet Annihilation Upconversion (TTA-UC).

The framework integrates:
*   **Graph Neural Networks (GNN):** To capture molecular topological features.
*   **IR Spectra:** To capture vibrational signatures (identified as dominant predictors).
*   **Molecular Descriptors:** RDKit-derived physicochemical properties.

By combining these features, the model achieves robust accuracy ($R^2 = 0.78$) and has been used to screen a virtual library of over 4,000 candidates, identifying promising long-lived sensitizers.

---

## 📂 Repository Structure

The repository is organized into data processing, model training, and virtual screening workflows.

### Root Directory (Model Training & Validation)
The following notebooks are numbered according to the execution workflow for the training set:

*   **`1.DScribe_gen_des.ipynb`**: Generates fundamental molecular descriptors using `RDKit` and `DScribe`.
*   **`2.Dimensional reduction.ipynb`**: Performs dimensionality reduction (PCA) on the generated descriptors.
*   **`3.Extract_IR.ipynb`**: Extracts Infrared (IR) spectral data (frequencies and intensities) from Gaussian output logs.
*   **`4.Extract_homo-lumo-gap.ipynb`**: Extracts electronic features (HOMO, LUMO, Energy Gap, etc.) from Gaussian output logs.
*   **`5.GNN.ipynb`**: Trains the Graph Neural Network (GNN) to generate 32-dimensional molecular embeddings.
*   **`6.predict.ipynb`**: The final Hybrid Model (GNN + RF). Combines GNN embeddings, IR spectra, and descriptors to predict triplet lifetimes.

### Virtual Screening (New Molecules)
Files related to the generation and screening of the new molecular library (4,435 candidates):

*   **`generate_new_molecules/`**: Folder containing generation scripts.
    *   `1. syn-smiles.ipynb`: Generates new candidate SMILES using reaction templates (substituent addition).
    *   `2. gen_3D.ipynb`: Generates 3D conformers for the new SMILES.
    *   `data/`: Contains IR data and properties for the new library.
*   **`7.[generate_new_molecules]DScribe_gen_des.ipynb`**: Generates descriptors for the *new* molecular library.
*   **`8.[generate_new_molecules]GNN.ipynb`**: Generates GNN embeddings for the *new* molecular library.
*   **`9.[generate_new_molecules]predict.ipynb`**: Performs final lifetime prediction on the new library to screen for high-performance candidates.

### Data
*   **`data/`**: Contains input/output files for the main model training (CSV files, extracted features, etc.).

> **Note:** This repository focuses on the ML framework (TripletFusionNet). The Molecular Dynamics (MD) simulations and electronic coupling calculations mentioned in the paper were performed using GROMACS and ORCA separately and are not included in this codebase.

---

## ⚙️ Dependencies

To run the notebooks, you will need the following Python packages:

*   **Python 3.8+**
*   **PyTorch** (for GNN implementation)
*   **RDKit** (for molecular handling and descriptor generation)
*   **DScribe** (for advanced descriptors)
*   **Scikit-learn** (for Random Forest and PCA)
*   **Pandas & NumPy** (for data manipulation)
*   **Matplotlib/Seaborn** (for visualization)

---

## 🚀 Usage

### 1. Training the Model (Files 1-6)
To reproduce the model training and validation results:
1.  Ensure raw data (Gaussian logs or extracted CSVs) is placed in `data/`.
2.  Run notebooks `1` through `4` to preprocess data and extract features.
3.  Run `5.GNN.ipynb` to train the graph encoder and save embeddings.
4.  Run `6.predict.ipynb` to train the final Random Forest regressor and evaluate performance.

### 2. Screening New Candidates (Files 7-9)
To screen the virtual library:
1.  Use the scripts in `generate_new_molecules/` to create the library.
2.  Run notebooks `7` through `9` sequentially to generate features and predict lifetimes for the new candidates.

---

## 📄 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{TripletFusionNet2024,
  title={TripletFusionNet: A Hybrid Graph–Spectral Framework for Predicting Molecular Triplet Lifetimes and Screening TTA-UC Systems},
  author={Yue, Xiaoyu and Jang, Jun and Wang, Song},
  journal={Submitted},
  year={2024},
  publisher={University of Science and Technology of China}
}
