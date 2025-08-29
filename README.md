# QuantumFed: A Framework for Quantum Federated Learning in Cybersecurity

**QuantumFed** is an open-source research framework designed to explore the intersection of **Federated Learning (FL)**, **Quantum Machine Learning (QML)**, and **Cybersecurity**.  
It provides a flexible and extensible platform for training quantum-enhanced models on distributed data for tasks such as **intrusion detection, malware analysis, and anomaly detection**.  

> *This project is currently under active development as part of a PhD research project.*

---

## Key Features

- **🚀 Modular & Extensible**  
  Add new datasets, models (quantum or classical), and federated strategies without altering the core codebase.

- **⚛️ Quantum-Ready**  
  Built on top of [PennyLane](https://pennylane.ai/) for seamless integration of QML models with PyTorch.

- **🌐 Federation-Powered**  
  Uses [Flower](https://flower.dev/) for a robust, scalable federated learning backend.

- **⚙️ Config-Driven**  
  Experiments are managed via YAML configs → reproducible, transparent, and easy to iterate.

- **💪 Robust**  
  Includes built-in checks for issues such as handling very small datasets in stratified splitting.

---

## Project Status

- [x] **Project Scaffolding & Setup** → Directory structure, Git repository, Conda environment.  
- [x] **Data Loading Module** → Flexible pipeline for data loading & preprocessing.  
- [ ] **Model Module** → Define base classes and initial QNN (Quantum Neural Network) models.  
- [ ] **Federated Learning Logic** → Implement Flower client and server logic.  
- [ ] **Configuration Management** → Integrate YAML configs for experiment control.  
- [ ] **End-to-End Pipeline** → Connect all modules to run a full federated training round.  

---

## Getting Started

### 1. Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management.  
- [Git](https://git-scm.com/) for version control.  

### 2. Installation

Clone the repository and set up the Conda environment:

```bash
# Clone the repository
git clone https://github.com/ailabteam/quantumfed.git
cd quantumfed

# Create and activate the conda environment
conda create --name qfed python=3.10
conda activate qfed

# Install dependencies
pip install -r requirements.txt
````

---

### 3. Running the Data Module Test

To verify that the **data loading module** works correctly, run `main.py`.
This will load a sample dataset, preprocess it, and print the resulting data array shapes.

```bash
# Make sure the sample data file is created
# (See quantumfed/data/datasets.py for details — automation coming soon)

# Run the test
python main.py
```

You should see output confirming **successful data loading and preprocessing**.

---
