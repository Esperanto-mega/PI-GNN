# PI-GNN

Welcome to the **PI-GNN** repository. 

## 📂 Repository Structure

The project is organized into the following directories and files:

* **`data-generator/`**: Contains scripts and utilities for generating synthetic datasets or processing raw data into graph structures suitable for the GNN.
* **`dataset/`**: The storage directory for the generated or downloaded datasets used during model training and evaluation.
* **`main/`**: Contains the core execution scripts (e.g., `main.py`) to launch the training, validation, and testing loops.
* **`utils/`**: Helper functions, configuration loaders, and other miscellaneous tools required across the project.
* **`model.py`**: The core Graph Neural Network architecture definitions and PyTorch/framework model classes.

## 🚀 Features

* **Custom Data Generation**: Easily create and format complex graph datasets tailored to your specific problem domain.
* **Modular Architecture**: Model definitions (`model.py`), utilities (`utils/`), and execution scripts (`main/`) are kept separate for clean code management.
* **End-to-End Pipeline**: From raw data generation to final model inference.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Esperanto-mega/PI-GNN.git](https://github.com/Esperanto-mega/PI-GNN.git)
   cd PI-GNN
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   conda create -n pignn_env python=3.8
   conda activate pignn_env
   ```

3. **Install Dependencies:**
   Make sure you have the appropriate Deep Learning frameworks installed (such as PyTorch and PyTorch Geometric). 
   ```bash
   # Example for pip-based installation
   pip install -r requirements.txt 
   ```
   *(Note: If a `requirements.txt` is not present, manually install your required packages like `torch`, `torch-geometric`, `numpy`, and `pandas`).*

## 💻 Usage

### 1. Generate / Prepare Data
First, run the data generator scripts to populate the `dataset/` directory with the necessary graph data.
```bash
python data-generator/generate.py
```

### 2. Train the Model
Navigate to the `main/` directory (or run from the root) to start the training process. You can likely pass arguments to customize the training parameters.
```bash
python main/train.py --epochs 100 --batch_size 32
```

### 3. Model Architecture
If you need to tweak the GNN layers, modify the classes inside `model.py`. 

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Esperanto-mega/PI-GNN/issues) if you want to contribute.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
