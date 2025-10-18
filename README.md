# COVID-19 X-ray Image Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/) [![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-green.svg)](https://gradio.app/)

This project is a machine learning application for classifying chest X-ray images to detect signs of COVID-19 or pneumonia using a custom Convolutional Neural Network (CNN). It includes a Jupyter notebook for data exploration and model training, and a Gradio web app for real-time predictions.

The model classifies X-ray images into **Normal** (healthy lungs) or **COVID-19/Pneumonia** (affected lungs). It was trained on a dataset of chest X-rays and achieves high accuracy on small validation and test sets.

## Features

- **Data Exploration & Model Building**: A Kaggle notebook for loading, preprocessing, training, and evaluating the CNN model.
- **Custom CNN Model**: A PyTorch-based `COVIDCNN` model for binary image classification.
- **Gradio Web App**: Upload an X-ray image to get instant probability predictions for Normal vs. COVID-19.
- **High Performance**: Achieves 100% accuracy on validation (30 images) and test (40 images) sets.
- **Easy Deployment**: Run the app locally or deploy it online.

## Dataset

The model uses a dataset of chest X-ray images, likely sourced from public datasets like the [Chest X-Ray Images (Pneumonia) dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). It includes:

- **Normal**: Healthy chest X-rays.
- **Pneumonia/COVID-19**: X-rays showing lung opacities indicative of infection.

Images are preprocessed by resizing to 224x224 pixels, converting to grayscale, and normalizing.

## Model Details

- **Architecture**: Custom `COVIDCNN` with ~26 million parameters, built from scratch in PyTorch.
- **Training**: Trained for 25 epochs using the Adam optimizer and cross-entropy loss. Achieves up to 99% training accuracy and 100% validation accuracy.
- **Evaluation Metrics** (from `covid19_xray_results.json`):
  - Validation: 100% accuracy, precision, recall, and F1-score for both classes.
  - Test: 100% accuracy, precision, recall, and F1-score for both classes.
- **Saved Model**: Best model saved as `models/best_covid_model.pth`.

See the [Kaggle Notebook](https://www.kaggle.com/code/ahmedmgelwan/covid-19-xray-image-classification-complete-nb) for full training details.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ahmedmgelwan/covid19-xray-classifier.git
   cd covid19-xray-classifier
   ```

2. Install dependencies:

   ```bash
   pip install torch torchvision torchaudio gradio pillow scikit-learn matplotlib
   ```

   Ensure PyTorch is compatible with your CUDA version if using a GPU.

## Usage

### Running the Notebook

- Open `covid-19-xray-image-classification-complete-nb.ipynb` in Jupyter or Kaggle.
- Follow the steps to explore data, train the model, and save the best checkpoint.

### Running the Gradio App

1. Ensure the model file (`models/best_covid_model.pth`) is in the `models` folder.

2. Run the app:

   ```bash
   python app.py
   ```

3. Open the local URL (e.g., [http://127.0.0.1:7860](http://127.0.0.1:7860/)) in your browser.

4. Upload a chest X-ray image to view predictions.

**Example Output**:

- Normal: 0.95
- COVID-19: 0.05

## Results

The model performs exceptionally on the test set:

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| NORMAL       | 1.0       | 1.0    | 1.0      | 20      |
| PNEUMONIA    | 1.0       | 1.0    | 1.0      | 20      |
| **Accuracy** | -         | -      | 1.0      | 40      |

**Note**: Results are based on a small test set; real-world performance may vary with diverse data.

## Contributing

Contributions are welcome! Open issues or pull requests for improvements, such as adding transfer learning or multi-class classification.

## License

This project is licensed under the MIT License - see the [LICENSE](https://grok.com/c/LICENSE) file for details.

## Acknowledgements

- Inspired by Kaggle community projects on medical image classification.
- Thanks to PyTorch and Gradio for enabling accessible ML development.