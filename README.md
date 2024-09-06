
# Multi-Modal Data Classification using BERT

This project implements a data processing and classification pipeline using a BERT-based model to classify various types of input data (text, images, PDFs, URLs, and videos). It leverages natural language processing (NLP) techniques to unify and preprocess different data formats into a common textual format for classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview

The project processes multiple data types, including text, images, PDFs, URLs, and videos, using a BERT model for sequence classification. Each data type is preprocessed to extract text:
- **Text**: Processed directly or from files.
- **Images and Videos**: Text is extracted using Optical Character Recognition (OCR) with Tesseract.
- **PDFs**: Text is extracted using `pdfplumber`.
- **URLs**: Web scraping is used to extract textual content.

Once the textual data is extracted, the BERT tokenizer is used to tokenize the text, which is then passed to a BERT model for classification.

## Features
- Unified processing of multiple data types (text, images, PDFs, URLs, and videos).
- Text extraction from images, PDFs, and videos using OCR.
- Web scraping for URL content extraction.
- BERT-based model for sequence classification.
- Multi-class classification of input data sources.

## Installation
1. Clone the repository:
    ```
    git clone https://github.com/your-repo/multimodal-data-classification.git
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Ensure Tesseract is installed and added to your system PATH. You can find installation instructions [here](https://github.com/tesseract-ocr/tesseract).

## Usage
To preprocess the input files and run classification:
1. Update the `train_files` and `validation_files` dictionaries in the code with paths to your data files.
2. Run the script:
    ```
    python main.py
    ```

## Model Training
The project uses a BERT model fine-tuned for sequence classification. The training process involves:
1. Tokenizing the input texts.
2. Using a custom dataset and DataLoader for batching.
3. Optimizing the model using AdamW with a learning rate of 1e-5.
4. Training and validation loops over the specified number of epochs.

## Results
After training, the model will output the training loss, validation loss, and validation accuracy after each epoch.

## Dependencies
- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- pdfplumber
- pytesseract
- BeautifulSoup
- OpenCV
- tqdm

## License
This project is licensed under the MIT License.
