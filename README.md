<div align="center">

# NewsLies
**An Arabic Fake News Detection using LSTM and AraBERT**

</div>

This repository contains a project aimed at detecting fake news in Arabic using advanced Natural Language Processing (NLP) techniques. The project leverages the Arabic Fake News Dataset (AFND) and builds a deep learning model using Long Short-Term Memory (LSTM) networks and AraBERT for text classification.  
*This model is still under development*

## Table of Contents

- [Dataset](#dataset)
- [Dataset Structure](#dataset-structure)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Model Details](#model-details)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contributions](#contributions)

## Dataset

The dataset used in this project is the [Arabic Fake News Dataset (AFND)](https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd/data) from Kaggle. This dataset is a collection of over 600,000 public Arabic news articles collected from 134 different Arabic news websites. The articles are classified into three categories: credible, not credible, and undecided.

### Dataset Structure

- **sources.json**: Contains 134 lines corresponding to 134 public Arabic news websites. The URLs of the websites are anonymized as "source_1", "source_2", etc.
- **Dataset Directory**: Contains 134 sub-directories named after the anonymous sources. Each sub-directory has a `scraped_articles.json` file, which stores the title, text, and publication date of the articles from that source.

### Creators of the Dataset:
- Ashwaq Khalil
- Moath Jarrah
- Monther Aldwairi

## Project Structure

The entire project, including data preprocessing, model building, training, and evaluation, is contained within a single Jupyter notebook:

- `NewsLies.ipynb`: This notebook includes:
  - **Data Preprocessing**: Advanced text preprocessing including text normalization, stopword removal, and stemming using Farasa and ISRIStemmer.
  - **Model Definition**: LSTM-based model architecture with AraBERT embeddings and an attention mechanism for improved classification.
  - **Model Training**: Training the model on the Arabic Fake News Dataset, along with evaluation of its performance.
  - **Inference**: Running the trained model on new Arabic news articles to classify them.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Assem-ElQersh/NewsLies.git
   cd NewsLies
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place it in the data/ directory.
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook NewsLies.ipynb
   ```
5. Run the cells in the notebook sequentially to preprocess the data, train the model, and perform inference.

## Model Details

-  **Preprocessing**: The text is normalized, diacritics and special characters are removed, and stemming is performed using Farasa tools.
-  **Embedding Layer**: AraBERT is used to generate dynamic embeddings for Arabic text.
-  **LSTM Layers**: The model contains multiple LSTM layers to capture the temporal dependencies in the text.
-  **Attention Mechanism**: An attention layer is added to focus on the most important parts of the text.
-  **Output Layer**: The output is a softmax layer for multi-class classification.

## Evaluation
  The model is evaluated on the test set using accuracy, precision, recall, and F1-score. A confusion matrix is also provided for a detailed view of the model's performance.

## Results
-  **Accuracy**: The model achieves high accuracy in detecting fake news articles, with detailed metrics provided in the results section.
-  **Confusion Matrix**: Visualizes the performance across all three classes (credible, not credible, undecided).

## License
The dataset used in this project does not specify a license. Please respect the usage policies set by the dataset creators.
The code used in this project

## Acknowledgements
Special thanks to the dataset Owners Ashwaq Khalil, Moath Jarrah, and Monther Aldwairi for making the Arabic Fake News Dataset available for research and development.

## Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request.
