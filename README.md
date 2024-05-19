# Automated Fact-Checking System for Climate Science Claims

This repository contains the code and resources for an automated fact-checking system specifically designed to scrutinize claims pertaining to climate science. The system retrieves pertinent evidence passages from a designated knowledge source and classifies the veracity of the claims using various Natural Language Processing (NLP) techniques.

## Table of Contents
- [Objective](#objective)
- [Methodology](#methodology)
  - [Data](#data)
  - [System Design](#system-design)
  - [Implementation](#implementation)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Code](#running-the-code)
- [Experiments and Results](#experiments-and-results)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [References](#references)

## Objective
The objective of this project is to create an automated fact-checking system designed to extract evidence passages and classify the veracity of climate science claims. This helps mitigate misinformation and improve public understanding of climate science.

## Methodology

### Data
The dataset includes training, validation, and testing datasets, along with a JSON file containing numerous evidence paragraphs. Each claim in the training and validation datasets is labeled with SUPPORTS, REFUTES, NOT_ENOUGH_INFO, or DISPUTED. The test dataset lacks these labels.

### System Design
The system employs three primary models:
1. **Bi-Encoder**: Retrieves evidence paragraphs related to a claim.
2. **Cross-Encoder**: Further filters the retrieved evidence paragraphs to determine their relevance.
3. **Classifier**: Pairs and classifies the claim and the evidence paragraphs to predict the veracity label.

### Implementation
The implementation involves text preprocessing, model training, and evaluation. The models use pre-trained transformers from the `sentence-transformers` and `transformers` libraries. Detailed descriptions of the models and training procedures are provided in the project report.

## Usage

### Installation
To run the project, ensure you have Python installed and clone this repository. Then, install the necessary dependencies using:

```sh
pip install -r requirements.txt
```

### Running the Code
```sh
python3 main.py
```

## Experiments and Results
Extensive experiments were conducted to evaluate the performance of the models. The retrieve and rerank method significantly outperformed the simple vector similarity approach. The final model achieved an F1 score of 0.1744 for evidence retrieval and an accuracy of 0.3766 for claim classification, ranking 35th in the CodaLab competition.

## Conclusion and Future Work
This project demonstrated the potential of NLP techniques in building an efficient and accurate automated fact-checking system. Future work could explore different pre-trained models, address memory overflow issues, and refine preprocessing steps to further improve performance.

## References
- Zhenzhong Lan et al. (2020). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.
- Nils Reimers and Iryna Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Thomas Wolf et al. (2020). Transformers: State-of-the-Art Natural Language Processing.
