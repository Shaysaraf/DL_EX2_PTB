#PTB Language Modeling with LSTM and Transformer
This project implements and compares LSTM and Transformer models on the Penn Treebank (PTB) dataset for language modeling. It evaluates the models based on perplexity over training and test sets.

Models Included
LSTM (no dropout) – Based on Zaremba et al., with 2 layers and no regularization.

LSTM (dropout 0.3) – Same architecture, with 0.3 dropout for regularization.

Transformer – A lightweight Transformer model using positional encoding.

Dataset
The models are trained and evaluated on the word-level PTB dataset:

ptb.train.txt

ptb.valid.txt

ptb.test.txt

Each word is tokenized and encoded based on a vocabulary built from the training set.

Requirements
Python 3.7+

PyTorch

tqdm

matplotlib

Install dependencies:

bash
Copy
Edit
pip install torch tqdm matplotlib
Running the Code
Simply run the main Python file:

bash
Copy
Edit
python ptb_language_modeling.py
The script will:

Preprocess data and build the vocabulary

Train LSTM (no dropout), LSTM (dropout 0.3), and Transformer models

Log perplexity during training

Save and show a plot comparing training and test perplexity across epochs (all_models_perplexity.png)

Results
The Transformer model achieves better test perplexity than both LSTM models.

Dropout helps regularize the LSTM but doesn't outperform the Transformer.

A plot of perplexity vs. epochs is saved and displayed.

