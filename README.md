# Toxic-Comments-Detection-Using-Bi-LSTM
Toxic Comment Classification This project is a machine learning model designed to detect toxic comments in text data. It uses a Bidirectional LSTM neural network to classify comments into multiple categories of toxicity, such as toxic, severe_toxic, obscene, threat, insult, and identity_hate. The model is trained on the Jigsaw Toxic Comment Classification Challenge dataset.

The project includes:

Data preprocessing using TensorFlow's TextVectorization.

A deep learning model built with Keras.

Evaluation metrics such as precision, recall, and accuracy.

A Gradio interface for real-time predictions.

Table of Contents Installation

Usage

Model Architecture

Evaluation

Deployment

Contributing

Installation To set up the project, follow these steps:

Clone the repository:

bash Copy git clone https://github.com/your-username/toxic-comment-classification.git cd toxic-comment-classification Install dependencies: Ensure you have Python 3.8+ installed. Then, install the required libraries:

bash Copy pip install -r requirements.txt Alternatively, you can install the dependencies manually:

bash Copy pip install tensorflow pandas numpy matplotlib gradio Download the dataset:

Download the dataset from the Jigsaw Toxic Comment Classification Challenge.

Place the train.csv file in the jigsaw-toxic-comment-classification-challenge/train.csv directory.

Usage Training the Model To train the model, run the Jupyter Notebook or Python script:

bash Copy jupyter notebook Toxicity.ipynb or

bash Copy python train.py Making Predictions You can use the trained model to classify comments as toxic or non-toxic. The model outputs probabilities for each toxicity category.

Example:

python Copy comment = "You freaking suck! I am going to hit you." prediction = model.predict(vectorizer([comment])) print(prediction) Gradio Interface To launch the Gradio interface for real-time predictions:

bash Copy python app.py This will start a local server where you can input comments and see the model's predictions.

Model Architecture The model consists of the following layers:

Embedding Layer: Converts text into dense vectors.

Bidirectional LSTM: Captures sequential dependencies in the text.

Dense Layers: Fully connected layers for feature extraction.

Output Layer: Sigmoid activation for multi-label classification.

Hyperparameters Max Features: 200,000

Sequence Length: 1800

Embedding Dimension: 32

LSTM Units: 32

Optimizer: Adam

Loss Function: Binary Cross-Entropy

Evaluation The model is evaluated using the following metrics:

Precision: Measures the accuracy of positive predictions.

Recall: Measures the fraction of true positives identified.

Accuracy: Measures the overall correctness of the model.

Example evaluation results:

Copy Precision: 0.92, Recall: 0.85, Accuracy: 0.89 Deployment The model is deployed using Gradio, a Python library for creating user-friendly interfaces. To deploy the model locally:

Run the Gradio app:

bash Copy python app.py Open the provided link in your browser.

Input a comment and see the model's predictions in real-time.
