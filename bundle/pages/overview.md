# Codabench Competition Tagger

## Goal

The goal of this benchmark is to evaluate the ability of machine learning models to automatically assign standardised taxonomy tags to machine learning competitions published on the Codabench platform. Tagging competitions with their ML methodology area (**Machine Learning Domain**) and application domain (**Fields of Application**) enables discovery, filtering, and curation at scale. A sufficiently accurate tagger will be deployed on the Codabench platform to assist organizers and users alike.

## Task

Given the title and description of a competition (as a single text string of the form `"<title> : <description>"`), predict which tags apply across the 2 taxonomy categories. Each competition may belong to multiple tags simultaneously — this is a **multi-label** classification task.

**Machine Learning Domain** taxonomy (13 tags):
- Computer Vision
- NLP
- Tabular
- Time Series
- LLM
- Generative AI
- Reinforcement Learning
- Audio
- Graphs
- Multimodal
- AutoML
- Algorithms
- Other

**Fields of Application** taxonomy (18 tags):
- Healthcare
- Biology
- Finance
- E-commerce
- Earth Observation
- Climate
- Energy
- Robotics
- Cybersecurity
- Transportation
- Agriculture
- Sports
- Education
- Social Media
- Astronomy
- Legal
- Science
- Other

## Data

Data sourced from CodaLab and Codabench competition archives, collected by Abderrahmane Moujar and Aleksandra Kruchinina. A 10-row sample is provided in the starting kit to illustrate the data format. The full dataset is evaluated using a stratified train/test split.

## Evaluation

Submitted models are evaluated via a multi-label stratified train/test split (75% train / 25% test, `random_state=42`). The primary metric is the **Mean Macro F1** averaged across categories over the tags present in the test set. The leaderboard also displays the standard deviation of per-tag F1 scores, as well as training time and inference time.

## Submission

Submit a Python file `model.py` implementing a `Model` class with `fit(X_train, y_train)` and `predict(X_test)` methods returning `{category: array}` (see the starting kit for the interface specification). Zip `model.py` at the archive root and upload.

## Credits

Bundle by Adrien Pavão and Ayemane Bouarbi.  
Data collection: Abderrahmane Moujar and Aleksandra Kruchinina.  
Based on: [Pavão et al., *AI Competitions and Benchmarks*, 2024](https://ai-competitions-book.github.io/ai-competitions-book-full-project.pdf)
