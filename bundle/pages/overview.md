# Codabench Competition Tagger

## Goal

The goal of this benchmark is to evaluate the ability of machine learning models to automatically assign standardised taxonomy tags to machine learning competitions published on the Codabench platform. Tagging competitions with their ML methodology area (**ML_sector**) and application domain (**Field_sector**) enables discovery, filtering, and curation at scale. A sufficiently accurate tagger will be deployed on the Codabench platform to assist organizers and users alike.

## Task

Given the title and description of a competition (as a single text string of the form `"<title> : <description>"`), predict which ML sector(s) and Field sector(s) apply. Each competition may belong to multiple sectors simultaneously — this is a **multi-label** classification task.

**ML_sector** taxonomy (11 labels):
- NLP / Text
- Computer Vision
- Speech / Audio
- Time Series / Forecasting
- Graph / Networks
- Reinforcement Learning
- Generative / LLM
- Tabular / Classical ML
- ML / AutoML / HPO / NAS
- Algorithmics / non-ML
- Other

**Field_sector** taxonomy (11 labels):
- Healthcare / Biology
- Climate / Energy
- E-commerce / Retail / Finance
- Security & Privacy
- Robotics & Autonomous
- Agriculture & Food
- Transportation & Mobility
- Hard Sciences / Mathematics
- Human Sciences
- Media / Social
- Other

## Data

Data sourced from CodaLab and Codabench competition archives, collected by Abderrahmane Moujar and Aleksandra Kruchinina. A 10-row sample is provided in the starting kit to illustrate the data format. The full dataset is used exclusively in the evaluation, with 4-fold cross-validation.

## Evaluation

Submitted models are trained and evaluated on the full dataset via 4-fold cross-validation with a fixed random seed (`random_state=42`). The primary metric is the **Mean Macro F1** averaged across both output columns. The leaderboard also displays the standard deviation of the F1 score across folds (as a measure of robustness), as well as training time and inference time.

## Submission

Submit a Python file `model.py` implementing a `Model` class with `fit` and `predict` methods (see the starting kit for the interface specification). Zip `model.py` at the archive root and upload.

## Credits

Bundle by Adrien Pavão and Ayemane Bouarbi.  
Data collection: Abderrahmane Moujar and Aleksandra Kruchinina.  
Based on: [Pavão et al., *AI Competitions and Benchmarks*, 2024](https://ai-competitions-book.github.io/ai-competitions-book-full-project.pdf)
