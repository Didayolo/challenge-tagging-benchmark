![logo](bundle/logo.png)

### Codabench bundle: Challenge Tagger

Layout:

- `competition.yaml`           — written by write_competition_yaml
- `pages/`                     — markdown pages referenced from competition.yaml
- `ingestion_program/`         — ingestion.py + metadata.yaml: trains the model and predicts
- `scoring_program/`           — score.py + metadata.yaml: scores the predictions
- `feedback_phase/`, `final_phase/` — per phase: `input_data/` (train data + labels, valid or test data) and `reference_data/` (valid or test labels)
- `starting_kit/`              — files for participants to download
- `public_data/`               — (optional) publicly downloadable data

Data:

- `data/input_data.csv`, `data/reference_data.csv` — full labeled dataset (5090 samples)
- `data/split_data.py` — shuffles it into train (3030) / valid (1030) / test (1030), checks every label is present in each set, and writes the phase folders of the bundle
