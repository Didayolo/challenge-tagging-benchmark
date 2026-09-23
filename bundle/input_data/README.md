# Input Data

This competition uses a **code submission** protocol with 4-fold cross-validation.

`input_data.csv` holds the competition texts (one `text` column, `;`-separated).
The ground-truth labels are held separately in the scoring program's reference
data (`reference_data.csv`), so participants never see them.

Participants submit `model.py` — the ingestion program forwards it, together
with `input_data.csv`, to the scoring program, which pairs the texts with the
reference labels by row order and trains and evaluates the model on the full
dataset.

A 10-row sample illustrating the data schema is available in the Starting Kit
download, as `sample_input_data.csv` and `sample_reference_data.csv`.
