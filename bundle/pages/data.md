# Data

## Source

Data sourced from CodaLab and Codabench competition archives, collected by Abderrahmane Moujar and Aleksandra Kruchinina.

## Format

Each sample is one ML competition described by its title and description, concatenated as a single UTF-8 string:

```
"<title> : <description>"
```

Ground-truth labels are assigned across two independent multi-label taxonomies:

- **ML_sector** — the ML methodology area(s) of the competition (11 possible labels, multi-label)
- **Field_sector** — the application domain(s) of the competition (11 possible labels, multi-label)

A competition may belong to multiple sectors in each column simultaneously.

## CSV Schema

The dataset is stored as two CSV files, aligned **row by row**: row *i* of the input file is described by row *i* of the reference file. There is no identifier column — row order is the only link between the two files.

Both files are **semicolon-separated** (`;`). Inside a label cell, several labels are separated by a comma (`,`).

`input_data.csv` — the texts:

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | `"<title> : <description>"` |

`reference_data.csv` — the ground-truth labels:

| Column | Type | Description |
|--------|------|-------------|
| `ML_sector` | str | Comma-separated label names from the ML_sector taxonomy (empty string if none) |
| `Field_sector` | str | Comma-separated label names from the Field_sector taxonomy (empty string if none) |

Example — first row of each file:

```
text
Clinical Named Entity Recognition : Identify diseases treatments and medications in clinical notes and discharge summaries using sequence labeling.
```

```
ML_sector;Field_sector
NLP / Text;Healthcare / Biology
```

## Sample Data

A 10-row sample illustrating the data schema is available in the **Starting Kit** download, as `sample_input_data.csv` and `sample_reference_data.csv`. Its sole purpose is to show participants the data format — no evaluation is performed on it.

## Split Policy

Inside the scoring program, the full dataset is split into 4 folds using:
```python
sklearn.model_selection.KFold(n_splits=4, shuffle=True, random_state=42)
```
Each fold has approximately 150 training samples and 50 test samples.

## License

CC-BY 4.0. To be confirmed when real competition archive data replaces the synthetic stand-in.
