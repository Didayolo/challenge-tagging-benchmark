# Data

## Source

Data sourced from CodaLab and Codabench competition archives, collected by Abderrahmane Moujar and Aleksandra Kruchinina.

## Format

Each sample is one ML competition described by its title and description, concatenated as a single UTF-8 string:

```
"<title> : <description>"
```

Ground-truth labels are evaluated across two independent multi-label taxonomies:

- **Machine Learning Domain** — the ML methodology area(s) of the competition (13 possible labels, multi-label)
- **Fields of Application** — the application domain(s) of the competition (18 possible labels, multi-label)

A competition may belong to multiple tags in each category simultaneously, or none.

## CSV Schema

Each set is stored as two CSV files, aligned **row by row**: row *i* of the data file is described by row *i* of the labels file. There is no identifier column — row order is the only link between the two files.

All files are **semicolon-separated** (`;`). Inside a label cell, several labels are separated by a comma (`,`).

| File | Content |
|------|---------|
| `train_data.csv` | training texts (given to `fit`) |
| `train_labels.csv` | training labels (given to `fit`) |
| `test_data.csv` | validation or test texts (given to `predict`) |
| `test_labels.csv` | validation or test labels (hidden, used for scoring) |

`*_data.csv` — the texts:

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | `"<title> : <description>"` |

`*_labels.csv` — the ground-truth labels:

| Column | Type | Description |
|--------|------|-------------|
| `Machine Learning Domain` | str | Comma-separated label names (empty string if none) |
| `Fields of Application` | str | Comma-separated label names (empty string if none) |
| `Conferences` | str | Optional column (ignored during scoring) |

Example — row from the dataset:

```
text
Shared Task 1: BanglaKnight : Classify Bangla prompts (Bangla script, romanized, code-mixed) as Safe, Offensive, Harmful or Severe.
```

```
Machine Learning Domain;Fields of Application;Conferences
NLP,LLM;Security,Linguistics;NAACL
```

## Sample Data

A 10-row sample illustrating the data schema is available in the **Starting Kit** download, as `sample_input_data.csv` (texts) and `sample_reference_data.csv` (labels). Its sole purpose is to show participants the data format — no evaluation is performed on it.

## Split Policy

The 5090 labeled samples are shuffled (`SEED=42`) and split into **3030 train**, **1030 validation** and **1030 test** samples. The split is checked so that every tag appears at least once in each of the three sets.

- **Feedback Phase**: train on the train set, evaluate on the validation set.
- **Final Phase**: train on the same train set, evaluate on the test set.

## License

CC-BY 4.0.
