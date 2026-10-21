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

The dataset is stored as two CSV files, aligned **row by row**: row *i* of the input file is described by row *i* of the reference file. There is no identifier column — row order is the only link between the two files.

Both files are **semicolon-separated** (`;`). Inside a label cell, several labels are separated by a comma (`,`).

`input_data.csv` — the texts:

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | `"<title> : <description>"` |

`reference_data.csv` — the ground-truth labels:

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
NLP,LLM;Cybersecurity;NAACL
```

## Sample Data

A 10-row sample illustrating the data schema is available in the **Starting Kit** download, as `sample_input_data.csv` and `sample_reference_data.csv`. Its sole purpose is to show participants the data format — no evaluation is performed on it.

## Split Policy

Inside the scoring program, the dataset is split using iterative multi-label stratification (`TEST_SIZE=0.25`, `SEED=42`), allocating 75% for training and 25% for test evaluation.

## License

CC-BY 4.0.
