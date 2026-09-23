#!/usr/bin/env python3
"""Ingestion program for codabench-tagger.

Participants submit a model.py containing a Model class.
This program copies model.py to the output directory so the
scoring program can import it and run 4-fold cross-validation.

It also forwards input_data.csv (the unlabeled texts) to the output
directory: the scoring program can only read the reference data and
the ingestion output, so the texts must travel through here to be
paired with the labels in reference_data.csv.

Args (positional, Codabench convention):
  1: /app/input_data/         -- task input data (input_data.csv)
  2: /app/output/             -- output directory (write model.py here)
  3: /app/program/            -- ingestion program dir
  4: /app/ingested_program/   -- participant's submitted files
"""
import sys
import os
import shutil

INPUT_DATA_FILE = 'input_data.csv'


def main():
    input_data_dir = sys.argv[1] if len(sys.argv) > 1 else '/app/input_data'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/app/output'
    program_dir = sys.argv[3] if len(sys.argv) > 3 else '/app/program'
    submission_dir = sys.argv[4] if len(sys.argv) > 4 else '/app/ingested_program'

    os.makedirs(output_dir, exist_ok=True)

    src = os.path.join(submission_dir, 'model.py')
    dst = os.path.join(output_dir, 'model.py')

    if not os.path.exists(src):
        raise FileNotFoundError(
            f"model.py not found in submission at: {src}\n"
            "Ensure your submission zip contains model.py at its root."
        )

    shutil.copy2(src, dst)
    print(f"Copied model.py from {src} to {dst}")

    data_src = os.path.join(input_data_dir, INPUT_DATA_FILE)
    data_dst = os.path.join(output_dir, INPUT_DATA_FILE)

    if not os.path.exists(data_src):
        raise FileNotFoundError(
            f"{INPUT_DATA_FILE} not found in input data at: {data_src}"
        )

    shutil.copy2(data_src, data_dst)
    print(f"Copied {INPUT_DATA_FILE} from {data_src} to {data_dst}")


if __name__ == '__main__':
    main()
