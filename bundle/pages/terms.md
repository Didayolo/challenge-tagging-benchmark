# Terms and Conditions

## Phases

This benchmark has no prizes. It runs in two phases: a **Feedback Phase** with unlimited submissions, evaluated on the validation set, followed by a **Final Phase** with a single submission per participant, evaluated on the unseen test set.

## Participation Rules

1. **Eligible participants**: Any registered Codabench user may participate.
2. **Team size**: No limit.
3. **Submissions**: Unlimited submissions in the Feedback Phase, one submission in the Final Phase. Each submission is a `model.py` file containing a `Model` class conforming to the specified interface.
4. **Code ownership**: All submitted code remains the intellectual property of the participant. By submitting you confirm you have the right to use and submit the code.
5. **No cheating**: Multi-account usage to game the leaderboard is prohibited and will result in disqualification.
6. **No data leakage**: The validation and test labels are never exposed to participant code. Attempting to extract it from the scoring environment is prohibited.

## Reproducibility

The train / validation / test split is fixed (shuffled once with `SEED=42`) and identical for every participant. Submitted models may set their own random seeds.

## Data License

The dataset is made available under **CC-BY 4.0**. Attribution: Abderrahmane Moujar and Aleksandra Kruchinina (data collection); Adrien Pavão and Ayemane Bouarbi (competition design).

## Bundle License

MIT. The competition bundle itself (scoring program, ingestion program, baseline) is open-source.

## Disclaimer

This benchmark is provided as-is for research purposes. The organizers make no guarantees about platform uptime or continued operation.

---

By participating you agree to these terms.
