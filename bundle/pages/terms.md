# Terms and Conditions

## Open Benchmark

This is an **open benchmark** with no prizes, no submission caps, and no deadline. All participants may submit freely and indefinitely.

## Participation Rules

1. **Eligible participants**: Any registered Codabench user may participate.
2. **Team size**: No limit.
3. **Submissions**: Unlimited daily submissions. Each submission is a `model.py` file containing a `Model` class conforming to the specified interface.
4. **Code ownership**: All submitted code remains the intellectual property of the participant. By submitting you confirm you have the right to use and submit the code.
5. **No cheating**: Multi-account usage to game the leaderboard is prohibited and will result in disqualification.
6. **No data leakage**: The full labeled dataset is never exposed to participant code. Attempting to extract it from the scoring environment is prohibited.

## Reproducibility

The scoring program enforces reproducibility via a fixed random seed (`random_state=42`) in the cross-validation split. Submitted models may set their own random seeds; variance in results across re-submissions will appear in the `f1_std` column.

## Data License

The dataset is made available under **CC-BY 4.0**. Attribution: Abderrahmane Moujar and Aleksandra Kruchinina (data collection); Adrien Pavão and Ayemane Bouarbi (competition design).

## Bundle License

MIT. The competition bundle itself (scoring program, ingestion program, baseline) is open-source.

## Disclaimer

This benchmark is provided as-is for research purposes. The organizers make no guarantees about platform uptime or continued operation.

---

By participating you agree to these terms.
