# autoresearch-futures

This repo is an autonomous research harness for 3-day futures move prediction.

The workflow is intentionally narrow:

- `prepare.py` and everything under `src/autoresearch_futures/` are the fixed harness.
- `train.py` is the file the agent is allowed to mutate during experiments.
- `results.tsv` is the experiment ledger.

## Setup

Before starting a run:

1. Create a fresh branch like `autoresearch/<tag>`.
2. Read:
   - `README.md`
   - `prepare.py`
   - `src/autoresearch_futures/data.py`
   - `src/autoresearch_futures/eval.py`
   - `train.py`
3. Make sure prepared data exists. If not, run `uv run prepare.py --input ...` or `uv run prepare.py --synthetic-demo`.
4. Run `uv run train.py` once to establish the baseline on this machine.
5. Log the baseline in `results.tsv` with `python scripts/log_result.py --status keep --description "baseline"`.

## Rules

What you CAN change:

- `train.py`

What you should treat as fixed unless the human explicitly asks otherwise:

- `prepare.py`
- `src/autoresearch_futures/data.py`
- `src/autoresearch_futures/eval.py`
- the raw/prepared dataset
- the target definition and split logic inside the fixed harness

Why the guardrail exists:

- if you let the agent rewrite the evaluator, it will optimize the harness instead of the signal.

## Goal

Maximize `val_sharpe`.

Secondary checks:

- `val_macro_f1`
- `val_bal_acc`
- `test_sharpe`

Higher `val_sharpe` is better. If two runs are effectively tied, prefer the simpler `train.py`.

## Output Format

At the end of each run, `train.py` prints lines like:

```text
val_sharpe:       0.423100
val_macro_f1:     0.401200
val_bal_acc:      0.487500
test_sharpe:      0.215700
val_loss:         1.102300
test_loss:        1.118900
training_seconds: 300.0
total_seconds:    302.7
num_steps:        5743
num_params_M:     0.55
context_length:   512
num_features:     11
```

To extract the metrics needed for ranking:

```bash
grep "^val_sharpe:\|^test_sharpe:\|^val_macro_f1:\|^val_bal_acc:\|^val_loss:\|^test_loss:\|^training_seconds:\|^total_seconds:\|^num_steps:\|^num_params_M:\|^context_length:\|^num_features:" run.log
```

## Logging

`results.tsv` is tab-separated with this schema:

```text
commit	val_sharpe	test_sharpe	val_macro_f1	val_bal_acc	val_loss	test_loss	training_seconds	total_seconds	num_steps	num_params_M	context_length	num_features	status	description
```

Status is one of:

- `keep`
- `discard`
- `crash`

Preferred logging command:

```bash
python scripts/log_result.py --status keep --description "baseline"
```

For crashes:

```bash
python scripts/log_result.py --status crash --description "OOM at larger context"
```

## Experiment Loop

Repeat indefinitely:

1. Inspect the current branch and kept commit.
2. Modify `train.py` with one concrete idea.
3. Commit only the intended files.
4. Run `uv run train.py > run.log 2>&1`.
5. Read out the full summary block from `run.log`.
6. If the run crashed, inspect `tail -n 80 run.log`, fix obvious bugs, or discard the idea.
7. Append the result to `results.tsv` with `python scripts/log_result.py --status ... --description ...`.
8. Keep the change only if `val_sharpe` improved meaningfully.
9. If not improved, revert to the last kept commit before trying the next idea.

Do not stop to ask whether to continue once the loop has started.

## Ranking Runs Later

After a long unattended session:

1. Sort `results.tsv` by `val_sharpe` descending.
2. Ignore `crash` rows.
3. Check that top rows also have reasonable `test_sharpe`.
4. Use `val_macro_f1`, `val_bal_acc`, and `val_loss` as sanity checks.
5. Prefer simpler changes in `train.py` when metrics are close.

`val_sharpe` is the ranking key. `test_sharpe` is a holdout check, not the optimization target.
