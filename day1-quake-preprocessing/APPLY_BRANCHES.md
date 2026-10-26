# Apply Tasks A–D as separate Git branches

**Important:** The completed-code archive is a reference copy. Do not extract it over your entire existing project. Keep your existing repository and copy only the files listed for the current task. Start only after `git status` is clean, and keep a backup of uncommitted work.

From your existing project root, use the extracted completed-code folder as the source of the files.

## Task A — ETL

```bash
git switch earthquake-6
git switch -c task-a-etl
```

Copy only `quakes/etl.py` from the completed-code folder into your project, then run:

```bash
pytest -q tests/test_etl.py
pytest -v tests/test_etl.py
git add quakes/etl.py
git commit -m "Task A: implement logged idempotent ETL"
```

## Task B — models and polynomial preprocessing

```bash
git switch -c task-b-models
```

Copy only `quakes/models.py` and `quakes/transform.py`, then run:

```bash
pytest -q tests/test_models.py
pytest -v tests/test_models.py
pytest -q tests/test_transform.py
git add quakes/models.py quakes/transform.py
git commit -m "Task B: implement model factories and polynomial preprocessing"
```

## Task C — evaluation

```bash
git switch -c task-c-evaluate
```

Copy only `quakes/evaluate.py`, then run:

```bash
pytest -q tests/test_evaluate.py
pytest -v tests/test_evaluate.py
git add quakes/evaluate.py
git commit -m "Task C: implement evaluation metrics and threshold search"
```

## Task D — training CLI and written deliverables

```bash
git switch -c task-d-train
```

Copy `quakes/train.py`, `README_PART2.md`, and `ANSWERS_ML.md`, then run:

```bash
pytest -q
pytest -v
python -m quakes.etl --mode batch
python -m quakes.train --task regression --model linear
python -m quakes.train --task classification --model logistic
git add quakes/train.py README_PART2.md ANSWERS_ML.md
git commit -m "Task D: implement training CLI and lab documentation"
```

The branches are stacked: Task B starts from Task A, Task C from Task B, and Task D from Task C. Once all checks pass, merge in order:

```bash
git switch earthquake-6
git merge task-a-etl
git merge task-b-models
git merge task-c-evaluate
git merge task-d-train
```

If any test fails, stop at that task and share the full failure output before moving on. The ETL and training commands need network access or a valid instructor-provided fallback file and a populated database.
