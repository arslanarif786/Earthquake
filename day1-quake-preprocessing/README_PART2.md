# Day 2-3 Lab: Part 2 additions

Copy these files into your Day 1 project (same folder structure). Then update your environment once:

```bash
conda env update -f environment.yml --prune
```

| File | Task |
|---|---|
| `quakes/etl.py` | A: ETL with logging |
| `quakes/models.py` | B: estimator and model factories (also edit `transform.build_preprocessor`) |
| `quakes/evaluate.py` | C: metrics |
| `quakes/train.py` | D: training command line |
| `notebooks/day2-3_part1_sklearn_basics.ipynb` | Part 1 (do this first) |

```bash
pytest tests/test_etl.py        # Task A
pytest tests/test_models.py     # Task B
pytest tests/test_evaluate.py   # Task C
python -m quakes.etl --mode batch
python -m quakes.train --task regression --model linear
python -m quakes.train --task classification --model logistic
```

Create `ANSWERS_ML.md` with your answers to the six lab questions in the handout.

## Scheduling note (write in this file)
Add the exact cron line and the Windows Task Scheduler steps that would run
`python -m quakes.etl --mode batch` every hour.

## Hourly scheduling (Task A)

### Cron (Linux/macOS)

Run this from the project root after replacing the two example paths with the absolute project and Conda environment paths on the machine that will run the job:

```cron
0 * * * * cd /absolute/path/to/day1-quake-preprocessing && /absolute/path/to/conda/envs/niai-ml/bin/python -m quakes.etl --mode batch >> logs/cron-etl.log 2>&1
```

The job runs at minute 0 of every hour. The application log is written to `logs/etl.log`; cron stdout/stderr are appended to `logs/cron-etl.log`. Check the exit status in the scheduler/cron monitoring and inspect the end of both logs for a traceback or `ERROR` message. A successful application run ends with `ETL completed successfully` in `logs/etl.log`.

### Windows Task Scheduler

1. Open **Task Scheduler** and select **Create Basic Task** (or **Create Task** for more control).
2. Name it `NIAI Earthquake ETL` and choose **Daily** as the trigger.
3. In the trigger's advanced settings, repeat the task every **1 hour** for a duration of **Indefinitely**; enable the task.
4. Add an action **Start a program**. Set **Program/script** to the full path of `python.exe` inside the `niai-ml` Conda environment.
5. Set **Add arguments** to `-m quakes.etl --mode batch`.
6. Set **Start in** to the absolute path of the `day1-quake-preprocessing` project folder (without quotes unless needed by the dialog).
7. In **Settings**, allow the task to run on demand and configure restart-on-failure if desired. Save the task.
8. Right-click the task and select **Run** to test it. Check `<project>\logs\etl.log` for `ETL completed successfully`; inspect the last log lines and Task Scheduler's **Last Run Result** if it fails.

The Python logger writes to `logs/etl.log` (relative to the project root resolved by `quakes/config.py`) and also writes messages to the console. Ensure the scheduled account has write access to `logs/`, `data/processed/`, `models/`, and `reports/`.
