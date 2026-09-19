# Setup — one time, about five minutes

After this, the system runs daily on its own. Your Mac does not need to be on.

## 1. Create the repo

```bash
cd ~/  # or wherever you keep projects
unzip cbets-method-v2.zip && cd compounding

git init
git add -A
git commit -m "cbets: Method v2 implementation + daily agent"

gh repo create compounding-bets --public --source=. --push
```

No `gh`? Create an empty **public** repo called `compounding-bets` on github.com, then:

```bash
git remote add origin https://github.com/<your-username>/compounding-bets.git
git branch -M main && git push -u origin main
```

**Why public:** the daily task runs in Claude's cloud sandbox and reads the repo
over plain HTTPS. A private repo would need a GitHub token, and handing a token
to an automated session is not something worth doing for this. Nothing private
goes in the repo — it holds public football-data.co.uk mirrors and this code.
**Your pick log and results live in the Claude project, not on GitHub.**

## 2. Let the workflow write to the repo

GitHub → your repo → **Settings → Actions → General → Workflow permissions** →
select **Read and write permissions** → Save.

Without this the daily fetch runs but cannot commit, and the data never updates.

## 3. Prime the data

GitHub → **Actions** tab → **fetch football data** → **Run workflow**.

It takes a few minutes on the first run (it pulls seven seasons across seventeen
divisions). After that it runs itself at 05:10 UTC daily and only re-fetches the
current and previous season, so later runs are quick.

When it finishes you should see a `data/` folder with `fixtures.csv` and a folder
per season.

## 4. Tell Claude the repo

Reply in the conversation with your GitHub username, or paste the raw base URL:

```
https://raw.githubusercontent.com/<your-username>/compounding-bets/main
```

That goes into the project's `claude/daily-config.md`, and the scheduled task
starts working from its next run.

---

## What then happens every day, without you

| Time (CEST) | What runs | Where |
|---|---|---|
| 07:10 | GitHub Action fetches fixtures + results | GitHub's servers |
| 08:00 | Claude task: settle → refit → scan → log | Claude cloud |

The Claude run, in order:

1. **Settles** yesterday's pending picks — finds each match in the updated season
   file, records the result and the **closing** odds, computes raw and fair CLV.
2. **Refits** Dixon–Coles per division on everything dated before today.
3. **Scans** the next three days of fixtures. The probability is computed before
   the odds are read — structurally, not as a discipline rule: the fitting step
   never touches the odds columns.
4. **Applies the gates.** Edge must clear 4%; one selection per fixture; 1X2 and
   totals only.
5. **Logs** qualifying picks at stake 0 and updates the running CLV.
6. **Stays quiet** unless something happened.

You get a message only when: a selection clears the gates, the CLV verdict
changes, the kill criterion is met, or the run breaks.

Most days you will hear nothing. Per Method v2 §3 that is the correct output, and
it is the main thing v1 got wrong — a method that must produce five picks a week
will produce five picks a week.

## Changing it

- **Stricter or looser edge threshold** — `EDGE_THRESHOLD` in `daily.py`
- **Different leagues** — `CBETS_DIVISIONS` in `.github/workflows/fetch-data.yml`
- **Different time** — the cron in the workflow, and the scheduled task in Claude
- **Pause it** — disable the scheduled task in Claude; the data keeps updating
- **Stop it entirely** — disable the workflow in the Actions tab too

## Checking it yourself

```bash
python -m cbets.cli validate      # the ground-truth harness, ~30s, no network
python daily.py --repo . --log picks.csv --out report.json --today 2026-09-19
python -m pytest tests/ -q        # 34 tests
```
