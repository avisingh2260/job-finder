# Job Co-Pilot

A résumé-aware job finder and matcher powered by Google Gemini. Point it at your
résumé and it searches the major job boards, scrapes LinkedIn "hiring" posts for
hidden openings, and scores every result against your background — then hands you
a ranked report.

## What it does

1. **Reads your résumé** from a folder (PDF / TXT / MD).
2. **Parses it with Gemini** into a structured profile (skills, experience, suggested titles & keywords).
3. **Searches job portals** — LinkedIn, Indeed, Glassdoor, ZipRecruiter — via [`python-jobspy`](https://github.com/Bunsly/JobSpy), using search terms derived from your résumé.
4. **Scrapes LinkedIn hiring posts** discovered through DuckDuckGo, extracting company, role, location, and how to apply / who to contact.
5. **Scores each opening** against your résumé (match %, verdict, matching & missing skills, rationale) and ranks them.
6. **Saves a report** as `job_matches.json` and `job_matches.md`, and prints a ranked summary to your terminal.

## Setup

Requires **Python 3.9+**.

```bash
pip install -r requirements.txt
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>, then provide it
in **any one** of these ways:

**Option A — `.env` file (easiest).** Copy the template and paste your key in:

```bash
cp .env.example .env      # then open .env and set GEMINI_API_KEY=your-key-here
```

The `.env` file is git-ignored, so your key never gets committed.

**Option B — environment variable:**

```bash
# Windows (PowerShell) — reopen the terminal afterwards (setx only affects new terminals)
setx GEMINI_API_KEY "your-key-here"

# macOS / Linux
export GEMINI_API_KEY="your-key-here"
```

**Option C — inline flag:** `python job_copilot.py --api-key your-key-here`

## Usage

Drop your résumé into the `resumes/` folder and run:

```bash
python job_copilot.py
```

Or customize:

```bash
python job_copilot.py --resume ./MyResume.pdf --location "San Francisco"
python job_copilot.py --query "Frontend Engineer" --sites linkedin,indeed --results 20
python job_copilot.py --no-posts          # skip LinkedIn post scraping
python job_copilot.py --help              # all options
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--resume` | – | Path to a specific résumé file (PDF/TXT/MD). |
| `--resume-dir` | `resumes` | Folder to read the résumé from (most recent file wins). |
| `--query` | derived | Override the search term (default: derived from your résumé). |
| `--location` | – | Location filter, e.g. `Remote` or `San Francisco`. |
| `--sites` | `linkedin,indeed` | Comma-separated portals: `linkedin,indeed,glassdoor,zip_recruiter`. |
| `--results` | `15` | Results wanted per portal per query. |
| `--max-queries` | `2` | How many résumé-derived terms to search. |
| `--max-posts` | `8` | Max LinkedIn posts to scrape. |
| `--no-posts` | off | Skip LinkedIn post scraping entirely. |
| `--max-match` | `30` | Cap on how many openings to AI-score. |
| `--top` | `15` | How many matches to print to the console. |
| `--env-file` | `.env` | Path to a `.env` file holding `GEMINI_API_KEY`. |
| `--api-key` | env/.env | Gemini API key (overrides the env var and `.env` file). |
| `--output` | `job_matches` | Output filename stem (`.json` + `.md`). |

## Output

- **Console** — a ranked table of the best matches.
- **`job_matches.json`** — full structured results (every field + match details).
- **`job_matches.md`** — a readable report, one section per opening.

## Notes & limitations

- Your résumé and API key stay on your machine; the only data sent out is to the Gemini API for parsing, extraction, and scoring.
- **LinkedIn post scraping is best-effort.** LinkedIn has no public posts API and serves login walls / CAPTCHAs to unauthenticated requests, so posts are discovered via search and the script degrades gracefully (falls back to search snippets, skips blocked pages) rather than failing. The job-board search is the more reliable source.
- Scrapers can break or get rate-limited. The script keeps going when a source fails, so a partial run still produces useful results.
- Please respect each site's Terms of Service and use reasonable rate limits.
