#!/usr/bin/env python3
"""
job_copilot.py - Resume-aware job finder & matcher (Google Gemini powered).

WHAT IT DOES
  1. Reads your resume from a folder (PDF / TXT / MD).
  2. Uses Google Gemini to parse it into a structured profile.
  3. Searches LinkedIn, Indeed, Glassdoor & ZipRecruiter (via python-jobspy).
  4. Scrapes LinkedIn "hiring" posts (discovered via DuckDuckGo) for hidden
     openings, and extracts company / contact / role details from each.
  5. Scores every opening against your resume with Gemini and ranks them.
  6. Saves JSON, Markdown, and a graphical HTML report, and prints a ranked summary.

SETUP (one time)
  pip install google-generativeai python-jobspy pypdf ddgs requests beautifulsoup4

  Get a free Gemini API key at https://aistudio.google.com/apikey , then provide
  it in ANY one of these ways:
    * Easiest - create a file named ".env" next to this script, containing:
          GEMINI_API_KEY=your-key-here
    * Environment variable:
          Windows (PowerShell):  setx GEMINI_API_KEY "your-key-here"  (reopen terminal)
          macOS / Linux:         export GEMINI_API_KEY="your-key-here"
    * Inline flag:               python job_copilot.py --api-key your-key-here

USAGE
  # Drop your resume into a folder named "resumes", then:
  python job_copilot.py

  # Or point at a specific file / folder and add options:
  python job_copilot.py --resume ./MyResume.pdf --location "San Francisco"
  python job_copilot.py --query "Frontend Engineer" --sites linkedin,indeed --results 20
  python job_copilot.py --no-posts        # skip LinkedIn post scraping
  python job_copilot.py --hours-old 72 --max-applicants 10   # recent, low-competition
  python job_copilot.py --help

NOTES
  * Your resume and API key stay on your machine; only Gemini calls leave it.
  * --hours-old keeps only recent postings (jobs and LinkedIn posts; default 1 week).
    --max-applicants drops LinkedIn jobs
    with a confirmed applicant count at/above the threshold (best-effort: read
    from LinkedIn's page when shown; jobs with an unknown count are kept).
  * Respect each site's Terms of Service and rate limits. Scrapers can break or
    hit CAPTCHAs - the script degrades gracefully and keeps going when a source
    fails, so a partial run still produces useful results.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

SUPPORTED_RESUME_EXTS = (".pdf", ".txt", ".md")
VALID_SITES = ("linkedin", "indeed", "glassdoor", "zip_recruiter")


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def info(msg: str) -> None:
    print(f"  {msg}", flush=True)


def step(msg: str) -> None:
    print(f"\n> {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def _require(module_name: str, pip_name: str):
    """Import a soft dependency, with a friendly message if it's missing."""
    try:
        return __import__(module_name)
    except ImportError:
        die(
            f"Missing dependency '{pip_name}'.\n"
            f"Install everything with:\n"
            f"  pip install google-generativeai python-jobspy pypdf ddgs requests beautifulsoup4"
        )


def _strip_json_fences(text: str) -> str:
    """Gemini occasionally wraps JSON in ```json fences - remove them."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def load_env_file(path: str) -> dict:
    """Tiny .env parser (no dependency). Reads KEY=value lines.

    Supports surrounding quotes, '#' comments, blank lines, and an optional
    leading 'export'. Returns a dict; missing file yields an empty dict.
    """
    values: dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return values
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    values[key] = val
    except OSError:
        pass
    return values


# --------------------------------------------------------------------------- #
#  Gemini
# --------------------------------------------------------------------------- #
def configure_gemini(api_key: str) -> str:
    """Configure the SDK and pick the best available flash model."""
    genai = _require("google.generativeai", "google-generativeai")
    import google.generativeai as genai  # noqa: F811  (real, configured handle)

    genai.configure(api_key=api_key)

    try:
        flash, other = [], []
        for m in genai.list_models():
            if "generateContent" not in getattr(m, "supported_generation_methods", []):
                continue
            model_id = m.name.replace("models/", "")
            if "tuned" in model_id.lower():
                continue
            (flash if "flash" in model_id.lower() else other).append(model_id)
        flash.sort(reverse=True)
        other.sort(reverse=True)
        chosen = (flash or other or ["gemini-1.5-flash"])[0]
    except Exception as exc:  # noqa: BLE001
        info(f"Could not list models ({exc}); falling back to gemini-1.5-flash.")
        chosen = "gemini-1.5-flash"

    info(f"Using Gemini model: {chosen}")
    return chosen


def gemini_json(model_name: str, prompt: str, retries: int = 3) -> dict | None:
    """Run a prompt expecting a JSON object back. Retries with backoff."""
    import google.generativeai as genai

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={"response_mime_type": "application/json"},
    )
    for attempt in range(1, retries + 1):
        try:
            resp = model.generate_content(prompt)
            data = json.loads(_strip_json_fences(resp.text))
            # Gemini sometimes returns a JSON array; unwrap to the first object
            # so callers can always rely on getting a dict (or None).
            if isinstance(data, list):
                data = next((item for item in data if isinstance(item, dict)), None)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            if attempt == retries:
                return None
        except Exception as exc:  # noqa: BLE001  (rate limits, transient API errors)
            if attempt == retries:
                info(f"Gemini call failed after {retries} tries: {exc}")
                return None
            time.sleep(2 ** attempt)
    return None


# --------------------------------------------------------------------------- #
#  Resume reading & parsing
# --------------------------------------------------------------------------- #
def pick_resume_file(resume: str | None, resume_dir: str) -> str:
    if resume:
        if not os.path.isfile(resume):
            die(f"Resume file not found: {resume}")
        return resume

    if not os.path.isdir(resume_dir):
        die(
            f"Resume folder '{resume_dir}' does not exist.\n"
            f"Create it and drop your resume (PDF/TXT/MD) inside, or pass --resume <file>."
        )

    candidates = [
        p
        for p in glob.glob(os.path.join(resume_dir, "*"))
        if p.lower().endswith(SUPPORTED_RESUME_EXTS)
        and not os.path.basename(p).lower().startswith("readme")
    ]
    if not candidates:
        die(
            f"No resume found in '{resume_dir}'. Add a .pdf, .txt, or .md file there, "
            f"or pass --resume <file>."
        )
    # Most recently modified wins.
    chosen = max(candidates, key=os.path.getmtime)
    if len(candidates) > 1:
        info(f"Found {len(candidates)} files; using the most recent: {os.path.basename(chosen)}")
    return chosen


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        _require("pypdf", "pypdf")
        import logging as _logging
        import warnings as _warnings

        # pypdf logs noisy recovery warnings (e.g. "incorrect startxref pointer")
        # on many real-world PDFs even though extraction succeeds — quiet them.
        _logging.getLogger("pypdf").setLevel(_logging.ERROR)
        from pypdf import PdfReader

        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                reader = PdfReader(path)
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            die(
                f"Could not read PDF '{path}': {exc}\n"
                "If it's a scanned/image-only PDF, export a text-based PDF or save it as .txt/.md."
            )
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()

    text = text.strip()
    if not text:
        die(
            f"No text could be extracted from '{path}'. "
            "If it's a scanned/image-only PDF, export a text-based version or save it as .txt/.md."
        )
    return text


def parse_resume(model_name: str, raw_text: str) -> dict:
    prompt = f"""
Analyze the following resume text and extract a structured JSON profile.

Resume:
---
{raw_text}
---

Return ONLY a valid JSON object with this structure:
{{
  "name": "Full name or empty string",
  "contact": {{"email": "", "phone": "", "location": ""}},
  "summary": "Concise professional summary",
  "skills": {{
    "languages": ["programming languages"],
    "frameworks_and_tools": ["frameworks, databases, tools"],
    "soft_skills": ["e.g. Communication, Leadership"]
  }},
  "experience": [
    {{"role": "", "company": "", "duration": "", "highlights": ["bullet"]}}
  ],
  "suggested_titles": ["3-4 job titles to search for based on this resume"],
  "suggested_keywords": ["4-6 optimized job-search keywords"]
}}
"""
    profile = gemini_json(model_name, prompt)
    if not profile:
        die("Gemini could not parse the resume. Check your API key / quota and try again.")
    profile["raw_text"] = raw_text
    return profile


# --------------------------------------------------------------------------- #
#  Portal scraping (python-jobspy)
# --------------------------------------------------------------------------- #
def search_portals(
    terms: list[str], location: str, sites: list[str], results: int, hours_old: int | None = None
) -> list[dict]:
    _require("jobspy", "python-jobspy")
    from jobspy import scrape_jobs
    import pandas as pd

    sites = [s for s in sites if s in VALID_SITES] or ["linkedin"]
    needs_country = any(s in ("indeed", "glassdoor") for s in sites)

    collected: list[dict] = []
    for term in terms:
        info(f"Searching {', '.join(sites)} for: {term!r}")
        try:
            df = scrape_jobs(
                site_name=sites,
                search_term=term,
                location=location or None,
                results_wanted=results,
                country_indeed="USA" if needs_country else None,
                # Only return jobs posted within this many hours (recency filter).
                hours_old=hours_old if hours_old and hours_old > 0 else None,
                # LinkedIn returns no description unless we ask for it; without this,
                # every LinkedIn match scores 0% ("No description"). Slower but needed.
                linkedin_fetch_description=True,
            )
        except Exception as exc:  # noqa: BLE001
            info(f"  Scrape failed for {term!r}: {exc}")
            continue

        if df is None or df.empty:
            continue

        df = df.where(pd.notnull(df), None)
        for _, row in df.iterrows():
            collected.append(
                {
                    "source": str(row.get("site") or "portal"),
                    "title": str(row.get("title") or "Unknown Title"),
                    "company": str(row.get("company") or "Unknown Company"),
                    "location": str(row.get("location") or ""),
                    "date_posted": str(row.get("date_posted") or ""),
                    "job_url": str(row.get("job_url") or ""),
                    "description": str(row.get("description") or ""),
                    "contact": "",
                    "applicants": None,
                    "applicants_label": "",
                }
            )
    return collected


# --------------------------------------------------------------------------- #
#  Applicant-count filter (LinkedIn, best-effort)
# --------------------------------------------------------------------------- #
def linkedin_applicant_count(job_url: str) -> tuple[int | None, str]:
    """Best-effort scrape of LinkedIn's applicant count for a job.

    JobSpy does not expose this, so we hit the public guest job endpoint and
    parse the applicant caption. Returns (count, label):
      * exact number shown  -> (N, "N applicants")
      * "first N" early note -> (None, "Among the first N applicants")
      * unknown / blocked    -> (None, "")
    """
    import requests

    m = re.search(r"(\d{6,})", job_url or "")
    if not m:
        return None, ""
    url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{m.group(1)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return None, ""
        page = resp.text
    except Exception:  # noqa: BLE001
        return None, ""

    early = re.search(r"first\s+(\d+)\s+applicants?", page, re.I)
    if early:
        return None, f"Among the first {early.group(1)} applicants"
    exact = re.search(r"([\d,]+)\s+applicants?", page, re.I)
    if exact:
        n = int(exact.group(1).replace(",", ""))
        return n, f"{n} applicants"
    return None, ""


def filter_by_applicants(jobs: list[dict], max_applicants: int) -> tuple[list[dict], int]:
    """Annotate LinkedIn jobs with applicant counts and drop those with a
    *confirmed* count >= max_applicants. Unknown or 'first N' (early-stage)
    jobs and non-LinkedIn sources are kept, since we can't confirm a count."""
    if not max_applicants or max_applicants <= 0:
        return jobs, 0

    li = [j for j in jobs if j.get("source") == "linkedin" and j.get("job_url")]
    if li:
        info(f"Checking applicant counts on {len(li)} LinkedIn job(s) (best-effort)...")

    kept, dropped = [], 0
    for job in jobs:
        if job.get("source") != "linkedin" or not job.get("job_url"):
            kept.append(job)
            continue
        count, label = linkedin_applicant_count(job["job_url"])
        job["applicants"] = count
        job["applicants_label"] = label
        if isinstance(count, int) and count >= max_applicants:
            dropped += 1
        else:
            kept.append(job)
        time.sleep(0.3)
    return kept, dropped


# --------------------------------------------------------------------------- #
#  LinkedIn post scraping (DuckDuckGo discovery + page fetch + Gemini extract)
# --------------------------------------------------------------------------- #
def _ddgs_timelimit(hours_old: int | None) -> str | None:
    """Map an hours-old window to DuckDuckGo's coarse time filter (d/w/m/y)."""
    if not hours_old or hours_old <= 0:
        return None
    if hours_old <= 24:
        return "d"
    if hours_old <= 168:
        return "w"
    if hours_old <= 744:
        return "m"
    return "y"


def _ddgs_search(query: str, max_results: int, timelimit: str | None = None) -> list[dict]:
    """Return DuckDuckGo text results, tolerating either ddgs package name."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            die("Missing dependency 'ddgs'. Install with: pip install ddgs")
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results, timelimit=timelimit))
    except Exception as exc:  # noqa: BLE001
        info(f"  DuckDuckGo search failed: {exc}")
        return []


def fetch_page_text(url: str) -> str:
    requests = _require("requests", "requests")
    import requests  # noqa: F811
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
        return text[:6000]
    except Exception:  # noqa: BLE001
        return ""


def find_linkedin_posts(
    terms: list[str], location: str, max_posts: int, hours_old: int | None = None
) -> list[dict]:
    seen_urls: set[str] = set()
    posts: list[dict] = []
    timelimit = _ddgs_timelimit(hours_old)

    loc = f' "{location}"' if location else ""
    queries = []
    for term in terms[:2]:
        queries.append(f'site:linkedin.com/posts/ "hiring" "{term}"{loc}')
        queries.append(
            f'site:linkedin.com/posts/ ("hiring" OR "we are hiring") "{term}"{loc} ("email" OR "DM")'
        )

    for query in queries:
        if len(posts) >= max_posts:
            break
        for hit in _ddgs_search(query, max_results=max_posts, timelimit=timelimit):
            url = hit.get("href") or hit.get("url") or ""
            if "linkedin.com/posts/" not in url or url in seen_urls:
                continue
            seen_urls.add(url)
            posts.append(
                {
                    "url": url,
                    "snippet": f"{hit.get('title', '')}. {hit.get('body', '')}".strip(),
                }
            )
            if len(posts) >= max_posts:
                break
        time.sleep(1)  # be polite to the search endpoint

    return posts


def extract_post_job(model_name: str, post: dict) -> dict | None:
    page_text = fetch_page_text(post["url"])
    combined = f"Search snippet: {post['snippet']}\n\nPage text: {page_text}".strip()

    prompt = f"""
The text below comes from a LinkedIn post that may describe a job opening.

Text:
---
{combined[:6000]}
---

Decide whether this is a genuine hiring/job post. Return ONLY a JSON object:
{{
  "is_job_post": true,
  "title": "Role title being hired for, or best guess",
  "company": "Company name if mentioned, else empty",
  "location": "Location if mentioned, else empty",
  "contact": "Any email, 'DM the poster', or application link mentioned, else empty",
  "description": "1-3 sentence summary of the opening and how to apply"
}}
If it is NOT a job/hiring post, return {{"is_job_post": false}}.
"""
    data = gemini_json(model_name, prompt)
    if not data or not data.get("is_job_post"):
        return None
    return {
        "source": "linkedin_post",
        "title": data.get("title") or "LinkedIn hiring post",
        "company": data.get("company") or "",
        "location": data.get("location") or "",
        "date_posted": "",
        "job_url": post["url"],
        "description": data.get("description") or post["snippet"],
        "contact": data.get("contact") or "",
    }


# --------------------------------------------------------------------------- #
#  Matching
# --------------------------------------------------------------------------- #
def match_job(model_name: str, resume_text: str, job: dict) -> dict:
    desc = (job.get("description") or "").strip()
    limited = False
    if not desc:
        # No full description (can happen on LinkedIn/Glassdoor). Fall back to the
        # title/company so we still estimate a fit instead of returning a flat 0%.
        desc = " — ".join(
            filter(None, [job.get("title"), job.get("company"), job.get("location")])
        ).strip()
        limited = True
        if not desc:
            return {
                "match_score": 0,
                "match_verdict": "No data",
                "matching_skills": [],
                "missing_skills": [],
                "match_rationale": "No description or title was available to evaluate.",
            }
    if limited:
        desc = "(Only the job title/company was available; estimate the fit.)\n" + desc

    prompt = f"""
You are an expert technical recruiter. Compare the candidate's resume with the job
and grade the fit objectively.

Candidate resume:
---
{resume_text[:8000]}
---

Job title: {job.get('title', '')}
Company: {job.get('company', '')}
Job description:
---
{desc[:6000]}
---

Weigh the candidate's years of experience and seniority heavily: reward roles that
match their level and domain, and lower the score for clear seniority mismatches
(over- or under-qualified). Favor roles aligned with their actual experience.
Then score the fit as an integer from 0 to 100 based on genuine overlap (skills,
seniority, domain). Even when the job detail is limited, give your best estimate
from the title and company — do NOT default to 0.

Return ONLY a JSON object. Fill every field with your real assessment; the values
below are format examples, not answers to copy:
{{
  "match_score": 78,
  "match_verdict": "one of: Excellent Fit (85-100), Strong Competitor (70-84), Skill Gap (50-69), Not Aligned (0-49)",
  "matching_skills": ["skills the candidate has that the job wants"],
  "missing_skills": ["key skills the job wants that the resume lacks"],
  "match_rationale": "2-3 sentences explaining the score."
}}
"""
    result = gemini_json(model_name, prompt)
    if not result:
        return {
            "match_score": None,
            "match_verdict": "Could not evaluate",
            "matching_skills": [],
            "missing_skills": [],
            "match_rationale": "Gemini did not return a usable evaluation for this opening.",
        }
    try:
        result["match_score"] = int(result.get("match_score") or 0)
    except (TypeError, ValueError):
        result["match_score"] = 0
    return result


def dedupe_jobs(jobs: list[dict]) -> list[dict]:
    seen: set = set()
    unique: list[dict] = []
    for job in jobs:
        key = job.get("job_url") or (job.get("title", "").lower(), job.get("company", "").lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    return unique


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def write_reports(payload: dict, json_path: str, md_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    lines = [
        f"# Job matches for {payload['candidate'].get('name') or 'candidate'}",
        "",
        f"_Generated {payload['generated_at']} - {len(payload['results'])} openings ranked by fit._",
        "",
        f"**Search terms:** {', '.join(payload['search_terms'])}",
        "",
    ]
    for i, job in enumerate(payload["results"], 1):
        m = job.get("match", {})
        score = m.get("match_score")
        score_str = f"{score}%" if isinstance(score, int) else "n/a"
        lines += [
            f"## {i}. {job['title']} — {job.get('company') or 'Unknown'}  ·  {score_str}",
            "",
            f"- **Source:** {job['source']}",
            f"- **Location:** {job.get('location') or 'n/a'}",
            f"- **Verdict:** {m.get('match_verdict', 'n/a')}",
        ]
        if job.get("applicants_label"):
            lines.append(f"- **Applicants:** {job['applicants_label']}")
        if job.get("contact"):
            lines.append(f"- **Contact:** {job['contact']}")
        if job.get("job_url"):
            lines.append(f"- **Link:** {job['job_url']}")
        if m.get("matching_skills"):
            lines.append(f"- **Matching skills:** {', '.join(m['matching_skills'])}")
        if m.get("missing_skills"):
            lines.append(f"- **Missing skills:** {', '.join(m['missing_skills'])}")
        if m.get("match_rationale"):
            lines += ["", f"> {m['match_rationale']}"]
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Job Matches - __NAME__</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--faint:#6e7681;--blue:#58a6ff}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);padding:40px 20px;line-height:1.5;background-image:radial-gradient(1000px 500px at 85% -10%,rgba(88,166,255,.08),transparent)}
.wrap{max-width:880px;margin:0 auto}
h1{font-size:1.6rem;font-weight:600;margin-bottom:4px}
.gen{color:var(--muted);font-size:.85rem;margin-bottom:20px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.stat{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 16px;min-width:120px}
.stat .n{font-size:1.4rem;font-weight:700}
.stat .l{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);margin-top:2px}
.terms{margin-bottom:26px}
.term{display:inline-block;background:rgba(88,166,255,.12);color:var(--blue);border:1px solid rgba(88,166,255,.3);border-radius:999px;padding:3px 11px;font-size:.78rem;margin:0 6px 6px 0}
.card{display:flex;gap:18px;background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:20px;margin-bottom:14px}
.ring{flex-shrink:0;width:72px;height:72px;border-radius:50%;display:flex;align-items:center;justify-content:center}
.hole{width:56px;height:56px;border-radius:50%;background:var(--panel);display:flex;flex-direction:column;align-items:center;justify-content:center}
.hole span{font-size:1.05rem;font-weight:700;line-height:1}
.hole small{font-size:.55rem;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);margin-top:2px}
.meta{min-width:0;flex:1}
.toprow{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.rank{font-size:.72rem;color:var(--faint);font-weight:600}
.src{font-size:.66rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);border:1px solid var(--border);border-radius:5px;padding:1px 7px}
.verdict{font-size:.72rem;color:var(--text);background:#21262d;border-radius:5px;padding:1px 8px;margin-left:auto}
h3{font-size:1.05rem;font-weight:600;overflow-wrap:anywhere}
.sub{color:var(--muted);font-size:.85rem;margin-top:2px}
.appl{color:#3fb950;font-size:.78rem;font-weight:600;margin-top:6px}
.rationale{font-size:.86rem;color:var(--text);margin-top:10px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip{font-size:.72rem;border-radius:6px;padding:2px 9px;border:1px solid var(--border)}
.chip.ok{color:#3fb950;background:rgba(63,185,80,.1);border-color:rgba(63,185,80,.3)}
.chip.miss{color:var(--faint);text-decoration:line-through}
.contact{font-size:.8rem;color:var(--blue);margin-top:10px;overflow-wrap:anywhere}
.links{margin-top:12px}
.links a{color:var(--blue);font-size:.84rem;text-decoration:none}
.links a:hover{text-decoration:underline}
.empty{color:var(--muted);text-align:center;padding:50px;background:var(--panel);border:1px solid var(--border);border-radius:14px}
footer{text-align:center;color:var(--faint);font-size:.78rem;margin-top:24px}
</style></head>
<body><div class="wrap">
<h1>Job Matches &mdash; __NAME__</h1>
<p class="gen">Generated __GENERATED__</p>
<div class="stats">
<div class="stat"><div class="n">__COUNT__</div><div class="l">Openings</div></div>
<div class="stat"><div class="n">__TOP__</div><div class="l">Top match</div></div>
<div class="stat"><div class="n" style="font-size:.95rem;font-weight:600">__SOURCES__</div><div class="l">Sources</div></div>
</div>
<div class="terms">__TERMS__</div>
__CARDS__
<footer>Generated by Job Co-Pilot &middot; ranked by r&eacute;sum&eacute; match score</footer>
</div></body></html>
"""


def write_html_report(payload: dict, html_path: str) -> None:
    """Render the ranked matches as a self-contained graphical HTML page."""
    e = html.escape
    source_labels = {
        "linkedin": "LinkedIn", "indeed": "Indeed", "glassdoor": "Glassdoor",
        "zip_recruiter": "ZipRecruiter", "linkedin_post": "LinkedIn Post",
    }

    def tone(score):
        if not isinstance(score, int):
            return "#8b949e"
        if score >= 80:
            return "#3fb950"
        if score >= 50:
            return "#d29922"
        return "#f85149"

    results = payload.get("results", [])
    scores = [
        j.get("match", {}).get("match_score")
        for j in results
        if isinstance(j.get("match", {}).get("match_score"), int)
    ]
    top_score = f"{max(scores)}%" if scores else "&mdash;"

    counts: dict[str, int] = {}
    for j in results:
        src = j.get("source", "")
        counts[src] = counts.get(src, 0) + 1
    sources_str = " · ".join(
        f"{source_labels.get(s, s)} {n}" for s, n in counts.items()
    ) or "&mdash;"

    cards = []
    for i, job in enumerate(results, 1):
        m = job.get("match", {}) or {}
        score = m.get("match_score")
        color = tone(score)
        score_txt = f"{score}%" if isinstance(score, int) else "&mdash;"
        pct = score if isinstance(score, int) else 0

        verdict = m.get("match_verdict") or ""
        verdict_html = f'<span class="verdict">{e(verdict)}</span>' if verdict else ""

        sub = " &middot; ".join(
            e(b) for b in [job.get("company"), job.get("location"), job.get("date_posted")] if b
        )

        appl = job.get("applicants_label") or ""
        appl_html = f'<p class="appl">{e(appl)}</p>' if appl else ""

        text = (m.get("match_rationale") or job.get("description") or "").strip()
        rationale_html = (
            f'<p class="rationale">{e(text[:280])}{"…" if len(text) > 280 else ""}</p>'
            if text else ""
        )

        chips = ""
        if m.get("matching_skills"):
            chips += '<div class="chips">' + "".join(
                f'<span class="chip ok">+ {e(s)}</span>' for s in m["matching_skills"][:12]
            ) + "</div>"
        if m.get("missing_skills"):
            chips += '<div class="chips">' + "".join(
                f'<span class="chip miss">{e(s)}</span>' for s in m["missing_skills"][:12]
            ) + "</div>"

        contact_html = (
            f'<p class="contact">Contact: {e(job["contact"])}</p>' if job.get("contact") else ""
        )

        links_html = (
            f'<div class="links"><a href="{e(job["job_url"])}" target="_blank" rel="noopener">Open posting &#8599;</a></div>'
            if job.get("job_url") else ""
        )

        src_label = source_labels.get(job.get("source", ""), job.get("source", "")) or "Source"

        cards.append(
            '<article class="card">'
            f'<div class="ring" style="background:conic-gradient({color} {pct}%, #21262d 0)">'
            f'<div class="hole"><span style="color:{color}">{score_txt}</span><small>match</small></div>'
            "</div>"
            '<div class="meta">'
            f'<div class="toprow"><span class="rank">#{i}</span><span class="src">{e(src_label)}</span>{verdict_html}</div>'
            f'<h3>{e(job.get("title") or "Untitled role")}</h3>'
            f'<p class="sub">{sub}</p>'
            f"{appl_html}{rationale_html}{chips}{contact_html}{links_html}"
            "</div></article>"
        )

    cards_html = "\n".join(cards) if cards else '<p class="empty">No openings found.</p>'
    terms_html = "".join(
        f'<span class="term">{e(t)}</span>' for t in payload.get("search_terms", [])
    )

    doc = _HTML_TEMPLATE
    for token, value in {
        "__NAME__": e(payload.get("candidate", {}).get("name") or "Candidate"),
        "__GENERATED__": e(payload.get("generated_at", "")),
        "__COUNT__": str(len(results)),
        "__TOP__": top_score,
        "__SOURCES__": sources_str,
        "__TERMS__": terms_html,
        "__CARDS__": cards_html,
    }.items():
        doc = doc.replace(token, value)

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(doc)


def print_summary(results: list[dict], top: int) -> None:
    step(f"Top {min(top, len(results))} matches")
    if not results:
        info("No openings found. Try a broader --query or different --location/--sites.")
        return
    print(f"\n  {'#':>2}  {'SCORE':>5}  {'TITLE':<38}  {'COMPANY':<22}  SOURCE")
    print("  " + "-" * 88)
    for i, job in enumerate(results[:top], 1):
        score = job.get("match", {}).get("match_score")
        score_str = f"{score}%" if isinstance(score, int) else "  --"
        title = (job["title"][:36] + "..") if len(job["title"]) > 38 else job["title"]
        company = (job.get("company") or "")[:22]
        print(f"  {i:>2}  {score_str:>5}  {title:<38}  {company:<22}  {job['source']}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resume-aware job finder & matcher (Gemini powered).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resume", help="Path to a specific resume file (PDF/TXT/MD).")
    p.add_argument("--resume-dir", default="resumes", help="Folder to read the resume from.")
    p.add_argument("--query", help="Override search term (default: derived from your resume).")
    p.add_argument("--location", default="", help="Location filter, e.g. 'Remote' or 'San Francisco'.")
    p.add_argument("--sites", default="linkedin,indeed,glassdoor,zip_recruiter", help="Comma-separated portals to search.")
    p.add_argument("--results", type=int, default=40, help="Results wanted per portal per query.")
    p.add_argument("--hours-old", type=int, default=168, help="Only jobs posted within this many hours (0 = no limit).")
    p.add_argument("--max-queries", type=int, default=5, help="How many resume-derived titles/keywords to search (more = broader, more experience-aligned).")
    p.add_argument("--max-posts", type=int, default=15, help="Max LinkedIn posts to scrape.")
    p.add_argument("--no-posts", action="store_true", help="Skip LinkedIn post scraping.")
    p.add_argument("--max-applicants", type=int, default=10, help="Drop LinkedIn jobs with >= this many applicants (0 = off; best-effort).")
    p.add_argument("--max-match", type=int, default=60, help="Cap how many openings to AI-score.")
    p.add_argument("--top", type=int, default=40, help="How many matches to print to the console.")
    p.add_argument("--env-file", default=".env", help="Path to a .env file holding GEMINI_API_KEY.")
    p.add_argument("--api-key", help="Gemini API key (overrides the env var and .env file).")
    p.add_argument("--output", default="job_matches", help="Output filename stem (.json + .md).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env_file = load_env_file(args.env_file)
    api_key = (
        args.api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or env_file.get("GEMINI_API_KEY")
        or env_file.get("GOOGLE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        die(
            "No Gemini API key found. Provide it in any one of these ways:\n"
            f"  1. Create a file named '{args.env_file}' in this folder containing:\n"
            "         GEMINI_API_KEY=your-key-here\n"
            "  2. Set an environment variable:\n"
            "         setx GEMINI_API_KEY \"your-key\"   (Windows - then REOPEN the terminal)\n"
            "         export GEMINI_API_KEY=\"your-key\"  (macOS/Linux)\n"
            "  3. Pass it inline:  python job_copilot.py --api-key your-key\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )

    print("=" * 70)
    print("  JOB CO-PILOT  -  resume-aware job finder & matcher")
    print("=" * 70)

    step("Connecting to Gemini")
    model_name = configure_gemini(api_key)

    step("Reading your resume")
    resume_path = pick_resume_file(args.resume, args.resume_dir)
    info(f"Resume: {resume_path}")
    raw_text = extract_text(resume_path)
    info(f"Extracted {len(raw_text):,} characters of text.")

    step("Parsing resume with Gemini")
    profile = parse_resume(model_name, raw_text)
    info(f"Candidate: {profile.get('name') or 'unknown'}")

    # Decide what to search for.
    if args.query:
        terms = [args.query]
    else:
        terms = (profile.get("suggested_titles") or []) + (profile.get("suggested_keywords") or [])
        terms = [t for t in dict.fromkeys(terms) if t]  # dedupe, keep order
    terms = terms[: max(1, args.max_queries)]
    if not terms:
        die("Could not derive search terms from the resume. Pass --query \"...\" manually.")
    info(f"Search terms: {', '.join(terms)}")

    sites = [s.strip().lower() for s in args.sites.split(",") if s.strip()]

    recency = f" (last {args.hours_old}h)" if args.hours_old and args.hours_old > 0 else ""
    step(f"Searching job portals{recency}")
    jobs = search_portals(terms, args.location, sites, args.results, args.hours_old)
    info(f"Collected {len(jobs)} portal listings.")

    if not args.no_posts:
        step(f"Scraping LinkedIn hiring posts{recency}")
        posts = find_linkedin_posts(terms, args.location, args.max_posts, args.hours_old)
        info(f"Found {len(posts)} candidate posts; extracting details...")
        for i, post in enumerate(posts, 1):
            extracted = extract_post_job(model_name, post)
            if extracted:
                jobs.append(extracted)
                info(f"  [{i}/{len(posts)}] kept: {extracted['title']} @ {extracted['company'] or '?'}")
            else:
                info(f"  [{i}/{len(posts)}] skipped (not a job post or blocked)")
            time.sleep(0.5)

    jobs = dedupe_jobs(jobs)
    if not jobs:
        die("No openings found from any source. Try --query, a different --location, or more --sites.")

    if args.max_applicants and args.max_applicants > 0:
        step(f"Filtering to jobs with fewer than {args.max_applicants} applicants (LinkedIn, best-effort)")
        jobs, dropped = filter_by_applicants(jobs, args.max_applicants)
        info(f"Dropped {dropped} job(s) with {args.max_applicants}+ applicants; {len(jobs)} remain.")
        if not jobs:
            die("Every job was filtered out. Raise --max-applicants, widen --hours-old, or broaden the search.")

    if len(jobs) < 30:
        info(
            f"Only {len(jobs)} openings after filters. For more volume, relax the filters — "
            "e.g. --hours-old 720 (last month), --max-applicants 0 (keep high-competition "
            "LinkedIn jobs too), or --results 60."
        )

    if len(jobs) > args.max_match:
        info(f"Limiting AI scoring to {args.max_match} of {len(jobs)} openings (use --max-match to change).")
        jobs = jobs[: args.max_match]

    empties = sum(1 for j in jobs if not (j.get("description") or "").strip())
    if empties:
        info(
            f"Note: {empties}/{len(jobs)} openings arrived with no description "
            f"(matched on title only — a board may be rate-limiting description fetches)."
        )

    step(f"Scoring {len(jobs)} openings against your resume")
    for i, job in enumerate(jobs, 1):
        job["match"] = match_job(model_name, raw_text, job)
        score = job["match"].get("match_score")
        flag = "" if (job.get("description") or "").strip() else "  [title-only]"
        info(f"  [{i}/{len(jobs)}] {score if score is not None else '--'}%  {job['title'][:48]}{flag}")
        time.sleep(0.4)

    jobs.sort(key=lambda j: j.get("match", {}).get("match_score") or -1, reverse=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "resume_file": resume_path,
        "candidate": {
            "name": profile.get("name", ""),
            "location": profile.get("contact", {}).get("location", ""),
        },
        "search_terms": terms,
        "results": jobs,
    }

    json_path = f"{args.output}.json"
    md_path = f"{args.output}.md"
    html_path = f"{args.output}.html"
    write_reports(payload, json_path, md_path)
    write_html_report(payload, html_path)

    print_summary(jobs, args.top)
    step("Done")
    info(f"Reports saved: {json_path}  ·  {md_path}  ·  {html_path}")
    info(f"Open {html_path} in your browser for the graphical view.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        die("Interrupted.", code=130)
