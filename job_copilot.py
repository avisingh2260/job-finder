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
  6. Saves a JSON + Markdown report and prints a ranked summary to the console.

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
  python job_copilot.py --help

NOTES
  * Your resume and API key stay on your machine; only Gemini calls leave it.
  * Respect each site's Terms of Service and rate limits. Scrapers can break or
    hit CAPTCHAs - the script degrades gracefully and keeps going when a source
    fails, so a partial run still produces useful results.
"""

from __future__ import annotations

import argparse
import glob
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
            return json.loads(_strip_json_fences(resp.text))
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
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()

    text = text.strip()
    if not text:
        die(f"Could not extract any text from '{path}'.")
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
def search_portals(terms: list[str], location: str, sites: list[str], results: int) -> list[dict]:
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
                }
            )
    return collected


# --------------------------------------------------------------------------- #
#  LinkedIn post scraping (DuckDuckGo discovery + page fetch + Gemini extract)
# --------------------------------------------------------------------------- #
def _ddgs_search(query: str, max_results: int) -> list[dict]:
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
            return list(ddgs.text(query, max_results=max_results))
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


def find_linkedin_posts(terms: list[str], location: str, max_posts: int) -> list[dict]:
    seen_urls: set[str] = set()
    posts: list[dict] = []

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
        for hit in _ddgs_search(query, max_results=max_posts):
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
    desc = job.get("description") or ""
    if not desc.strip():
        return {
            "match_score": 0,
            "match_verdict": "No description",
            "matching_skills": [],
            "missing_skills": [],
            "match_rationale": "No description was available to evaluate this opening.",
        }

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

Return ONLY a JSON object:
{{
  "match_score": 0,
  "match_verdict": "Excellent Fit (85-100) | Strong Competitor (70-84) | Skill Gap (50-69) | Not Aligned (0-49)",
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
    p.add_argument("--sites", default="linkedin,indeed", help="Comma-separated portals to search.")
    p.add_argument("--results", type=int, default=15, help="Results wanted per portal per query.")
    p.add_argument("--max-queries", type=int, default=2, help="How many resume-derived terms to search.")
    p.add_argument("--max-posts", type=int, default=8, help="Max LinkedIn posts to scrape.")
    p.add_argument("--no-posts", action="store_true", help="Skip LinkedIn post scraping.")
    p.add_argument("--max-match", type=int, default=30, help="Cap how many openings to AI-score.")
    p.add_argument("--top", type=int, default=15, help="How many matches to print to the console.")
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

    step("Searching job portals")
    jobs = search_portals(terms, args.location, sites, args.results)
    info(f"Collected {len(jobs)} portal listings.")

    if not args.no_posts:
        step("Scraping LinkedIn hiring posts")
        posts = find_linkedin_posts(terms, args.location, args.max_posts)
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

    if len(jobs) > args.max_match:
        info(f"Limiting AI scoring to {args.max_match} of {len(jobs)} openings (use --max-match to change).")
        jobs = jobs[: args.max_match]

    step(f"Scoring {len(jobs)} openings against your resume")
    for i, job in enumerate(jobs, 1):
        job["match"] = match_job(model_name, raw_text, job)
        score = job["match"].get("match_score")
        info(f"  [{i}/{len(jobs)}] {score if score is not None else '--'}%  {job['title'][:50]}")
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
    write_reports(payload, json_path, md_path)

    print_summary(jobs, args.top)
    step("Done")
    info(f"Full report saved to: {json_path}  and  {md_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        die("Interrupted.", code=130)
