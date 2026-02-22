import os
import re
import logging
import uuid
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

try:
    from content_agent import safe_groq_chat
except Exception:
    # Fallback to a simple stub for environments without GROQ configured; tests should monkeypatch this.
    def safe_groq_chat(prompt: str, *args, **kwargs):
        raise RuntimeError("Groq client not configured. Set GROQ_API_KEY or monkeypatch safe_groq_chat for tests.")
from database import save_social_post, get_reddit_posts_grouped

logger = logging.getLogger(__name__)

app = FastAPI(title="Reddit Agent", version="0.1.0")

# Models
class BrandProfile(BaseModel):
    brand_profile: str

class KeywordsResponse(BaseModel):
    keywords: List[str]

class SubredditSearchRequest(BaseModel):
    keywords: List[str]
    max_results: Optional[int] = 10

class SubredditInfo(BaseModel):
    name: str
    title: Optional[str]
    subscribers: Optional[int]
    active_user_count: Optional[int]
    avg_upvotes_last_posts: Optional[float]
    url: Optional[str]

class PostRequest(BaseModel):
    subreddit: str
    title: str
    body: Optional[str] = ""
    send: Optional[bool] = False


def build_keyword_prompt(brand_profile: str) -> str:
    return f"""
You are an expert in Reddit communities.

Given this brand profile:
{brand_profile}

Return a JSON object with a single key `keywords` containing a list of short search terms or subreddit name suggestions (no more than 12 items) that would help find relevant subreddits for this brand. Return only JSON.
"""


def extract_subreddit_keywords_sync(brand_profile: str) -> List[str]:
    prompt = build_keyword_prompt(brand_profile)
    resp = safe_groq_chat(prompt)
    # Expecting {'keywords': [ ... ]} or {'raw_text': ...}
    if isinstance(resp, dict) and "keywords" in resp and isinstance(resp["keywords"], list):
        return [k.strip() for k in resp["keywords"] if isinstance(k, str)]
    # try to parse free text
    text = resp.get("raw_text") if isinstance(resp, dict) else str(resp)
    # very naive extraction: lines or csv
    parts = re.split(r"[\n,;]+", text)
    keywords = [p.strip() for p in parts if p.strip()][:12]
    return keywords


def reddit_search_subreddits_unauthenticated(q: str, limit: int = 8) -> List[Dict[str, Any]]:
    headers = {"User-Agent": os.getenv("REDDIT_USER_AGENT", "hacking-agent/0.1")}
    url = f"https://www.reddit.com/subreddits/search.json?q={requests.utils.quote(q)}&limit={limit}"
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = []
    for child in data.get("data", {}).get("children", []):
        sd = child.get("data", {})
        results.append({
            "name": sd.get("display_name"),
            "title": sd.get("title"),
            "subscribers": sd.get("subscribers"),
            "active_user_count": sd.get("active_user_count"),
            "url": sd.get("url")
        })
    return results


def estimate_engagement_for_subreddit(subreddit: str, sample_posts: int = 6) -> float:
    # fetch recent posts (new) and compute avg upvotes
    headers = {"User-Agent": os.getenv("REDDIT_USER_AGENT", "hacking-agent/0.1")}
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={sample_posts}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        scores = []
        for child in data.get("data", {}).get("children", []):
            sd = child.get("data", {})
            scores.append(sd.get("score", 0) or 0)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
    except Exception:
        return 0.0


def find_top_subreddits(keywords: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
    seen = {}
    for k in keywords:
        try:
            candidates = reddit_search_subreddits_unauthenticated(k, limit=6)
        except Exception as e:
            logger.warning(f"Subreddit search failed for {k}: {e}")
            continue
        for c in candidates:
            name = c.get("name")
            if not name:
                continue
            if name in seen:
                continue
            # get engagement metric
            engagement = estimate_engagement_for_subreddit(name)
            seen[name] = {
                "name": name,
                "title": c.get("title"),
                "subscribers": c.get("subscribers"),
                "active_user_count": c.get("active_user_count"),
                "avg_upvotes_last_posts": engagement,
                "url": c.get("url")
            }
    # sort by subscribers * 0.7 + engagement * 0.3
    lst = list(seen.values())
    lst.sort(key=lambda x: ((x.get("subscribers") or 0) * 0.7 + (x.get("avg_upvotes_last_posts") or 0) * 0.3), reverse=True)
    return lst[:max_results]


def build_post_prompt(subreddit: str, brand_profile: str) -> str:
    return f"""
You are a copywriter crafting a Reddit post for the subreddit r/{subreddit} given this brand profile:

{brand_profile}

Return only a JSON object: {{"title":"...","body":"..."}}. Title should be concise (<=300 chars); body suitable for a Reddit text post.
"""


def generate_reddit_post_text(subreddit: str, brand_profile: str) -> Dict[str, str]:
    prompt = build_post_prompt(subreddit, brand_profile)
    resp = safe_groq_chat(prompt)
    if isinstance(resp, dict) and "title" in resp:
        return {"title": resp.get("title"), "body": resp.get("body", "")}
    # fallback: parse raw_text heuristically
    raw = resp.get("raw_text") if isinstance(resp, dict) else str(resp)
    # naive split
    lines = raw.strip().split("\n")
    title = lines[0][:300]
    body = "\n".join(lines[1:])
    return {"title": title, "body": body}


def post_to_reddit_authenticated(subreddit: str, title: str, body: str) -> Dict[str, Any]:
    try:
        import praw
    except Exception as e:
        raise HTTPException(status_code=500, detail="praw library not installed")

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    user_agent = os.getenv("REDDIT_USER_AGENT", "hacking-agent/0.1")

    if not all([client_id, client_secret, username, password]):
        raise HTTPException(status_code=400, detail="Reddit credentials not configured in environment variables")

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, username=username, password=password, user_agent=user_agent)
    sr = reddit.subreddit(subreddit)
    submission = sr.submit(title=title, selftext=body)
    return {"id": submission.id, "url": submission.url}


# Endpoints
@app.post("/extract-keywords", response_model=KeywordsResponse)
def extract_keywords_endpoint(req: BrandProfile):
    kws = extract_subreddit_keywords_sync(req.brand_profile)
    return KeywordsResponse(keywords=kws)


@app.post("/search-subreddits")
def search_subreddits(req: SubredditSearchRequest):
    subs = find_top_subreddits(req.keywords, max_results=req.max_results or 10)
    return {"subreddits": subs}


@app.post("/generate-post")
def generate_post(req: BrandProfile, subreddit: str):
    post = generate_reddit_post_text(subreddit, req.brand_profile)
    return post


@app.post("/post")
def post_endpoint(req: PostRequest):
    # create content and optionally send
    if req.send:
        result = post_to_reddit_authenticated(req.subreddit, req.title, req.body or "")
        # save to DB as a social post with platform 'reddit'
        try:
            save_social_post(result.get("id"), "reddit", result.get("url"))
        except Exception as e:
            logger.warning(f"Could not save reddit post: {e}")
        return result
    else:
        # dry-run
        return {"id": None, "url": None, "title": req.title, "body": req.body}


@app.get("/reddit-posts-summary")
def reddit_posts_summary():
    return get_reddit_posts_grouped()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)