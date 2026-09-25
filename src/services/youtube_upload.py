import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# YouTube Data API settings (used only by this upload module).
YOUTUBE_CLIENT_SECRETS = os.getenv("YOUTUBE_CLIENT_SECRETS", "client_secrets.json")
YOUTUBE_TOKEN_FILE = os.getenv("YOUTUBE_TOKEN_FILE", "token.json")
# Fixed localhost port for the OAuth redirect. If your client is a
# "Web application" type, register this exact URI in Google Cloud Console:
#   http://localhost:8080/  and  http://127.0.0.1:8080/
# Desktop-app clients accept any loopback port automatically.
YOUTUBE_REDIRECT_PORT = int(os.getenv("YOUTUBE_REDIRECT_PORT", "8080"))
# public | private | unlisted
YOUTUBE_PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY_STATUS", "private")
# YouTube video category id (22 = People & Blogs, 27 = Education)
YOUTUBE_CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "27")
# Default tags appended to every upload (e.g. "shorts", "faceless")
YOUTUBE_TAGS = os.getenv("YOUTUBE_TAGS", "")
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

from src.services.providers import call_text


def get_metadata_prompt(topic, script_text):
    return f"""You are a YouTube SEO expert for a faceless mystery/dark-history Shorts channel. Generate a title, description, and tags for a video.

Topic: {topic}

Video script (narration):
{script_text}

Return ONLY valid JSON (no markdown, no code fences) in this exact shape:
{{
  "title": "<40-65 char title>",
  "title_alternates": ["<alt title 1>", "<alt title 2>", "<alt title 3>"],
  "description": "<300-500 word description>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
}}

===== THINK FIRST (do this internally before writing the final JSON) =====
You MUST go through this full thought process before outputting anything. The quality of the output depends on it.

STEP 1 — Read the story and extract the emotional core.
Ask yourself: what is the ONE thing in this script that makes a viewer feel something?
- A disappearance with no answer -> curiosity + unease
- A cover-up, a death squad, an experiment on humans -> injustice + outrage
- Something hidden for decades that just came out -> shock + curiosity
Pick the strongest single emotional beat. This becomes the heart of the title.

STEP 2 — Draft 4 candidate titles. Each must use a DIFFERENT archetype AND a DIFFERENT emotion angle, and each must be an answer to "why would someone click this?"
1. Listicle/dark-truth + fear: "<Topic>: the dark truth" with a dread word (nightmare, horror, disturbing)
2. Question-hook + curiosity: a question that cannot be answered without watching ("What happened to X?")
3. Statement-of-consequence + injustice: the outcome framed as wrongness ("They got away with it", "Covered up for decades")
4. Myth-bust/secret-reveal + shock: the hidden story ("The untold story of X", "Everyone believed X, but...")

STEP 3 — Score all 4 candidates against the CHECKLIST below. Keep the top 2. Merge the best emotion with the best structure.

STEP 4 — Write the final "title" (highest scoring) and 3 "title_alternates" (the others, fixed to pass every checklist rule).

===== CHECKLIST (every candidate must pass before being used) =====
- 40-65 characters total.
- Keyword or topic name within the first 30 characters.
- Contains at least ONE concrete specific (real number, name, date, or place) — not just an adjective doing the work.
- Triggers at least one of: curiosity gap (unanswered open loop), fear/dread, or a sense of injustice/wrongness. If a title feels neutral or informative, rewrite it — neutral titles get skipped.
- Does NOT resolve the core payoff in the title (curiosity gap stays open — the title must make someone need to watch to find out).
- Does NOT promise something the script doesn't actually deliver (overpromising kills retention).
- Uses negative/loss framing where the story supports it, not upbeat framing.
- At most ONE capitalized word (the one whose removal changes meaning).
- Not a near-duplicate structure of the other candidates or of previously-used titles.

===== DESCRIPTION PRINCIPLES (follow strictly) =====
- First 100-150 characters must be ad copy that sells the click (this is the search snippet).
- Total length: 300-500 words.
- Structure: hook (first 2 lines) -> what the video covers -> key takeaways/list -> hashtags at the end (2-3 max).
- Naturally integrate the topic keywords throughout.
- The description is a cold-start signal: it must clearly tell the algorithm what the video is about.

===== TAGS =====
- 8-15 tags max. Include the primary keyword first, then relevant long-tail variants. Lowercase, no punctuation except hyphens."""


def get_metadata_verifier_prompt(topic, script_text, metadata_json):
    return f"""You are a ruthless YouTube title/description quality auditor for a faceless mystery/dark-history Shorts channel. Your ONLY job is to verify that the generated metadata meets every principle, and reject anything that would get scrolled past.

Topic: {topic}

Video script (narration):
{script_text}

Generated metadata to verify (JSON):
{metadata_json}

===== YOUR JOB =====
1. Read the topic and script. Identify the strongest emotional beat (curiosity gap, fear/dread, or injustice/wrongness).
2. Audit the "title" against EVERY rule below. Be harsh. A title that is merely "fine" FAILS.
3. Audit the description and tags too.
4. Return ONLY valid JSON (no markdown, no code fences) in this exact shape:
{{
  "verdict": "PASS" or "FAIL",
  "title_score": <0-100>,
  "description_score": <0-100>,
  "emotion_triggered": "<curiosity|fear|injustice|shock|none>",
  "fails": ["<exact rule that failed>", ...],
  "fixed_title": "<a better title that passes every rule, ONLY if verdict is FAIL, else \"\">",
  "fixed_description": "<a better description if needed, else \"\">",
  "fixed_tags": ["<tags>"]
}}

===== TITLE CHECKLIST (fail on ANY miss) =====
- 40-65 characters.
- Keyword/topic name in first 30 characters.
- Has at least ONE concrete specific (number, name, date, place).
- Triggers a real emotion: curiosity gap, fear/dread, or injustice. "Informative" or "sounds like a headline" = FAIL. The title must make a scroller stop and click out of a triggered emotion.
- Does NOT spoil the payoff (open loop must remain).
- Doesn't overpromise (must match what the script delivers).
- Not generic — could this title be pasted onto a different video in the same niche and still fit? If yes, FAIL (too generic, no hook).
- At most ONE capitalized word.
- No filler words at the start ("The mysterious case of..." type openings are weak — prefer naming the concrete thing immediately).

===== DESCRIPTION CHECKLIST =====
- First 100-150 chars are click-selling ad copy, not a dry summary.
- 300-500 words total.
- Has a hook, covers what the video shows, lists takeaways, ends with 2-3 hashtags.
- Topic keywords appear naturally throughout.

===== TAGS CHECKLIST =====
- 8-15 tags. Primary keyword first. Lowercase.

If verdict is FAIL, the "fixed_title"/"fixed_description"/"fixed_tags" MUST be genuinely improved versions that pass every rule — this is what gets used. Never return empty fixes on a FAIL."""


def get_metadata_fix_prompt(topic, script_text, old_metadata_json, verifier_json):
    return f"""You are a YouTube SEO expert. Your previous metadata was rejected by a verifier. Rewrite it to fully pass every rule.

Topic: {topic}

Video script:
{script_text}

Previous (rejected) metadata:
{old_metadata_json}

Verifier feedback:
{verifier_json}

Fix EVERY failing rule the verifier flagged. The title must trigger a strong emotion (curiosity gap, fear/dread, or injustice) that makes a user click. Output the corrected metadata as valid JSON in this exact shape:
{{
  "title": "<40-65 char title>",
  "title_alternates": ["<alt title 1>", "<alt title 2>", "<alt title 3>"],
  "description": "<300-500 word description>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
}}
No markdown, no code fences."""


def get_credentials():
    """Load OAuth credentials from token file, refresh if expired, or run browser flow.

    Requires a client_secrets.json (Desktop OAuth client) in the project root
    on first run. The refresh token is persisted to token.json afterwards.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = None
    if os.path.exists(YOUTUBE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(YOUTUBE_TOKEN_FILE, YOUTUBE_SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    if not creds or not creds.valid:
        if not os.path.exists(YOUTUBE_CLIENT_SECRETS):
            raise FileNotFoundError(
                f"{YOUTUBE_CLIENT_SECRETS} not found. "
                "Enable the YouTube Data API v3 and download your OAuth client "
                "(Desktop app) credentials from Google Cloud Console."
            )

        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(YOUTUBE_CLIENT_SECRETS, YOUTUBE_SCOPES)
        creds = flow.run_local_server(port=YOUTUBE_REDIRECT_PORT, prompt="consent")
        with open(YOUTUBE_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        print(f"    [youtube] OAuth token saved to {YOUTUBE_TOKEN_FILE}")

    return creds


def build_client(creds=None):
    """Return an authenticated YouTube Data API v3 client."""
    from googleapiclient.discovery import build
    from google_auth_httplib2 import AuthorizedHttp

    if creds is None:
        creds = get_credentials()

    http = AuthorizedHttp(creds)
    return build("youtube", "v3", http=http, cache_discovery=False)


def _format_script(script: dict) -> str:
    """Flatten the script lines into readable narration text."""
    topic = script.get("topic", "")
    lines = script.get("lines", [])
    body = "\n".join(f"{i + 1}. {l.get('text', '')}" for i, l in enumerate(lines))
    return f"Topic: {topic}\n\n{body}"


def _fallback_metadata(topic: str, script_text: str) -> dict:
    """Deterministic metadata used when the LLM call fails."""
    title_words = topic.split()
    keyword = " ".join(title_words[:4]).rstrip(".")
    title = f"{keyword}: the dark truth (True Story)"
    title = title[:65] if len(title) > 65 else title

    hook = f"Inside the shocking story of {topic}. What really happened is worse than you think."
    covered = "\n".join(
        f"- {l.strip()}" for l in script_text.splitlines() if l.strip()
    )
    description = (
        f"{hook}\n\n"
        f"In this video we break down {topic} from start to finish.\n\n"
        f"{covered}\n\n"
        "Subscribe for more dark histories and untold stories."
    )
    tags = [
        topic.lower().rstrip("."),
        "history",
        "documentary",
        "dark history",
        "shorts",
        "untold story",
    ]
    return {"title": title, "description": description, "tags": tags}


def _extract_json(raw: str) -> dict:
    """Pull the first {...} JSON object out of an LLM response."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _clean_metadata(meta: dict) -> dict:
    """Validate and normalize a metadata dict; empty values fall back to fallback."""
    title = (meta.get("title") or "").strip()
    description = (meta.get("description") or "").strip()
    tags = meta.get("tags") or []

    if not title or not description or len(title) > 100 or len(description) < 50:
        return {}

    return {
        "title": title,
        "title_alternates": [t for t in (meta.get("title_alternates") or []) if isinstance(t, str)][:3],
        "description": description,
        "tags": [str(t).lower().replace(" ", "-") for t in tags if str(t).strip()][:15],
    }


def _metadata_call(prompt: str, temperature: float = 0.6) -> dict:
    """One metadata LLM call; returns parsed JSON or {} on any failure."""
    try:
        raw = call_text(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return _extract_json(raw)
    except Exception as e:
        print(f"    [youtube] LLM call failed: {e}")
        return {}


def _verify_round(topic: str, script_text: str, meta: dict) -> tuple[str, dict]:
    """Run one verifier pass. Returns (verdict, verifier) for the caller."""
    verifier = _metadata_call(
        get_metadata_verifier_prompt(topic, script_text, json.dumps(meta, indent=2)),
        temperature=0.2,
    )
    verdict = (verifier.get("verdict") or "").strip().upper()
    print(
        f"    [youtube] Verifier round: {verdict}"
        f" (title {verifier.get('title_score')}/100,"
        f" desc {verifier.get('description_score')}/100,"
        f" emotion: {verifier.get('emotion_triggered')})"
    )
    return verdict, verifier


def generate_metadata(topic: str, script: dict) -> dict:
    """Generate title, description, and tags using minimax-m3:cloud (Ollama).

    Two-pass with a verifier:
      1. The generator is told to think through title/description tactics first.
      2. A verifier (same model) audits the result against every rule and returns
         a PASS/FAIL verdict. On FAIL it is fed the verifier's feedback and
         rewrites, up to MAX_VERIFY_ROUNDS.
    Falls back to deterministic metadata if the LLM calls fail.
    """
    MAX_VERIFY_ROUNDS = 2
    script_text = _format_script(script)

    # Pass 1: generate (think-first prompt).
    meta = _metadata_call(get_metadata_prompt(topic, script_text), temperature=0.7)
    if not meta:
        print("    [youtube] Metadata generation failed, using fallback")
        return _fallback_metadata(topic, script_text)

    # Pass 2+: verify, and rewrite on FAIL until it passes.
    for _ in range(1, MAX_VERIFY_ROUNDS + 1):
        verdict, verifier = _verify_round(topic, script_text, meta)
        if verdict == "PASS":
            # Prefer the verifier's fixed title if it supplied a stronger one.
            fixed_title = (verifier.get("fixed_title") or "").strip()
            if fixed_title and _clean_metadata({"title": fixed_title, "description": meta.get("description", ""), "tags": meta.get("tags", [])}):
                meta["title"] = fixed_title
            break

        # FAIL: rewrite with the verifier's feedback.
        meta = _metadata_call(
            get_metadata_fix_prompt(topic, script_text, json.dumps(meta, indent=2), json.dumps(verifier, indent=2)),
            temperature=0.7,
        )
        if not meta:
            break

    cleaned = _clean_metadata(meta)
    if not cleaned:
        print("    [youtube] LLM metadata invalid after verification, using fallback")
        return _fallback_metadata(topic, script_text)

    return cleaned


def upload_video(
    youtube,
    video_path: str,
    title: str,
    description: str,
    tags: list,
    privacy_status: str = YOUTUBE_PRIVACY_STATUS,
) -> str:
    """Upload an MP4 to YouTube with a resumable upload. Returns the video id."""
    from googleapiclient.http import MediaFileUpload

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    extra_tags = [t.strip() for t in YOUTUBE_TAGS.split(",") if t.strip()]
    all_tags = list(dict.fromkeys([t for t in (tags + extra_tags) if t]))

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": all_tags,
            "categoryId": YOUTUBE_CATEGORY_ID,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"    [youtube] Uploading... {int(status.progress() * 100)}%")

    video_id = response.get("id")
    print(f"    [youtube] Uploaded video id: {video_id}")
    return video_id


def wait_for_processing(youtube, video_id: str, timeout: int = 300) -> str:
    """Poll the video until YouTube finishes processing. Returns final status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = (
            youtube.videos()
            .list(part="status", id=video_id)
            .execute()
        )
        items = res.get("items", [])
        if items:
            status = items[0]["status"].get("uploadStatus", "unknown")
            print(f"    [youtube] Upload status: {status}")
            if status == "processed":
                return status
        time.sleep(5)
    print("    [youtube] Timed out waiting for processing (video still uploading on YT).")
    return "pending"


def set_thumbnail(youtube, video_id: str, thumbnail_path: str) -> None:
    """Attach a local image as the video thumbnail."""
    from googleapiclient.http import MediaFileUpload

    if not os.path.exists(thumbnail_path):
        print(f"    [youtube] Thumbnail not found: {thumbnail_path}")
        return

    youtube.thumbnails().set(
        videoId=video_id,
        media_body=MediaFileUpload(thumbnail_path),
    ).execute()
    print(f"    [youtube] Thumbnail set for {video_id}")


def publish_video(video_path: str, topic: str, script: dict, thumbnail_path: str = None) -> str:
    """Generate metadata, upload the video, and report its status.

    Returns the YouTube video id.
    """
    print(f"\n📺 Publishing to YouTube: {topic}")

    print("    [youtube] Generating title & description...")
    meta = generate_metadata(topic, script)
    title = meta["title"]
    description = meta["description"]
    tags =  meta["tags"]

    print(f"    [youtube] Title: {title}")

    creds = get_credentials()
    youtube = build_client(creds)

    video_id = upload_video(
        youtube,
        video_path,
        title=title,
        description=description,
        tags=tags,
    )

    if thumbnail_path:
        set_thumbnail(youtube, video_id, thumbnail_path)

    wait_for_processing(youtube, video_id)

    url = f"https://youtu.be/{video_id}"
    print(f"\n✅ Published! {url}")
    return video_id
