import json
import os
import time

from src.config.prompt import (
    get_metadata_prompt,
    get_metadata_verifier_prompt,
    get_metadata_fix_prompt,
)
from src.config.settings import (
    YOUTUBE_CATEGORY_ID,
    YOUTUBE_CLIENT_SECRETS,
    YOUTUBE_PRIVACY_STATUS,
    YOUTUBE_REDIRECT_PORT,
    YOUTUBE_SCOPES,
    YOUTUBE_TAGS,
    YOUTUBE_TOKEN_FILE,
)
from src.services.script_builder import _call_ollama


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

    def _call(prompt: str, temperature: float = 0.6) -> dict:
        try:
            raw = _call_ollama(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return _extract_json(raw)
        except Exception as e:
            print(f"    [youtube] Ollama call failed: {e}")
            return {}

    # Pass 1: generate (think-first prompt).
    meta = _call(get_metadata_prompt(topic, script_text), temperature=0.7)
    if not meta:
        print("    [youtube] Metadata generation failed, using fallback")
        return _fallback_metadata(topic, script_text)

    # Pass 2+: verify, and rewrite on FAIL until it passes.
    for round_no in range(1, MAX_VERIFY_ROUNDS + 1):
        verifier = _call(
            get_metadata_verifier_prompt(topic, script_text, json.dumps(meta, indent=2)),
            temperature=0.2,
        )
        verdict = (verifier.get("verdict") or "").strip().upper()
        print(
            f"    [youtube] Verifier round {round_no}: {verdict}"
            f" (title {verifier.get('title_score')}/100,"
            f" desc {verifier.get('description_score')}/100,"
            f" emotion: {verifier.get('emotion_triggered')})"
        )

        if verdict == "PASS":
            # Prefer the verifier's fixed title if it supplied a stronger one.
            fixed_title = (verifier.get("fixed_title") or "").strip()
            if fixed_title and _clean_metadata({"title": fixed_title, "description": meta.get("description", ""), "tags": meta.get("tags", [])}):
                meta["title"] = fixed_title
            break

        # FAIL: rewrite with the verifier's feedback.
        meta = _call(
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


def update_video(youtube, video_id: str, title: str, description: str, tags: list) -> None:
    """Edit metadata of an already-uploaded video."""
    body = {
        "id": video_id,
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": YOUTUBE_CATEGORY_ID,
        },
    }
    youtube.videos().update(part="snippet", body=body).execute()
    print(f"    [youtube] Metadata updated for {video_id}")


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
