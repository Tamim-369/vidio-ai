import json
import os
import time

from src.config.prompt import get_metadata_prompt
from src.config.settings import (
    YOUTUBE_CATEGORY_ID,
    YOUTUBE_CLIENT_SECRETS,
    YOUTUBE_PRIVACY_STATUS,
    YOUTUBE_REDIRECT_PORT,
    YOUTUBE_SCOPES,
    YOUTUBE_TAGS,
    YOUTUBE_TOKEN_FILE,
)
from src.services.script_builder import _call_groq


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


def generate_metadata(topic: str, script: dict) -> dict:
    """Generate title, description, and tags using Groq + the SEO/CTR principles.

    Falls back to deterministic metadata if the LLM call fails.
    """
    script_text = _format_script(script)
    prompt = get_metadata_prompt(topic, script_text)

    try:
        raw = _call_groq(
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_retries=2,
        )
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            meta = json.loads(raw[start:end + 1])
        else:
            meta = {}
    except Exception as e:
        print(f"    [youtube] Metadata generation failed, using fallback: {e}")
        meta = {}

    title = meta.get("title") or ""
    description = meta.get("description") or ""
    tags = meta.get("tags") or []

    if not title or not description or len(title) > 100 or len(description) < 50:
        print("    [youtube] LLM metadata invalid, using fallback")
        return _fallback_metadata(topic, script_text)

    return {
        "title": title,
        "description": description,
        "tags": [str(t).lower().replace(" ", "-") for t in tags if str(t).strip()][:15],
    }


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
