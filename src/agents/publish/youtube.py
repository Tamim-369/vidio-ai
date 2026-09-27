import json
import os
import time
from urllib.parse import urlparse

from dotenv import load_dotenv


load_dotenv()

YOUTUBE_CLIENT_SECRETS = os.getenv("YOUTUBE_CLIENT_SECRETS", "client_secrets.json")
YOUTUBE_TOKEN_FILE = os.getenv("YOUTUBE_TOKEN_FILE", "token.json")
# OAuth client is of type "Web application". Google matches redirect_uri byte
# for byte, so the URI is read back out of client_secrets.json rather than
# assembled here -- see _registered_redirect_uri.
YOUTUBE_REDIRECT_PORT = int(os.getenv("YOUTUBE_REDIRECT_PORT", "8080"))
YOUTUBE_PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY_STATUS", "private")
# 42 "Shorts" is returned by videoCategories.list but videos.insert rejects it
# as invalidCategoryId: Shorts is derived from the vertical format, never
# assigned. 23 Comedy is assignable and fits the parody voice format.
YOUTUBE_CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "23")
YOUTUBE_TAGS = os.getenv("YOUTUBE_TAGS", "")
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]


from src.agents.publish.metadata import deterministic_metadata, generate_metadata


RETRYABLE_UPLOAD_STATUS = frozenset({408, 429, 500, 502, 503, 504})

OAUTH_URL_FILE = "oauth_url.txt"
OAUTH_TIMEOUT_SECONDS = 600


class _ConsentPrompt(str):
    """Consent prompt that also mirrors the URL to a file, unbroken.

    run_local_server() renders this with .format(url=...), which is the only
    hook that ever sees the finished URL. Building the URL beforehand instead
    would mint a different state and code_challenge, so the browser would
    open a URL the local server is not listening for.

    The file exists because the URL runs to roughly 450 characters: in an
    80-column terminal it wraps over six lines, and copying it from there
    carries newlines into the query string. Google answers a malformed query
    with a bare "400 ... That's all we know" naming no reason at all, which is
    indistinguishable from a real misconfiguration.
    """

    def __new__(cls, file_name: str):
        obj = super().__new__(
            cls,
            "Open this URL to authorize, then approve the consent screen:\n"
            "{url}\n(also saved to " + file_name + ")",
        )
        obj.file_name = file_name
        return obj

    def format(self, *args, **kwargs) -> str:
        rendered = super().format(*args, **kwargs)
        url = kwargs.get("url")
        if url:
            try:
                with open(self.file_name, "w") as fh:
                    fh.write(url + "\n")
            except OSError:
                pass
        return rendered


class UploadRejected(RuntimeError):
    """The API refused the upload in a way a retry cannot fix.

    Most 4xx responses mean the request itself is wrong -- bad metadata, an
    unknown category, an exceeded upload limit -- so rebuilding the video would
    fail identically. Raised so the batch stops instead of burning an attempt.
    """

    def __init__(self, message: str, status: int, reason: str | None = None):
        super().__init__(message)
        self.status = status
        self.reason = reason


def _describe_upload_error(exc) -> Exception:
    """Re-raise a transient API error, or convert a permanent one.

    The response body is the only place the real reason is recorded --
    googleapiclient surfaces just a status line, which is why a rejected upload
    otherwise looks identical to a malformed one.
    """
    from googleapiclient.errors import HttpError

    status = exc.resp.status
    body = (exc.content or b"").decode("utf-8", "replace")
    reason = None
    try:
        reason = json.loads(body)["error"]["errors"][0]["reason"]
    except (ValueError, KeyError, IndexError, TypeError):
        pass

    detail = f"[youtube] HTTP {status} reason={reason or 'unknown'}"
    if status in RETRYABLE_UPLOAD_STATUS:
        print(f"    {detail} -- transient, letting the batch retry")
        return exc
    print(f"    {detail} -- permanent, not retrying\n    {body[:1000]}")
    assert isinstance(exc, HttpError)
    return UploadRejected(f"{detail}: {body[:400]}", status, reason)


def _registered_redirect_uri() -> str | None:
    """Return the loopback redirect URI this client has registered.

    Google compares ``redirect_uri`` byte for byte, so replaying the value
    recorded in client_secrets.json makes ``redirect_uri_mismatch`` impossible
    for any URI the web client actually has. Returns None for a Desktop client,
    which registers no loopback URI and takes the library default.
    """
    try:
        with open(YOUTUBE_CLIENT_SECRETS) as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return None

    uris = (config.get("web") or {}).get("redirect_uris") or []
    for uri in uris:
        if urlparse(uri).hostname in ("localhost", "127.0.0.1"):
            return uri
    return None


def get_credentials():
    """Load OAuth credentials from token file, refresh if expired, or run browser flow.

    Requires a client_secrets.json (Web OAuth client) in the project root on
    first run. The refresh token is persisted to token.json afterwards.
    """
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import WSGITimeoutError
    from googleapiclient.errors import HttpError

    creds = None
    if os.path.exists(YOUTUBE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(YOUTUBE_TOKEN_FILE, YOUTUBE_SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            # A refresh token minted while the consent screen was in Testing
            # status expires after 7 days and the API reports it as
            # invalid_grant. Re-authorising is the only way back, so drop the
            # dead file and fall through to the browser flow below instead of
            # crashing on an unhandled exception.
            print(f"    [youtube] Stored token is no longer valid ({exc}); "
                  f"re-authorising.")
            os.remove(YOUTUBE_TOKEN_FILE)
            creds = None

    if not creds or not creds.valid:
        if not os.path.exists(YOUTUBE_CLIENT_SECRETS):
            raise FileNotFoundError(
                f"{YOUTUBE_CLIENT_SECRETS} not found. "
                "Enable the YouTube Data API v3, then create an OAuth client of "
                "type Web application in Google Cloud Console and add "
                "http://localhost:<port> as an authorized redirect URI. "
                "Download the JSON and save it as client_secrets.json."
            )

        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(YOUTUBE_CLIENT_SECRETS, YOUTUBE_SCOPES)
        # run_local_server() rebuilds the URI itself as http://{host}:{port}[/]
        # and ignores any redirect_uri argument, so the registered URI is
        # reproduced from its parts instead. Google matches redirect_uri byte
        # for byte, which is why a trailing slash difference is a hard failure.
        registered = _registered_redirect_uri()
        if registered:
            parsed = urlparse(registered)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            trailing_slash = registered.endswith("/")
            print(
                f"    [youtube] Registered redirect URI: {registered} "
                f"(host {host}, port {port}, trailing slash {trailing_slash})"
            )
        else:
            host, port, trailing_slash = "localhost", YOUTUBE_REDIRECT_PORT, True
            print(
                f"    [youtube] No loopback URI registered in {YOUTUBE_CLIENT_SECRETS}; "
                f"using http://{host}:{port}/"
            )
        try:
            creds = flow.run_local_server(
                host=host,
                port=port,
                prompt="consent",
                # Without access_type=offline Google issues an access token that
                # dies in an hour and NO refresh token, so every later run would
                # force the browser consent flow again. The whole batch uploader
                # depends on token.json surviving between runs.
                access_type="offline",
                redirect_uri_trailing_slash=trailing_slash,
                authorization_prompt_message=_ConsentPrompt(OAUTH_URL_FILE),
                timeout_seconds=OAUTH_TIMEOUT_SECONDS,
            )
        except HttpError as exc:
            body = (exc.content or b"").decode("utf-8", "replace")
            print(f"    [youtube] OAuth request failed: HTTP {exc.resp.status} "
                  f"{exc.resp.reason}\n    url: {exc.uri}\n    {body[:1000]}")
            if exc.resp.status in (401, 403):
                print("    A 401/403 here means the consent screen is in Testing "
                      "status and this Google account is not on the test-user list.")
            raise
        except WSGITimeoutError:
            # The local server was listening but Google never redirected back,
            # which means consent was never granted (or was granted in a
            # different browser than the one that opened).
            raise RuntimeError(
                f"No authorization received after {OAUTH_TIMEOUT_SECONDS // 60} minutes. "
                f"Open the URL in {OAUTH_URL_FILE} in the SAME browser profile that "
                f"is signed in to the YouTube channel, approve consent, and re-run. "
                f"If you approved it in another browser, sign that one in instead."
            ) from None
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

    from googleapiclient.errors import HttpError

    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
        except HttpError as exc:
            raise _describe_upload_error(exc) from exc
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

    # A script that carries its own title can build its own description and
    # tags from the quote it already contains. That path is unique by
    # construction -- the quote pool is deduplicated -- and it makes no model
    # call, where the LLM path costs a generate plus up to two verifier rounds
    # per video and will happily write the same description twice in a batch.
    # The LLM stays as the fallback for a script with no title of its own.
    if (script or {}).get("title"):
        meta = deterministic_metadata(script)
    else:
        print("    [youtube] Generating title & description...")
        meta = generate_metadata(topic, script)
    title = (script or {}).get("title") or meta["title"]
    description = meta["description"]
    tags = meta["tags"]

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
