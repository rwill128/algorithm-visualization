from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import mimetypes
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
VALID_PRIVACY_STATUSES = {"private", "unlisted", "public"}
DEFAULT_QUEUE_PATH = Path("data/youtube_upload_queue.json")
DEFAULT_TOKEN_PATH = Path("secrets/youtube_token.json")


@dataclass(frozen=True)
class UploadManifest:
    video_path: Path
    title: str
    description: str
    tags: list[str]
    privacy_status: str
    category_id: str
    made_for_kids: bool
    thumbnail_path: Path | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload approved videos from the curated YouTube queue.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Show queue state.")
    add_common_args(list_parser)

    upload_parser = subparsers.add_parser("upload-next", help="Upload the next approved queue item(s).")
    add_common_args(upload_parser)
    upload_parser.add_argument("--limit", type=int, default=1, help="Maximum uploads in this run.")
    upload_parser.add_argument("--privacy", choices=sorted(VALID_PRIVACY_STATUSES), default=None)
    upload_parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN_PATH)
    upload_parser.add_argument("--approved", action="store_true", help="Required for any upload.")
    upload_parser.add_argument("--allow-public", action="store_true", help="Required when uploading public videos.")
    upload_parser.add_argument("--dry-run", action="store_true", help="Show what would upload without calling YouTube.")

    args = parser.parse_args(argv)
    if args.command == "list":
        return list_queue(args.queue)
    if args.command == "upload-next":
        return upload_next(
            queue_path=args.queue,
            token_path=args.token,
            limit=args.limit,
            privacy_override=args.privacy,
            approved=args.approved,
            allow_public=args.allow_public,
            dry_run=args.dry_run,
        )
    raise AssertionError(args.command)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE_PATH)


def list_queue(queue_path: Path) -> int:
    queue = read_queue(queue_path)
    for index, item in enumerate(queue.get("items", []), start=1):
        state = item_state(item)
        approved = "approved" if item.get("approved_for_upload") else "not approved"
        enabled = "enabled" if item.get("enabled", True) else "disabled"
        print(f"{index:02d}. {item.get('id', '<missing-id>')}: {state}, {approved}, {enabled}")
        if item.get("youtube_url"):
            print(f"    {item['youtube_url']}")
    return 0


def upload_next(
    queue_path: Path,
    token_path: Path,
    limit: int,
    privacy_override: str | None,
    approved: bool,
    allow_public: bool,
    dry_run: bool,
) -> int:
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    if not approved and not dry_run:
        raise PermissionError("Refusing to upload without --approved.")

    with locked_queue(queue_path) as queue:
        candidates = list(next_upload_candidates(queue, limit))
        if not candidates:
            print("No approved pending videos are available.")
            return 0

        if dry_run:
            for _index, item, manifest in candidates:
                print(f"Would upload {item['id']}: {manifest.title} ({manifest.video_path})")
            return 0

        credentials = load_credentials(token_path, scopes=(YOUTUBE_UPLOAD_SCOPE,))
        youtube = build_youtube_client(credentials)
        uploaded = 0
        for index, item, manifest in candidates:
            if privacy_override:
                manifest = replace_privacy(manifest, privacy_override)
            if manifest.privacy_status == "public" and not allow_public:
                raise PermissionError("Refusing public upload without --allow-public.")

            try:
                video_id = upload_video(youtube, manifest)
            except Exception as exc:
                mark_failed(queue, index, exc)
                print(f"FAILED {item['id']}: {exc}", file=sys.stderr)
                raise

            mark_uploaded(queue, index, video_id, manifest.privacy_status)
            if manifest.thumbnail_path:
                try:
                    set_thumbnail(youtube, video_id, manifest.thumbnail_path)
                except Exception as exc:
                    mark_thumbnail_failed(queue, index, exc)
                    print(f"THUMBNAIL_FAILED {item['id']}: {exc}", file=sys.stderr)
            uploaded += 1
            print(f"Uploaded {item['id']}: https://youtu.be/{video_id}")

        print(f"Uploaded {uploaded} video(s).")
        return 0


@contextlib.contextmanager
def locked_queue(queue_path: Path) -> Any:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        queue = read_queue(queue_path)
        try:
            yield queue
        finally:
            write_queue(queue_path, queue)


def read_queue(queue_path: Path) -> dict[str, Any]:
    return json.loads(queue_path.read_text())


def write_queue(queue_path: Path, queue: dict[str, Any]) -> None:
    tmp_path = queue_path.with_suffix(queue_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n")
    tmp_path.replace(queue_path)


def next_upload_candidates(queue: dict[str, Any], limit: int) -> Iterable[tuple[int, dict[str, Any], UploadManifest]]:
    defaults = queue.get("defaults") or {}
    selected = 0
    for index, item in enumerate(queue.get("items", [])):
        if selected >= limit:
            return
        if item_state(item) != "pending":
            continue
        if not item.get("enabled", True):
            continue
        if not item.get("approved_for_upload", False):
            continue
        manifest = build_manifest(defaults, item)
        validate_manifest(manifest)
        yield index, item, manifest
        selected += 1


def item_state(item: dict[str, Any]) -> str:
    if item.get("youtube_video_id") or item.get("youtube_url") or item.get("uploaded_at"):
        return "uploaded"
    if item.get("last_error"):
        return "failed"
    return "pending"


def build_manifest(defaults: dict[str, Any], item: dict[str, Any]) -> UploadManifest:
    video_path = Path(str(item["video_path"]))
    thumbnail_raw = item.get("thumbnail_path")
    tags = [str(tag) for tag in defaults.get("tags", [])]
    tags.extend(str(tag) for tag in item.get("tags", []))
    tags = dedupe_preserving_order(tags)

    title_hashtags = collect_hashtags(defaults, item, "title_hashtags")
    title = append_hashtags(str(item["title"]), title_hashtags)

    description = str(item.get("description", ""))
    suffix = str(defaults.get("description_suffix", ""))
    if suffix and suffix.strip() not in description:
        description = description.rstrip() + suffix
    description_hashtags = collect_hashtags(defaults, item, "description_hashtags")
    description = append_hashtags(description, description_hashtags, separator="\n\n")

    return UploadManifest(
        video_path=video_path,
        title=title,
        description=description,
        tags=tags,
        privacy_status=str(item.get("privacy_status") or defaults.get("privacy_status", "private")),
        category_id=str(item.get("category_id") or defaults.get("category_id", "27")),
        made_for_kids=bool(item.get("made_for_kids", defaults.get("made_for_kids", False))),
        thumbnail_path=Path(str(thumbnail_raw)) if thumbnail_raw else None,
    )


def collect_hashtags(defaults: dict[str, Any], item: dict[str, Any], key: str) -> list[str]:
    values = [str(value) for value in defaults.get(key, [])]
    values.extend(str(value) for value in item.get(key, []))
    return dedupe_preserving_order(normalize_hashtag(value) for value in values)


def normalize_hashtag(value: str) -> str:
    cleaned = "".join(character for character in value.strip() if character.isalnum() or character == "_")
    return f"#{cleaned}" if cleaned else ""


def append_hashtags(text: str, hashtags: list[str], separator: str = " ") -> str:
    missing = [hashtag for hashtag in hashtags if hashtag and hashtag.casefold() not in text.casefold()]
    if not missing:
        return text
    return text.rstrip() + separator + " ".join(missing)


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def validate_manifest(manifest: UploadManifest) -> None:
    if not manifest.video_path.exists():
        raise FileNotFoundError(f"Missing video: {manifest.video_path}")
    if manifest.thumbnail_path and not manifest.thumbnail_path.exists():
        raise FileNotFoundError(f"Missing thumbnail: {manifest.thumbnail_path}")
    if not manifest.title.strip():
        raise ValueError("Video title is required.")
    if len(manifest.title) > 100:
        raise ValueError(f"YouTube title is too long: {len(manifest.title)} characters.")
    if manifest.privacy_status not in VALID_PRIVACY_STATUSES:
        raise ValueError(f"Invalid privacy status: {manifest.privacy_status}")


def replace_privacy(manifest: UploadManifest, privacy_status: str) -> UploadManifest:
    return UploadManifest(
        video_path=manifest.video_path,
        title=manifest.title,
        description=manifest.description,
        tags=manifest.tags,
        privacy_status=privacy_status,
        category_id=manifest.category_id,
        made_for_kids=manifest.made_for_kids,
        thumbnail_path=manifest.thumbnail_path,
    )


def mark_uploaded(queue: dict[str, Any], index: int, video_id: str, privacy_status: str) -> None:
    item = queue["items"][index]
    item["uploaded_at"] = datetime.now(timezone.utc).isoformat()
    item["youtube_video_id"] = video_id
    item["youtube_url"] = f"https://youtu.be/{video_id}"
    item["uploaded_privacy_status"] = privacy_status
    item.pop("last_error", None)
    item.pop("thumbnail_error", None)


def mark_failed(queue: dict[str, Any], index: int, exc: Exception) -> None:
    queue["items"][index]["last_error"] = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "type": type(exc).__name__,
        "message": str(exc),
    }


def mark_thumbnail_failed(queue: dict[str, Any], index: int, exc: Exception) -> None:
    queue["items"][index]["thumbnail_error"] = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "type": type(exc).__name__,
        "message": str(exc),
    }


def load_credentials(token_path: Path, scopes: Iterable[str]) -> Any:
    if not token_path.exists():
        raise FileNotFoundError(f"Missing YouTube OAuth token: {token_path}")
    require_youtube_dependencies()
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    credentials = Credentials.from_authorized_user_file(str(token_path), scopes=list(scopes))
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        token_path.write_text(credentials.to_json())
    if not credentials.valid:
        raise RuntimeError("YouTube OAuth token is invalid. Reauthorize the channel token.")
    return credentials


def build_youtube_client(credentials: Any) -> Any:
    require_youtube_dependencies()
    from googleapiclient.discovery import build

    return build("youtube", "v3", credentials=credentials)


def upload_video(youtube: Any, manifest: UploadManifest) -> str:
    require_youtube_dependencies()
    from googleapiclient.http import MediaFileUpload

    body = {
        "snippet": {
            "title": manifest.title,
            "description": manifest.description,
            "tags": manifest.tags,
            "categoryId": manifest.category_id,
        },
        "status": {
            "privacyStatus": manifest.privacy_status,
            "selfDeclaredMadeForKids": manifest.made_for_kids,
        },
    }
    media = MediaFileUpload(str(manifest.video_path), mimetype="video/*", chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = execute_resumable_upload(request)
    video_id = response.get("id")
    if not video_id:
        raise RuntimeError(f"YouTube upload response did not contain a video id: {response}")
    return str(video_id)


def set_thumbnail(youtube: Any, video_id: str, thumbnail_path: Path) -> None:
    require_youtube_dependencies()
    from googleapiclient.http import MediaFileUpload

    mimetype, _encoding = mimetypes.guess_type(thumbnail_path)
    if mimetype not in {"image/jpeg", "image/png"}:
        raise ValueError(f"Unsupported YouTube thumbnail type for {thumbnail_path}: {mimetype}")
    media = MediaFileUpload(str(thumbnail_path), mimetype=mimetype, resumable=True)
    youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


def execute_resumable_upload(request: Any) -> dict[str, Any]:
    response = None
    while response is None:
        _status, response = request.next_chunk()
    return response


def require_youtube_dependencies() -> None:
    try:
        import google_auth_oauthlib  # noqa: F401
        import googleapiclient  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("YouTube dependencies are missing. Run: .venv/bin/pip install -r requirements.txt") from exc


if __name__ == "__main__":
    raise SystemExit(main())
