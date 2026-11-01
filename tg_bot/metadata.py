import json
import os
import selectors
import signal
import subprocess
import time
from urllib.parse import urlsplit, parse_qs, urlencode
from urllib.request import Request, HTTPRedirectHandler, build_opener


class MetadataError(Exception):
    pass


def _fetch_with_ytdlp(url, timeout):
    args = [
        os.sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
        "--no-warnings", "--skip-download", "--dump-single-json",
        "--socket-timeout", "10", "--retries", "1", "--no-plugin-dirs", "--use-extractors",
        "TikTok,vm.tiktok,Instagram,instagram:reel,youtube", url,
    ]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        stdout = bytearray()
        deadline = time.monotonic() + timeout
        too_large = False
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MetadataError("metadata_timeout")
            for key, _ in selector.select(min(remaining, 0.5)):
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                elif key.data == "stdout":
                    if len(stdout) + len(data) > 512 * 1024:
                        too_large = True
                        raise MetadataError("metadata_too_large")
                    stdout.extend(data)
                # stderr is drained, never retained or shown.
        code = proc.wait(timeout=max(0.1, deadline-time.monotonic()))
        if too_large:
            raise MetadataError("metadata_too_large")
        if code:
            raise MetadataError("metadata_unavailable")
        data = json.loads(stdout)
        if data.get("_type") in {"playlist", "multi_video"} or data.get("is_live"):
            raise MetadataError("unsupported_media")
        from .links import supported_url
        final_url = str(data.get("webpage_url") or url)
        initial = supported_url(url)
        final = supported_url(final_url)
        final_parts = urlsplit(final_url)
        original_parts = urlsplit(url)
        original_id = original_parts.path.rstrip("/").rsplit("/", 1)[-1]
        canonical_watch = (
            initial is not None and initial[1] == "youtube" and
            final_parts.scheme == "https" and final_parts.hostname in
            {"youtube.com", "www.youtube.com", "m.youtube.com"} and
            final_parts.path == "/watch" and
            parse_qs(final_parts.query).get("v") == [original_id]
        )
        if not initial or not (final and final[1] == initial[1]) and not canonical_watch:
            raise MetadataError("redirect_unsupported")
        return {
            "id": str(data.get("id") or ""),
            "title": str(data.get("title") or "")[:300],
            "description": str(data.get("description") or "")[:1500],
            "url": final_url,
        }
    except MetadataError:
        if "proc" in locals() and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise
    except Exception:
        if "proc" in locals() and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise MetadataError("metadata_unavailable") from None


class _YoutubeOnlyRedirects(HTTPRedirectHandler):
    """Allow oEmbed redirects only when they remain on YouTube over HTTPS."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        source = urlsplit(req.full_url)
        target = urlsplit(newurl)
        if (source.scheme != "https" or source.hostname != "www.youtube.com" or
                target.scheme != "https" or target.hostname != "www.youtube.com" or
                target.username or target.password):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_youtube_oembed(short_url, timeout=10):
    from .links import supported_url

    checked = supported_url(short_url)
    if not checked or checked[1] != "youtube":
        raise MetadataError("metadata_unavailable")
    parts = urlsplit(checked[0])
    video_id = parts.path.rstrip("/").rsplit("/", 1)[-1]
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    endpoint = "https://www.youtube.com/oembed?" + urlencode({
        "url": watch_url,
        "format": "json",
    })
    request = Request(endpoint, headers={"Accept": "application/json"})
    try:
        opener = build_opener(_YoutubeOnlyRedirects())
        with opener.open(request, timeout=min(max(timeout, 1), 10)) as response:
            final = urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname != "www.youtube.com":
                raise MetadataError("metadata_unavailable")
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise MetadataError("metadata_unavailable")
            raw = response.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise MetadataError("metadata_unavailable")
        payload = json.loads(raw.decode("utf-8"))
        title = payload.get("title") if isinstance(payload, dict) else None
        if not isinstance(title, str) or not title.strip():
            raise MetadataError("metadata_unavailable")
        # YouTube oEmbed does not include a description. The regular yt-dlp
        # path remains the source for descriptions when it is available.
        return {
            "id": video_id,
            "title": title.strip()[:300],
            "description": "",
            "url": checked[0],
        }
    except MetadataError:
        raise
    except Exception:
        raise MetadataError("metadata_unavailable") from None


def fetch(url, timeout=30):
    from .links import supported_url

    checked = supported_url(url)
    try:
        return _fetch_with_ytdlp(url, timeout)
    except MetadataError as exc:
        # Only fall back for extractor failures/timeouts on a URL already
        # accepted as a YouTube Shorts link. Never pass arbitrary input to
        # oEmbed, and don't mask explicit redirect/media validation failures.
        if (checked and checked[1] == "youtube" and
                str(exc) in {"metadata_unavailable", "metadata_timeout"}):
            return _fetch_youtube_oembed(checked[0], timeout=min(timeout, 10))
        raise
