import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

URL_RE = re.compile(r"https?://[^\s<>\]\[()]+", re.I)
TRACKING = {"si", "feature", "share", "igsh", "igshid", "_t", "_r", "utm_source", "utm_medium", "utm_campaign"}

def supported_url(url: str) -> tuple[str, str] | None:
    try:
        p = urlsplit(url.strip().rstrip(".,!?;:"))
        if p.scheme != "https" or p.username or p.password or not p.hostname: return None
        host = p.hostname.lower()
        path = p.path
        if host in {"vm.tiktok.com", "vt.tiktok.com"}:
            return (urlunsplit(("https", host, path, "", "")), "tiktok") if path.strip("/") else None
        if host in {"tiktok.com", "www.tiktok.com"} and re.fullmatch(r"/@[^/]+/video/\d+/?", path): return (_clean(p), "tiktok")
        if host in {"instagram.com", "www.instagram.com"} and re.fullmatch(r"/(?:reel|reels)/[A-Za-z0-9_-]+/?", path): return (_clean(p), "instagram")
        if host in {"youtube.com", "www.youtube.com", "m.youtube.com"} and re.fullmatch(r"/shorts/[A-Za-z0-9_-]+/?", path): return (_clean(p), "youtube")
    except Exception: pass
    return None

def _clean(p):
    q = [(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in TRACKING]
    return urlunsplit(("https", p.hostname.lower(), p.path, urlencode(q), ""))

def extract_urls(text: str, entities=()) -> list[str]:
    urls = [m.group(0) for m in URL_RE.finditer(text or "")]
    # Telegram text-url entities carry a URL independently of visible text.
    urls += [getattr(e, "url", "") for e in entities if getattr(e, "url", None)]
    out=[]; seen=set()
    for u in urls:
        v=supported_url(u)
        if v and v[0] not in seen: seen.add(v[0]); out.append(v[0])
    return out


def has_url_candidate(text: str, entities=()) -> bool:
    """Report whether a message contains an HTTP URL, supported or not."""
    return bool(URL_RE.search(text or "")) or any(
        getattr(entity, "url", None) for entity in entities
    )
