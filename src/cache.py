"""
Local response cache for embeddings and LLM completions.

Goals:
- Cut cost & latency for repeated queries / re-ingestion.
- Fully OFFLINE and ZERO hard dependencies: the default backend is a plain
  file cache (one JSON file per key) under ``cache_dir``.
- Optional ``diskcache`` backend when installed and CACHE_BACKEND=diskcache.

Usage:
    from src.cache import default_cache
    val = default_cache().cached("embed", (texts, task), lambda: _compute())
"""
import hashlib
import json
import os
import threading

import config


class ResponseCache:
    """Content-addressed on-disk cache. Keys are SHA-256 of the name + parts."""

    def __init__(self, cache_dir: str | None = None, enabled: bool | None = None,
                 backend: str | None = None):
        s = config.get_settings()
        self.cache_dir = cache_dir or s.cache_dir
        self.enabled = enabled if enabled is not None else s.enable_cache
        self.backend = (backend or s.cache_backend).lower()
        self._lock = threading.Lock()
        self._dc = None
        if self.backend == "diskcache":
            try:
                import diskcache  # type: ignore

                self._dc = diskcache.Cache(self.cache_dir)
            except Exception:
                self.backend = "file"  # graceful fallback
        if self.enabled and self.backend == "file":
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
            except Exception:
                self.enabled = False

    # ---- hashing ----
    @staticmethod
    def _hash(*parts) -> str:
        h = hashlib.sha256()
        for p in parts:
            try:
                data = json.dumps(p, ensure_ascii=False, sort_keys=True).encode("utf-8")
            except Exception:
                data = repr(p).encode("utf-8")
            h.update(data)
        return h.hexdigest()[:40]

    def _file_path(self, name: str, key_parts) -> str:
        return os.path.join(self.cache_dir, f"{name}_{self._hash(name, *key_parts)}.json")

    # ---- get / set ----
    def get(self, name: str, key_parts) -> object | None:
        if not self.enabled:
            return None
        if self._dc is not None:
            try:
                return self._dc.get(self._hash(name, *key_parts))
            except Exception:
                return None
        path = self._file_path(name, key_parts)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("v")
        except Exception:
            return None

    def set(self, name: str, key_parts, value) -> None:
        if not self.enabled:
            return
        if self._dc is not None:
            try:
                self._dc.set(self._hash(name, *key_parts), value)
            except Exception:
                pass
            return
        path = self._file_path(name, key_parts)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"v": value}, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            pass

    def cached(self, name: str, key_parts, compute):
        """Return cached value if present, else compute, store, and return."""
        v = self.get(name, key_parts)
        if v is not None:
            return v
        v = compute()
        self.set(name, key_parts, v)
        return v

    def clear(self) -> None:
        if self._dc is not None:
            try:
                self._dc.clear()
            except Exception:
                pass
            return
        try:
            for fn in os.listdir(self.cache_dir):
                if fn.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, fn))
        except Exception:
            pass


_default_cache: ResponseCache | None = None


def default_cache() -> ResponseCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = ResponseCache()
    return _default_cache
