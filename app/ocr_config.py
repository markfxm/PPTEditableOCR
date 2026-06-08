from __future__ import annotations


def resolve_ocr_config(
    selected_backend: str,
    token: str | None,
    local_backend: str,
    remote_backend: str,
    token_length: int,
) -> tuple[str, str | None, str | None]:
    token = (token or "").strip()
    if selected_backend == remote_backend and len(token) != token_length:
        return local_backend, None, "远端 OCR 令牌未设置或长度不正确，可继续使用本地 OCR。"
    return selected_backend, token or None, None


def should_clear_paused_ocr_request(force_local_ocr: bool) -> bool:
    return not force_local_ocr
