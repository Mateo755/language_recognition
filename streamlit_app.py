"""Streamlit UI for language detection via the FastAPI HTTP API."""

from __future__ import annotations

import html
import json
import os
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import streamlit as st

REMOTE_API_ENV = "LANGUAGE_DETECTOR_API_URL"
LOCAL_API_ENV = "LANGUAGE_DETECTOR_LOCAL_API_URL"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000"
HTTP_TIMEOUT_SECONDS = 30.0

BACKEND_MODE_REMOTE = "Remote"
BACKEND_MODE_LOCAL = "Local (localhost)"


def _normalize_base_url(url: str) -> str:
    """Strip whitespace and trailing slashes so path joins behave consistently."""
    return url.strip().rstrip("/")


def _local_api_base_url() -> str:
    """Base URL for the local FastAPI instance (override via env)."""
    override = os.environ.get(LOCAL_API_ENV, "").strip()
    if override:
        return _normalize_base_url(override)
    return _normalize_base_url(DEFAULT_LOCAL_BASE_URL)


def _sidebar_wrapped_url_text(display_text: str) -> None:
    """Single block: monospace URL with line wrapping for narrow sidebars."""
    safe = html.escape(display_text, quote=True)
    st.markdown(
        (
            '<div style="word-break:break-all;overflow-wrap:anywhere;white-space:pre-wrap;'
            'font-family:ui-monospace,Consolas,monospace;font-size:0.8rem;line-height:1.5;'
            "padding:0.5rem 0.55rem;border-radius:0.35rem;"
            'border:solid 1px rgba(127,127,127,0.35);background:rgba(127,127,127,0.07);">'
            f"{safe}</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_endpoint_info_content(is_remote: bool, effective_base_url: str) -> None:
    """Protocol, host, full URL and /docs link; used inside bordered container or popover."""
    unset = not effective_base_url.strip()

    title = "Active endpoint · remote" if is_remote else "Active endpoint · local"
    st.markdown(f"**{title}**")

    if unset:
        st.warning(
            "Set the URL via **Set / change API URL** or the environment variable.",
            icon="⚠️",
        )
        _sidebar_wrapped_url_text("(not configured)")
        return

    parsed = urlparse(effective_base_url)
    has_norm = parsed.scheme in ("http", "https") and parsed.netloc

    if has_norm:
        c_proto, c_host = st.columns(2)
        with c_proto:
            st.caption("Protocol")
            st.markdown(f"**{parsed.scheme.upper()}**")
        with c_host:
            st.caption("Host")
            safe_host = html.escape(parsed.netloc, quote=True)
            st.markdown(
                f'<p style="margin:0;word-break:break-all;font-size:0.95rem;">{safe_host}</p>',
                unsafe_allow_html=True,
            )
        st.divider()

    st.caption("Full base URL (select to copy)")
    _sidebar_wrapped_url_text(effective_base_url)

    if has_norm:
        docs_href = effective_base_url.rstrip("/") + "/docs"
        st.link_button(
            "Open API docs (/docs)",
            docs_href,
            use_container_width=True,
            help="Open FastAPI Swagger in a new browser tab.",
        )


def _effective_base_url(is_remote: bool, remote_url: str) -> str:
    if is_remote:
        return _normalize_base_url(remote_url)
    return _local_api_base_url()


def _format_http_error_detail(response: httpx.Response) -> str:
    """Best-effort message from FastAPI / generic error bodies."""
    try:
        payload: Any = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, list):
            return json.dumps(detail, ensure_ascii=False)
        if detail is not None:
            return str(detail)
        return json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        text = response.text.strip()
        return text if text else f"HTTP {response.status_code}"


def _request_health(base_url: str) -> tuple[bool, str]:
    if not base_url:
        return False, "Enter the API base URL."

    endpoint = f"{base_url}/health"
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(endpoint)
    except httpx.TimeoutException:
        return False, "Request timed out."
    except httpx.RequestError as exc:
        return False, f"Connection error: {exc!s}"

    if response.status_code == 200:
        try:
            data = response.json()
            return True, json.dumps(data, ensure_ascii=False)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return True, response.text.strip() or "{}"

    return False, _format_http_error_detail(response)


def _request_detect(base_url: str, text: str) -> tuple[bool, Optional[dict[str, Any]], str]:
    """Call POST /detect. Returns (ok, json_body_if_ok, error_message)."""
    if not base_url:
        return False, None, "Enter the API base URL."

    endpoint = f"{base_url}/detect"
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json={"text": text})
    except httpx.TimeoutException:
        return False, None, "Request timed out."
    except httpx.RequestError as exc:
        return False, None, f"Connection error: {exc!s}"

    if response.status_code == 200:
        try:
            return True, response.json(), ""
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            return False, None, f"Invalid JSON response: {exc}"

    return False, None, _format_http_error_detail(response)


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title="Language detection", layout="centered")
    st.title("Language detection")
    st.caption(
        "Text is sent to the selected FastAPI backend (/detect). This app does not load the model."
    )

    if "remote_api_base" not in st.session_state:
        st.session_state.remote_api_base = os.environ.get(REMOTE_API_ENV, "")

    with st.sidebar:
        st.subheader("Backend")
        backend_mode = st.radio(
            "Backend mode",
            options=[BACKEND_MODE_REMOTE, BACKEND_MODE_LOCAL],
            horizontal=False,
        )
        is_remote = backend_mode == BACKEND_MODE_REMOTE

        if is_remote:
            with st.popover("Set / change API URL", use_container_width=True):
                st.caption(
                    "Initial value from **LANGUAGE_DETECTOR_API_URL**. Saving applies to this session only; "
                    "the panel closes after **Save**."
                )
                with st.form("remote_api_url_popover_form"):
                    new_url = st.text_input(
                        "Base URL (HTTPS/HTTP)",
                        value=st.session_state.remote_api_base,
                    )
                    submitted = st.form_submit_button("Save", type="primary")
                if submitted:
                    st.session_state.remote_api_base = new_url.strip()
                    st.rerun()
        base_url = _effective_base_url(is_remote, st.session_state.remote_api_base)
        with st.popover("Info endpoint", use_container_width=True):
            st.caption(
                "Base URL used this session for **/health** and **/detect**, plus a shortcut to **/docs**."
            )
            with st.container(border=True):
                _render_endpoint_info_content(is_remote, base_url)

        if st.button("Check health", type="secondary"):
            ok, payload = _request_health(base_url)
            if ok:
                st.success("Connection OK.")
                st.write(payload)
            else:
                st.error(payload)

    with st.container():
        user_text = st.text_area(
            "Text to analyse",
            placeholder="Paste or type text…",
            height=200,
        )
        detect_clicked = st.button("Detect language", type="primary")

    if detect_clicked:
        trimmed = user_text.strip()
        if not trimmed:
            st.warning("Enter non-empty text.")

        elif is_remote and not st.session_state.remote_api_base.strip():
            st.error(
                "In **Remote** mode, set the API URL via **Set / change API URL** in the sidebar, "
                "or **LANGUAGE_DETECTOR_API_URL** before starting the app."
            )

        elif trimmed:
            ok, payload, error_message = _request_detect(base_url, trimmed)

            if ok and payload:
                language = payload.get("language", "")
                st.success(f"**Detected language:** {language}")
                with st.expander("Response details"):
                    st.json(payload)
            else:
                st.error(error_message)
                if is_remote:
                    st.warning(
                        "The remote backend may be unreachable. "
                        "Try **Local (localhost)** and run FastAPI locally "
                        "(see README), with **`app/model/trained_pipeline-0.1.0.pkl`** in place."
                    )


if __name__ == "__main__":
    main()
