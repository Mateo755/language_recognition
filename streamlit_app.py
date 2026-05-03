"""Streamlit UI for language detection via the FastAPI HTTP API."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx
import streamlit as st

REMOTE_API_ENV = "LANGUAGE_DETECTOR_API_URL"
LOCAL_API_ENV = "LANGUAGE_DETECTOR_LOCAL_API_URL"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000"
HTTP_TIMEOUT_SECONDS = 30.0

BACKEND_MODE_REMOTE = "Zdalny"
BACKEND_MODE_LOCAL = "Lokalny (localhost)"


def _normalize_base_url(url: str) -> str:
    """Strip whitespace and trailing slashes so path joins behave consistently."""
    return url.strip().rstrip("/")


def _local_api_base_url() -> str:
    """Base URL for the local FastAPI instance (override via env)."""
    override = os.environ.get(LOCAL_API_ENV, "").strip()
    if override:
        return _normalize_base_url(override)
    return _normalize_base_url(DEFAULT_LOCAL_BASE_URL)


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
        return False, "Podaj adres bazowy API."

    endpoint = f"{base_url}/health"
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(endpoint)
    except httpx.TimeoutException:
        return False, "Przekroczono limit czasu połączenia (timeout)."
    except httpx.RequestError as exc:
        return False, f"Błąd połączenia: {exc!s}"

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
        return False, None, "Podaj adres bazowy API."

    endpoint = f"{base_url}/detect"
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json={"text": text})
    except httpx.TimeoutException:
        return False, None, "Przekroczono limit czasu połączenia (timeout)."
    except httpx.RequestError as exc:
        return False, None, f"Błąd połączenia: {exc!s}"

    if response.status_code == 200:
        try:
            return True, response.json(), ""
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            return False, None, f"Niepoprawna odpowiedź JSON: {exc}"

    return False, None, _format_http_error_detail(response)


@st.dialog("Adres zdalnego API")
def _remote_api_url_dialog() -> None:
    """Modal to configure the remote base URL for this Streamlit session only."""
    st.caption(
        "Wartość startowa pochodzi ze zmiennej LANGUAGE_DETECTOR_API_URL. "
        "Zapis zmienia tylko bieżącą sesję aplikacji."
    )
    with st.form("remote_api_url_dialog_form"):
        new_url = st.text_input(
            "Adres bazowy (HTTPS/HTTP)",
            value=st.session_state.remote_api_base,
        )
        submitted = st.form_submit_button("Zapisz", type="primary")
    if submitted:
        st.session_state.remote_api_base = new_url.strip()
        st.rerun()


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title="Wykrywanie języka", layout="centered")
    st.title("Wykrywanie języka")
    st.caption("Tekst jest wysyłany do wybranego backendu FastAPI (/detect); model nie działa w tej aplikacji.")

    if "remote_api_base" not in st.session_state:
        st.session_state.remote_api_base = os.environ.get(REMOTE_API_ENV, "")

    with st.sidebar:
        st.subheader("Backend")
        backend_mode = st.radio(
            "Tryb backendu",
            options=[BACKEND_MODE_REMOTE, BACKEND_MODE_LOCAL],
            horizontal=False,
        )
        is_remote = backend_mode == BACKEND_MODE_REMOTE

        if is_remote and st.button("Ustaw / zmień adres API", type="secondary"):
            _remote_api_url_dialog()

        base_url = _effective_base_url(is_remote, st.session_state.remote_api_base)
        st.markdown("**Aktualny adres API**")
        st.code(base_url or "(nie ustawiono)", language=None)

        if st.button("Sprawdź health", type="secondary"):
            ok, payload = _request_health(base_url)
            if ok:
                st.success("Połączenie OK.")
                st.write(payload)
            else:
                st.error(payload)

    with st.container():
        user_text = st.text_area(
            "Tekst do analizy",
            placeholder="Wklej lub wpisz tekst…",
            height=200,
        )
        detect_clicked = st.button("Wykryj język", type="primary")

    if detect_clicked:
        trimmed = user_text.strip()
        if not trimmed:
            st.warning("Podaj niepusty tekst.")

        elif is_remote and not st.session_state.remote_api_base.strip():
            st.error(
                "W trybie zdalnym ustaw adres API: sidebar → „Ustaw / zmień adres API” "
                "(lub zmienna LANGUAGE_DETECTOR_API_URL przy starcie)."
            )

        elif trimmed:
            ok, payload, error_message = _request_detect(base_url, trimmed)

            if ok and payload:
                language = payload.get("language", "")
                st.success(f"**Wykryty język:** {language}")
                with st.expander("Szczegóły odpowiedzi"):
                    st.json(payload)
            else:
                st.error(error_message)
                if is_remote:
                    st.warning(
                        "Zdalny backend może być niedostępny. "
                        "Spróbuj przełączyć na „Lokalny (localhost)” i uruchomić FastAPI "
                        "(np. uvicorn jak w README), z modelem pod app/model/."
                    )


if __name__ == "__main__":
    main()
