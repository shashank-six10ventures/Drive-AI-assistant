"""
styles.py — All custom CSS for the Drive AI Assistant Streamlit app.

Imported once in streamlit_app.py via inject_styles().
Keeping CSS separate from app logic improves readability and makes
visual tweaks easy to find without wading through business logic.
"""
from __future__ import annotations

import streamlit as st

_APP_CSS = """
<style>
.dataset-toolbar {
    position: sticky;
    top: 0.5rem;
    z-index: 20;
    background: linear-gradient(135deg, #f7fbff 0%, #eef7f2 100%);
    border: 1px solid #d9e8dd;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    margin: 0.25rem 0 1rem 0;
    box-shadow: 0 8px 20px rgba(30, 60, 90, 0.08);
}
.dataset-toolbar-title {
    font-weight: 700;
    font-size: 1rem;
    color: #16324f;
    margin-bottom: 0.2rem;
}
.dataset-toolbar-meta {
    color: #486581;
    font-size: 0.92rem;
    line-height: 1.35;
}
.dataset-mini-nav {
    position: sticky;
    top: 5.4rem;
    z-index: 19;
    background: rgba(255,255,255,0.92);
    border: 1px solid #d8e4ef;
    border-radius: 12px;
    padding: 0.5rem 0.8rem;
    margin: 0 0 1rem 0;
    backdrop-filter: blur(6px);
}
.dataset-mini-nav summary {
    cursor: pointer;
    font-weight: 700;
    color: #17324d;
    margin-bottom: 0.25rem;
}
.dataset-mini-nav a {
    display: inline-block;
    margin: 0.2rem 0.6rem 0.2rem 0;
    color: #245c8a;
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 600;
}
.dataset-mini-nav a:hover {
    text-decoration: underline;
}
.benchmark-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.75rem;
    margin: 0.35rem 0 1rem 0;
}
.benchmark-card {
    border-radius: 14px;
    padding: 0.9rem 1rem;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 8px 18px rgba(20, 34, 51, 0.06);
}
.benchmark-card.good {
    background: linear-gradient(135deg, #f2fff7 0%, #e4f9ea 100%);
    border-color: #b8e3c4;
}
.benchmark-card.warn {
    background: linear-gradient(135deg, #fff6f2 0%, #fde7df 100%);
    border-color: #f0c4b4;
}
.benchmark-card.neutral {
    background: linear-gradient(135deg, #f8fbff 0%, #eef3fb 100%);
    border-color: #d8e2f1;
}
.benchmark-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #5a6c7d;
    margin-bottom: 0.25rem;
}
.benchmark-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #16212f;
    margin-bottom: 0.15rem;
}
.benchmark-delta {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1f3c56;
    margin-bottom: 0.25rem;
}
.benchmark-detail {
    font-size: 0.85rem;
    color: #52606d;
    line-height: 1.3;
}
</style>
"""


def inject_styles() -> None:
    """Inject all custom CSS into the Streamlit page. Call once at app startup."""
    st.markdown(_APP_CSS, unsafe_allow_html=True)
