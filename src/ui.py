import streamlit as st


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

        html, body, [class*='css'] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 20% 20%, #f6f9ff 0%, #eef2ff 35%, #ffffff 100%);
        }

        [data-testid='stSidebar'] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }

        [data-testid='stSidebar'] * {
            color: #e2e8f0 !important;
        }

        .page-title {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.9rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
            color: #0f172a;
        }

        .page-subtitle {
            color: #475569;
            margin-bottom: 1rem;
        }

        .kpi-card {
            background: white;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .kpi-label {
            font-size: 0.8rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .kpi-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.3rem;
        }

        .badge-good {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            background: #dcfce7;
            color: #166534;
            font-weight: 700;
        }

        .badge-risk {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            background: #fee2e2;
            color: #991b1b;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def title_block(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def kpi_card(label: str, value: str) -> None:
    st.markdown(
        (
            "<div class='kpi-card'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
