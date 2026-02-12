# app/style.py
import streamlit as st


def apply_dark_style():
    st.markdown(
        """
        <style>
        :root {
          --bg: #0b1220;
          --card: rgba(255,255,255,0.045);
          --card2: rgba(255,255,255,0.06);
          --border: rgba(255,255,255,0.08);
          --text: rgba(255,255,255,0.92);
          --muted: rgba(255,255,255,0.65);
          --accent: #ff4b4b;
          --glow: rgba(255, 75, 75, 0.18);
        }

        html, body, [data-testid="stAppViewContainer"] {
          background: radial-gradient(1200px 600px at 30% 10%, rgba(255,75,75,0.10), transparent 55%),
                      radial-gradient(900px 500px at 90% 20%, rgba(110,231,255,0.08), transparent 60%),
                      var(--bg) !important;
          color: var(--text);
        }

        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

        .card {
          background: linear-gradient(180deg, var(--card), var(--card2));
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 18px 18px;
          box-shadow: 0 10px 25px rgba(0,0,0,0.35);
        }
        .glow {
          box-shadow: 0 0 0 1px rgba(255,75,75,0.15),
                      0 12px 30px rgba(0,0,0,0.38),
                      0 0 36px var(--glow);
        }
        .muted { color: var(--muted); }
        .tiny { font-size: 0.82rem; color: var(--muted); }
        .divider {
          height: 1px;
          background: var(--border);
          margin: 14px 0;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
          background: rgba(10, 16, 28, 0.75) !important;
          border-right: 1px solid var(--border);
        }

        /* Buttons */
        .stButton>button {
          border-radius: 14px !important;
          border: 1px solid rgba(255,255,255,0.10) !important;
          padding: 0.65rem 1rem !important;
        }
        .stButton>button[kind="primary"] {
          background: var(--accent) !important;
          border: 1px solid rgba(255,75,75,0.40) !important;
        }

        /* Inputs */
        .stNumberInput input, .stSelectbox div, .stTextInput input {
          border-radius: 14px !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
          border-radius: 12px 12px 0 0;
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
          border-radius: 16px;
          overflow: hidden;
          border: 1px solid var(--border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
