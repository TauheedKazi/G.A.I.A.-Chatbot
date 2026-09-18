import base64
import html
import json
import os
import re
from giaa_engine import query_giaa
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Darukaa.Earth — G.A.I.A",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# 2. Base64 Background Helper
def get_base64_image(image_path):
  if os.path.exists(image_path):
    with open(image_path, "rb") as img_file:
      return base64.b64encode(img_file.read()).decode()
  return ""


# 2b. Lightweight Markdown -> HTML for chat bubbles
#
# The AI backend replies using Markdown (**bold**, *italics*, ### headings,
# - bullet points, etc.), but the bubbles are built as raw HTML strings, so
# without this those literal characters ("**", "###", "-") were showing up
# in the chat instead of being rendered as real formatting. Text is
# HTML-escaped FIRST (so any HTML the model happens to output can't be
# injected/executed), then a small set of Markdown patterns are converted
# to real tags on top of the escaped text — line by line, since headings
# and bullets are line-level constructs.
def _format_inline(escaped_text):
  # **bold** or __bold__
  escaped_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped_text, flags=re.DOTALL)
  escaped_text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", escaped_text, flags=re.DOTALL)

  # *italic* or _italic_ (single markers, not already-converted bold tags)
  escaped_text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", escaped_text, flags=re.DOTALL)
  escaped_text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<em>\1</em>", escaped_text, flags=re.DOTALL)

  # `inline code`
  escaped_text = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped_text)
  return escaped_text


def markdown_lite_to_html(text):
  lines = text.split("\n")
  parts = []
  prev_was_block = False  # heading/bullet lines already sit on their own line

  for i, raw_line in enumerate(lines):
    escaped_line = html.escape(raw_line)

    heading_match = re.match(r"^(#{1,6})\s+(.*)$", escaped_line)
    bullet_match = re.match(r"^[-*]\s+(.*)$", escaped_line) if not heading_match else None

    if heading_match:
      level = min(len(heading_match.group(1)), 4)  # cap visual size at h4-ish
      content = _format_inline(heading_match.group(2))
      parts.append(f'<div class="gaia-heading gaia-h{level}">{content}</div>')
      prev_was_block = True
      continue

    if bullet_match:
      content = _format_inline(bullet_match.group(1))
      parts.append(f'<div class="gaia-bullet">• {content}</div>')
      prev_was_block = True
      continue

    if i > 0 and not prev_was_block:
      parts.append("<br>")
    parts.append(_format_inline(escaped_line))
    prev_was_block = False

  return "".join(parts)


# 2c. Structured (JSON) input support
#
# The hackathon brief requires BOTH free text (already handled by
# st.chat_input) AND "Structured input (JSON or similar)" as a distinct,
# mandatory input mode — not just a user typing JSON by hand into the chat
# box. STRUCTURED_FIELDS defines the environmental metrics the brief's own
# example use case asks for (SOC %, rainfall, crop/land use, region) plus
# the bonus geo-coordinates field, so a person can either fill in a form or
# paste raw JSON, and both paths converge on the same normalized JSON
# payload sent to query_giaa.
STRUCTURED_FIELDS = [
    "soil_organic_carbon_percent",
    "annual_rainfall_mm",
    "soil_ph",
    "land_use_or_crop",
    "region",
    "latitude",
    "longitude",
]


def build_structured_message(payload):
  """Turn a dict of structured metrics into the JSON user-message we send
  to query_giaa. Drops empty/None values so the model isn't told a field
  is "0" or "" when the person simply left it blank."""
  # Drop only truly blank fields (None or empty string) — not 0, since 0 is
  # a legitimate value for some of these metrics (e.g. rainfall_mm during a
  # drought). Numeric fields below use value=None so an untouched field
  # comes through here as None, distinguishing "left blank" from "entered 0".
  clean_payload = {k: v for k, v in payload.items() if v not in (None, "")}
  json_body = json.dumps(clean_payload, indent=2)
  return f"Structured input (submitted as JSON):\n{json_body}"


# NOTE: the original hardcoded Windows path ("d:\AI Environmentalist...")
# only exists on one machine and will silently fail everywhere else (the
# helper above just returns "" and the background never appears). Point
# this at a file that ships with the app instead, e.g. an "assets" folder
# next to app.py, so it works wherever the app is deployed.
BACKGROUND_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "streamlit_website_background.png",
)
img_base64 = get_base64_image(BACKGROUND_IMAGE_PATH)

# Fall back to a plain green gradient (rather than a blank white page) when
# the background image file isn't present yet, and surface a visible
# on-page hint so it's obvious why the photo isn't showing.
if img_base64:
    background_css = f'url("data:image/jpeg;base64,{img_base64}")'
else:
    background_css = "linear-gradient(160deg, #dfe9e4 0%, #c3d3cb 100%)"
    st.info(
        "🖼️ Background photo not found. Save it as "
        f"`{BACKGROUND_IMAGE_PATH}` (an `assets` folder next to `app.py`) "
        "and reload the page.",
        icon="ℹ️",
    )

# 3. G.A.I.A. UI
#
# Use a keyed Streamlit container for the card itself. This gives us a stable
# CSS hook (.st-key-gaia-card) instead of relying on fragile column selectors.
custom_css = f"""
<style>
/* ===== WEBSITE ===== */
.stApp {{
    background-image: {background_css};
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

.stAppViewContainer,
.stAppViewContainer > .main,
.main,
.main .block-container {{
    background: transparent !important;
}}

.main .block-container {{
    padding: 0 !important;
    max-width: none !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* ===== G.A.I.A. CARD ===== */
.st-key-gaia-card {{
    position: fixed !important;
    right: 18px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;

    width: 330px !important;
    box-sizing: border-box !important;

    z-index: 999998 !important;

    background: #ffffff !important;
    background-color: #ffffff !important;
    background-image: none !important;

    border: 1px solid #d9d9d9 !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 26px rgba(0,0,0,.24) !important;

    padding: 14px !important;

    /* This element is `position: fixed`, which takes it out of normal
       document flow — so Streamlit's own height/scroll wrapper (an
       ancestor) has NO effect on it; an ancestor's overflow/height never
       clips a fixed-positioned descendant. We therefore give the card its
       own fixed height and scroll here directly, instead of relying on
       st.container(height=...). This is what actually makes long AI
       replies and the chat input reachable by scrolling. */
    height: 480px !important;
    max-height: 480px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}}

/* Every wrapper inside the card is also opaque */
.st-key-gaia-card > div,
.st-key-gaia-card [data-testid="stVerticalBlock"],
.st-key-gaia-card [data-testid="stVerticalBlockBorderWrapper"] {{
    background: #ffffff !important;
    background-color: #ffffff !important;
    background-image: none !important;
}}

/* ===== HEADER ===== */
.gaia-header-title h3 {{
    color: #123d2e !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    margin: 0 !important;
}}

.gaia-header-subtitle {{
    color: #8a8f93;
    font-size: 10px;
    letter-spacing: .02em;
    margin-top: 2px;
}}

/* Minimize ("–") control, styled as its own button so it matches the
   flat, borderless look in the mockup rather than a default Streamlit
   button. */
.st-key-gaia_minimize button {{
    background: transparent !important;
    border: none !important;
    color: #9aa0a4 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    padding: 0 4px !important;
    min-height: 0 !important;
    box-shadow: none !important;
}}

.st-key-gaia_minimize button:hover {{
    color: #123d2e !important;
}}

.st-key-gaia-card hr {{
    border: 0 !important;
    border-top: 1px solid #eeeeee !important;
    margin: 7px 0 10px !important;
}}

/* ===== MESSAGE BUBBLES =====
   These are explicit HTML bubbles, so they do not depend on
   Streamlit's stChatMessage styling at all. */
.gaia-message {{
    display: flex;
    align-items: flex-start;
    gap: 8px;

    width: 100%;
    box-sizing: border-box;

    margin: 0 0 10px 0;
}}

.gaia-message.gaia-user {{
    flex-direction: row-reverse;
}}

.gaia-message .gaia-icon {{
    flex: 0 0 auto;
    width: 22px;
    height: 22px;
    border-radius: 6px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #f3a53f;
    color: #ffffff;
    font-size: 12px;
}}

.gaia-message.gaia-user .gaia-icon {{
    background: #123d2e;
}}

.gaia-message .gaia-bubble {{
    background: #f3f5f6 !important;
    background-color: #f3f5f6 !important;

    border: 1px solid #e1e5e8;
    border-radius: 11px;

    padding: 9px 10px;

    font-size: 12px;
    line-height: 1.45;
    color: #263238 !important;

    overflow-wrap: anywhere;
    max-width: calc(100% - 30px);
}}

.gaia-message .gaia-bubble strong {{
    color: #123d2e !important;
    font-weight: 700 !important;
}}

.gaia-message .gaia-bubble code {{
    background: #e6e9eb;
    border-radius: 4px;
    padding: 1px 4px;
    font-size: 11px;
}}

.gaia-message .gaia-bubble .gaia-heading {{
    color: #123d2e !important;
    font-weight: 700 !important;
    margin: 6px 0 3px 0;
}}

.gaia-message .gaia-bubble .gaia-heading:first-child {{
    margin-top: 0;
}}

.gaia-message .gaia-bubble .gaia-h1 {{ font-size: 14px; }}
.gaia-message .gaia-bubble .gaia-h2 {{ font-size: 13px; }}
.gaia-message .gaia-bubble .gaia-h3 {{ font-size: 12.5px; }}
.gaia-message .gaia-bubble .gaia-h4 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .02em; }}

.gaia-message .gaia-bubble .gaia-bullet {{
    margin: 2px 0;
    padding-left: 2px;
}}

/* ===== STRUCTURED INPUT FORM =====
   Scale Streamlit's default form/expander chrome down to fit the compact
   330px-wide card instead of looking oversized next to the 12px chat text. */
.st-key-gaia-card [data-testid="stExpander"] {{
    background: #ffffff !important;
    border: 1px solid #e1e5e8 !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}}

.st-key-gaia-card [data-testid="stExpander"] summary {{
    font-size: 12px !important;
    padding: 6px 10px !important;
}}

.st-key-gaia-card [data-testid="stExpander"] label,
.st-key-gaia-card [data-testid="stExpander"] p {{
    font-size: 11px !important;
}}

.st-key-gaia-card [data-testid="stExpander"] input,
.st-key-gaia-card [data-testid="stExpander"] textarea {{
    font-size: 12px !important;
}}

.st-key-gaia-card [data-testid="stExpander"] [data-testid="stFormSubmitButton"] button {{
    background: #123d2e !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 12px !important;
    padding: 4px 10px !important;
}}

/* ===== INPUT =====
   Sticky-positioned to the bottom of .st-key-gaia-card's own scroll area
   (see the height/overflow rule above) so it never scrolls out of view as
   the conversation grows — the person can always see and reach it without
   hunting for a scrollbar. */
[data-testid="stChatInput"] {{
    position: sticky !important;
    bottom: -14px !important;
    margin: 0 -14px !important;
    padding: 6px 14px 14px 14px !important;
    background: #ffffff !important;
    border: 0 !important;
    z-index: 5 !important;
}}

[data-testid="stChatInput"] > div {{
    background: #f7f8f9 !important;
    background-color: #f7f8f9 !important;
    border: 1px solid #d8dde2 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}}

[data-testid="stChatInput"] input {{
    color: #263238 !important;
}}

/* Dark-green rounded send button with the white arrow, to match the
   mockup instead of Streamlit's default grey icon button. */
[data-testid="stChatInputSubmitButton"] {{
    background: #123d2e !important;
    border-radius: 8px !important;
    width: 30px !important;
    height: 30px !important;
}}

[data-testid="stChatInputSubmitButton"] svg {{
    fill: #ffffff !important;
}}

/* ===== COLLAPSED "REOPEN" BUTTON ===== */
.st-key-gaia_reopen button {{
    position: fixed !important;
    right: 18px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;

    width: 56px !important;
    height: 56px !important;
    border-radius: 50% !important;

    background: #123d2e !important;
    color: #ffffff !important;
    font-size: 24px !important;
    border: none !important;

    box-shadow: 0 8px 26px rgba(0,0,0,.24) !important;
    z-index: 999998 !important;
}}

/* ===== MOBILE ===== */
@media (max-width: 700px) {{
    .st-key-gaia-card {{
        right: 10px !important;
        width: calc(100vw - 20px) !important;
        max-width: 330px !important;
    }}
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hello there! 🌿 I am G.A.I.A., Darukaa.Earth's Nature "
            "Intelligence Agent. I am thrilled to connect with you!"
        ),
    }, {
        "role": "assistant",
        "content": (
            "Whether you want to optimize your soil health, understand "
            "your local ecosystem dynamics, or plan for climate-resilient "
            "crops, I am here to help.\n\nWhat specific land metrics or "
            "environmental questions can I assist you with today?"
        ),
    }]

if "gaia_open" not in st.session_state:
    st.session_state.gaia_open = True

# 4. Chatbot
if st.session_state.gaia_open:
    with st.container(key="gaia-card"):
        header_col, minimize_col = st.columns([5, 1])
        with header_col:
            st.markdown(
                "<div class='gaia-header-title'><h3>G.A.I.A</h3></div>"
                "<div class='gaia-header-subtitle'>NATURE INTELLIGENCE AGENT</div>",
                unsafe_allow_html=True,
            )
        with minimize_col:
            if st.button("–", key="gaia_minimize", help="Minimize"):
                st.session_state.gaia_open = False
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # Shared submit path for BOTH input modes (free text via
        # st.chat_input below, and structured/JSON via the form here) so
        # there's exactly one place that appends to history, calls
        # query_giaa, and reruns.
        def _send_message(content):
          st.session_state.messages.append({"role": "user", "content": content})
          with st.spinner("Analyzing..."):
            bot_response = query_giaa(st.session_state.messages)
          st.session_state.messages.append({"role": "assistant", "content": bot_response})
          st.rerun()

        # Structured input mode — required by the brief alongside free
        # text ("Support at least: Text input (mandatory), Structured
        # input (JSON or similar)"). Two ways in: a plain form for the
        # metrics the brief's own example asks for, or a raw-JSON textbox
        # for anyone who wants to paste a payload directly.
        with st.expander("📋 Structured input (JSON)"):
          mode = st.radio(
              "Input mode",
              ["Form", "Raw JSON"],
              horizontal=True,
              key="struct_mode",
              label_visibility="collapsed",
          )

          if mode == "Form":
            with st.form(key="structured_form", clear_on_submit=True):
              soc = st.number_input(
                  "Soil Organic Carbon (%)", min_value=0.0, max_value=100.0,
                  value=None, format="%.2f", key="struct_soc",
              )
              rainfall = st.number_input(
                  "Annual Rainfall (mm)", min_value=0.0,
                  value=None, key="struct_rainfall",
              )
              ph = st.number_input(
                  "Soil pH", min_value=0.0, max_value=14.0,
                  value=None, format="%.1f", key="struct_ph",
              )
              land_use = st.text_input(
                  "Land Use / Crop Type", key="struct_land_use",
                  placeholder="e.g. monoculture wheat",
              )
              region = st.text_input(
                  "Region / Location", key="struct_region",
                  placeholder="e.g. semi-arid, Pune",
              )
              geo_col1, geo_col2 = st.columns(2)
              with geo_col1:
                lat = st.number_input(
                    "Latitude (optional)", value=None,
                    format="%.4f", key="struct_lat",
                )
              with geo_col2:
                lon = st.number_input(
                    "Longitude (optional)", value=None,
                    format="%.4f", key="struct_lon",
                )

              submitted = st.form_submit_button("Submit structured data")
              if submitted:
                payload = {
                    "soil_organic_carbon_percent": soc,
                    "annual_rainfall_mm": rainfall,
                    "soil_ph": ph,
                    "land_use_or_crop": land_use,
                    "region": region,
                    "latitude": lat,
                    "longitude": lon,
                }
                if any(v not in (None, "") for v in payload.values()):
                  _send_message(build_structured_message(payload))
                else:
                  st.warning("Fill in at least one field before submitting.")

          else:  # Raw JSON
            with st.form(key="structured_json_form", clear_on_submit=True):
              json_text = st.text_area(
                  "Paste JSON",
                  placeholder=(
                      '{\n  "soil_organic_carbon_percent": 0.3,\n'
                      '  "annual_rainfall_mm": 350,\n'
                      '  "land_use_or_crop": "monoculture wheat",\n'
                      '  "region": "semi-arid"\n}'
                  ),
                  key="struct_json_text",
                  height=140,
              )
              submitted = st.form_submit_button("Submit JSON")
              if submitted:
                if not json_text.strip():
                  st.warning("Paste a JSON payload before submitting.")
                else:
                  try:
                    parsed = json.loads(json_text)
                    _send_message(build_structured_message(parsed))
                  except json.JSONDecodeError as e:
                    st.error(f"That isn't valid JSON: {e}")

        # Explicit HTML bubbles: the white card and these bubbles are
        # independent of Streamlit's chat-message DOM/background behavior.
        # IMPORTANT: each bubble is rendered as its own single-line string
        # with no leading indentation. Markdown treats 4+ leading spaces as
        # a code block, which is exactly what turned the second message
        # into a raw <div> dump in the last screenshot.
        for message in st.session_state.messages:
            is_user = message["role"] == "user"
            role_class = "gaia-user" if is_user else "gaia-assistant"
            icon = "◉" if is_user else "🤖"
            text = str(message["content"])
            safe_text = markdown_lite_to_html(text)
            bubble_html = (
                f'<div class="gaia-message {role_class}">'
                f'<span class="gaia-icon">{icon}</span>'
                f'<div class="gaia-bubble">{safe_text}</div>'
                f'</div>'
            )
            st.markdown(bubble_html, unsafe_allow_html=True)

        # Placed last so it's the final item in the scrollable card; CSS
        # sticky-positions it to the bottom of that scroll area.
        if user_input := st.chat_input("Ask G.A.I.A..."):
            _send_message(user_input)

        # Auto-scroll the card to the newest message after each render, so
        # the person isn't left having to manually scroll down every turn.
        st.markdown(
            """
            <script>
            (function() {
                const card = window.parent.document.querySelector('.st-key-gaia-card');
                if (card) { card.scrollTop = card.scrollHeight; }
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )
else:
    if st.button("🌿", key="gaia_reopen", help="Open G.A.I.A"):
        st.session_state.gaia_open = True
        st.rerun()