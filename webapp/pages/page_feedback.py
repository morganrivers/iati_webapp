"""Feedback page – send feedback to the developers."""
import streamlit as st
import requests
import os


import logging

logger = logging.getLogger(__name__)

def render_feedback_page():
    _c1, _c2, _c3 = st.columns([1, 2, 1])
    with _c2:
        st.markdown("# 🖂 Feedback")
        st.markdown("### Help us improve")
        st.markdown("""
We'd love to hear your thoughts, suggestions, or bug reports about this forecasting tool.
Your feedback helps us make it better!
        """)

        with st.form("feedback_form"):
            name = st.text_input(
                "Your name",
                placeholder="Enter your name",
                help="How should we credit you?"
            )
            feedback = st.text_area(
                "Your feedback",
                placeholder="Share your thoughts, suggestions, or report a bug...",
                height=150,
                max_chars=2000,
                help="Maximum 2000 characters"
            )

            submitted = st.form_submit_button("Send Feedback", use_container_width=True)

            if submitted:
                if not name or not feedback:
                    st.error("Please fill in both name and feedback.")
                else:
                    # Send to Telegram bot endpoint
                    result = _send_feedback_to_telegram(name, feedback)
                    if result:
                        st.success("✅ Thank you! Your feedback has been sent.")
                        st.balloons()
                    else:
                        st.error("❌ Failed to send feedback. Please try again later.")


def _send_feedback_to_telegram(name: str, feedback: str) -> bool:
    """Send feedback directly to Telegram bot API."""
    try:
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if not token or not chat_id:
            logger.error("Error sending feedback: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env")
            return False

        text = f"Feedback from IATI forecasting webapp\nName: {name}\n\n{feedback}"
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={'chat_id': chat_id, 'text': text},
            timeout=10,
        )
        result = response.json()
        return result.get('ok', False)
    except Exception as e:
        logger.error(f"Error sending feedback: {e}")
        return False
