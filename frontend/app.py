import json
import sys
import time
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.pdf_generator import generate_mcq_papers, generate_descriptive_papers

API_BASE = "http://localhost:8000"


def _error_detail(resp) -> str:
    try:
        return resp.json().get("detail", "Unknown error")
    except Exception:
        return resp.text or "Unknown error"

st.set_page_config(page_title="EduCare", page_icon="📚", layout="wide")

# ── Session state init ──
for key, default in [
    ("session_id", None),
    ("chat_history", []),
    ("material_ready", False),
    ("generated_questions", None),
    ("generated_metadata", None),
    ("generated_type", None),
    ("generated_marks_per_q", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ════════════════════════════════════════════
#  SCREEN 1 — UPLOAD
# ════════════════════════════════════════════
if not st.session_state.material_ready:
    st.title("📚 EduCare — Educational RAG Agent")
    st.markdown("Upload your study material and start asking questions or generating question papers.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Upload file", type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT"
        )
    with col2:
        youtube_url = st.text_input(
            "Or paste a YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )

    if st.button("Process Material", type="primary", use_container_width=True):
        if not uploaded_file and not youtube_url.strip():
            st.error("Please upload a file or enter a YouTube URL.")
        else:
            with st.spinner("Processing study material... Building vector index..."):
                try:
                    old_sid = st.session_state.session_id or ""

                    if youtube_url.strip():
                        resp = requests.post(
                            f"{API_BASE}/upload",
                            data={"youtube_url": youtube_url.strip(), "old_session_id": old_sid},
                        )
                    else:
                        resp = requests.post(
                            f"{API_BASE}/upload",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                            data={"old_session_id": old_sid},
                        )

                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.material_ready = True
                        st.session_state.chat_history = []
                        st.session_state.generated_questions = None
                        st.success(f"Ready! {data['chunks_created']} chunks indexed.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Error: {_error_detail(resp)}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Make sure the FastAPI server is running on port 8000.")


# ════════════════════════════════════════════
#  SCREEN 2 — CHAT + GENERATE
# ════════════════════════════════════════════
else:
    with st.sidebar:
        st.title("📚 EduCare")
        st.success("Study material loaded")
        st.divider()
        if st.button("Upload New Material", use_container_width=True):
            if st.session_state.session_id:
                try:
                    requests.delete(f"{API_BASE}/session/{st.session_state.session_id}")
                except Exception:
                    pass
            st.session_state.session_id = None
            st.session_state.material_ready = False
            st.session_state.chat_history = []
            st.session_state.generated_questions = None
            st.rerun()

    tab1, tab2 = st.tabs(["💬 Chat", "📝 Generate Question Paper"])

    # ── TAB 1: CHAT ──
    with tab1:
        st.subheader("Chat with your study material")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask anything about the study material..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE}/chat",
                            json={"session_id": st.session_state.session_id, "question": prompt},
                        )
                        if resp.status_code == 200:
                            answer = resp.json()["answer"]
                            st.markdown(answer)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        else:
                            st.error(f"Error: {_error_detail(resp)}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend.")

    # ── TAB 2: GENERATE QUESTION PAPER ──
    with tab2:
        st.subheader("Configure Question Paper")

        col1, col2 = st.columns(2)
        with col1:
            q_type = st.radio("Question Type", ["MCQ", "Descriptive"], horizontal=True)
            topic_focus = st.text_input("Topic Focus (optional)", placeholder="e.g. Photosynthesis, World War II")
            total_marks = st.number_input("Total Marks", min_value=10, max_value=200, value=50, step=5)
            duration = st.number_input("Duration (minutes)", min_value=15, max_value=180, value=60, step=15)
            difficulty = st.select_slider("Difficulty Level", options=["easy", "medium", "hard"], value="medium")

        with col2:
            if q_type == "MCQ":
                st.markdown("**MCQ Settings**")
                num_questions = st.number_input("Number of Questions", min_value=5, max_value=100, value=25, step=5)
                marks_per_q = st.number_input("Marks per Question", min_value=1, max_value=10, value=2)
                negative_marking = st.checkbox("Enable Negative Marking")
            else:
                st.markdown("**Short Questions**")
                short_count = st.number_input("Number of Short Questions", min_value=0, max_value=20, value=5)
                short_marks = st.number_input("Marks each (Short)", min_value=1, max_value=10, value=2)
                st.markdown("**Long Questions**")
                long_count = st.number_input("Number of Long Questions", min_value=0, max_value=10, value=4)
                long_marks = st.number_input("Marks each (Long)", min_value=5, max_value=30, value=10)

        if st.button("Generate Question Paper", type="primary", use_container_width=True):
            payload = {
                "session_id": st.session_state.session_id,
                "type": q_type.lower(),
                "topic_focus": topic_focus,
                "total_marks": total_marks,
                "duration_minutes": duration,
                "difficulty": difficulty,
            }
            if q_type == "MCQ":
                payload.update({
                    "num_questions": num_questions,
                    "marks_per_question": marks_per_q,
                    "negative_marking": negative_marking,
                })
            else:
                payload.update({
                    "short_questions": {"count": short_count, "marks_each": short_marks},
                    "long_questions": {"count": long_count, "marks_each": long_marks},
                })

            with st.spinner("Generating question paper..."):
                try:
                    resp = requests.post(f"{API_BASE}/generate-questions", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.generated_questions = data["questions"]
                        st.session_state.generated_metadata = data["metadata"]
                        st.session_state.generated_type = q_type
                        st.session_state.generated_marks_per_q = marks_per_q if q_type == "MCQ" else None
                    else:
                        st.error(f"Error: {_error_detail(resp)}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend.")

        # ── Results — rendered from session state so toggles never trigger a regeneration ──
        if st.session_state.generated_questions:
            questions = st.session_state.generated_questions
            meta = st.session_state.generated_metadata
            q_type_stored = st.session_state.generated_type
            marks_per_q_stored = st.session_state.generated_marks_per_q

            st.divider()

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Type", meta["type"].upper())
            col_b.metric("Total Marks", meta["total_marks"])
            col_c.metric("Duration", f"{meta['duration_minutes']} min")
            col_d.metric("Difficulty", meta["difficulty"].capitalize())
            if meta.get("topic_focus") and meta["topic_focus"] != "General":
                st.caption(f"Topic Focus: {meta['topic_focus']}")

            # ── PDF Downloads ──
            st.markdown("#### Download")
            try:
                if q_type_stored == "MCQ":
                    qp_bytes, ak_bytes = generate_mcq_papers(questions, meta, marks_per_q_stored)
                else:
                    qp_bytes, ak_bytes = generate_descriptive_papers(questions, meta)

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        label="Download Question Paper (PDF)",
                        data=qp_bytes,
                        file_name="test_paper.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                with dl2:
                    st.download_button(
                        label="Download Answer Key (PDF)",
                        data=ak_bytes,
                        file_name="answer_key.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"PDF generation error: {e}")

            # ── Preview ──
            st.divider()
            st.markdown("#### Preview")
            show_answers = st.toggle("Show Answer Key in Preview")

            for q in questions:
                with st.container(border=True):
                    if q_type_stored == "MCQ":
                        st.markdown(f"**Q{q['q_no']}.** {q['question']} *[{marks_per_q_stored} mark(s)]*")
                        for opt, text in q["options"].items():
                            st.markdown(f"&nbsp;&nbsp;&nbsp;**{opt}.** {text}")
                        if show_answers:
                            st.success(
                                f"**Answer:** {q['correct_answer']}  \n"
                                f"**Explanation:** {q.get('explanation', '')}"
                            )
                    else:
                        q_label = "Short" if q.get("type") == "short" else "Long"
                        st.markdown(f"**Q{q['q_no']}.** [{q_label} — {q['marks']} marks]  \n{q['question']}")
                        if show_answers:
                            st.info(f"**Answer Hint:** {q.get('answer_hint', '')}")
