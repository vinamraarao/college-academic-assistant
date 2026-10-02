# """Streamlit front end for the AI College Academic Assistant.

# Presentation only. Retrieval, planning, routing and citation logic all live in
# `src/` and are called exactly as before.
# """

# from __future__ import annotations

# import logging
# import sys

# import streamlit as st

# from src.config import DATA_DIR, EMBEDDING_MODEL, OLLAMA_MODEL
# from src.graph.planner import WEEKDAY_NAMES, StudyPlan, format_plan
# from src.graph.workflow import run_assistant
# from src.llm.model import check_ollama
# from src.rag.vector_store import load_index
# from src.utils.helpers import AppError, OllamaUnavailableError

# logging.basicConfig(level=logging.WARNING)

# st.set_page_config(
#     page_title="College Academic Assistant",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )


# # --- Styling ---------------------------------------------------------------
# def inject_styles() -> None:
#     """Application styles. Clean, professional appearance with focus on
#     readability and subtle visual hierarchy."""
#     st.markdown(
#         """
#         <style>
#         /* ---- Base ---- */
#         :root {
#             --primary: #2563eb;
#             --primary-foreground: #ffffff;
#             --background: #ffffff;
#             --foreground: #171717;
#             --card: #f8fafc;
#             --card-foreground: #171717;
#             --popover: #ffffff;
#             --popover-foreground: #171717;
#             --border: #e2e8f0;
#             --input: #e2e8f0;
#             --accent: #2563eb;
#             --accent-foreground: #ffffff;
#         }

#         @media (prefers-color-scheme: dark) {
#             :root {
#                 --background: #0f172a;
#                 --foreground: #f8fafc;
#                 --card: #1e293b;
#                 --card-foreground: #f8fafc;
#                 --popover: #1e293b;
#                 --popover-foreground: #f8fafc;
#                 --border: #334155;
#                 --input: #334155;
#                 --accent: #3b82f6;
#                 --accent-foreground: #ffffff;
#             }
#         }

#         /* ---- Global ---- */
#         .stApp {
#             background-color: var(--background);
#             color: var(--foreground);
#         }

#         .main .block-container {
#             padding-top: 2rem;
#             padding-bottom: 4rem;
#             max-width: 800px;
#         }

#         #MainMenu, footer { visibility: hidden; }

#         /* ---- Header ---- */
#         .app-header {
#             text-align: center;
#             padding: 0 0 2rem 0;
#             border-bottom: 1px solid var(--border);
#             margin-bottom: 2rem;
#         }

#         .app-title {
#             font-size: 2.25rem;
#             font-weight: 800;
#             letter-spacing: -0.025em;
#             margin: 0 0 0.5rem 0;
#             background: linear-gradient(to right, var(--primary), #1d4ed8);
#             -webkit-background-clip: text;
#             -webkit-text-fill-color: transparent;
#             background-clip: text;
#         }

#         .app-subtitle {
#             font-size: 1.125rem;
#             opacity: 0.8;
#             margin: 0;
#             color: var(--foreground);
#         }

#         /* ---- Welcome Section ---- */
#         .welcome-container {
#             text-align: center;
#             padding: 3rem 1rem;
#             max-width: 600px;
#             margin: 0 auto;
#         }

#         .welcome-title {
#             font-size: 2rem;
#             font-weight: 700;
#             margin: 0 0 1rem 0;
#             color: var(--foreground);
#         }

#         .welcome-sub {
#             font-size: 1.125rem;
#             opacity: 0.7;
#             margin: 0 0 2rem 0;
#             color: var(--foreground);
#             line-height: 1.6;
#         }

#         .welcome-actions {
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
#             gap: 1rem;
#             margin-top: 2rem;
#         }

#         .welcome-action-btn {
#             background: var(--card);
#             border: 1px solid var(--border);
#             border-radius: 12px;
#             padding: 1.5rem;
#             text-align: left;
#             cursor: pointer;
#             transition: all 0.2s ease;
#             font-weight: 600;
#             font-size: 1rem;
#             color: var(--foreground);
#             display: flex;
#             flex-direction: column;
#             gap: 0.5rem;
#         }

#         .welcome-action-btn:hover {
#             border-color: var(--accent);
#             transform: translateY(-2px);
#             box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
#         }

#         .welcome-action-btn .action-icon {
#             font-size: 1.5rem;
#             margin-bottom: 0.5rem;
#         }

#         .welcome-action-btn .action-text {
#             font-weight: 600;
#         }

#         .welcome-action-btn .action-desc {
#             font-size: 0.875rem;
#             opacity: 0.7;
#         }

#         /* ---- Chat Messages ---- */
#         .stChatMessage {
#             padding: 1rem 0;
#         }

#         [data-testid="stChatMessage"]:has([data-testid="stChatMessageContent-Assistant"]) {
#             background-color: var(--card);
#             border-radius: 16px;
#             padding: 1.25rem;
#             margin: 0.5rem 0;
#             border: 1px solid var(--border);
#         }

#         [data-testid="stChatMessage"]:has([data-testid="stChatMessageContent-User"]) {
#             background-color: var(--accent);
#             border-radius: 16px;
#             padding: 1.25rem;
#             margin: 0.5rem 0;
#             border: 1px solid var(--accent);
#         }

#         [data-testid="stChatMessageContent-Assistant"] {
#             color: var(--foreground);
#             font-size: 1rem;
#             line-height: 1.6;
#         }

#         [data-testid="stChatMessageContent-User"] {
#             color: var(--primary-foreground);
#             font-weight: 500;
#             font-size: 1rem;
#             line-height: 1.6;
#         }

#         /* ---- Sources ---- */
#         .sources-container {
#             margin-top: 1rem;
#             padding-top: 0.75rem;
#             border-top: 1px solid var(--border);
#         }

#         .sources-title {
#             font-size: 0.875rem;
#             font-weight: 600;
#             color: var(--foreground);
#             opacity: 0.8;
#             margin-bottom: 0.5rem;
#         }

#         .source-item {
#             font-size: 0.875rem;
#             opacity: 0.7;
#             margin: 0.25rem 0;
#             padding-left: 1.5rem;
#             text-indent: -1.5rem;
#         }

#         .source-item::before {
#             content: "•";
#             color: var(--accent);
#             font-weight: bold;
#             display: inline-block;
#             width: 1.5rem;
#         }

#         /* ---- Unavailable Message ---- */
#         .unavailable-message {
#             font-size: 0.875rem;
#             opacity: 0.7;
#             font-style: italic;
#             margin-top: 1rem;
#             padding-top: 0.75rem;
#             border-top: 1px solid var(--border);
#             text-align: center;
#         }

#         /* ---- Study Plan Panel ---- */
#         .plan-panel {
#             background-color: var(--card);
#             border: 1px solid var(--border);
#             border-radius: 14px;
#             padding: 1.5rem;
#             margin: 1.5rem 0;
#         }

#         .plan-title {
#             font-size: 1.25rem;
#             font-weight: 600;
#             margin: 0 0 1rem 0;
#             color: var(--foreground);
#             display: flex;
#             align-items: center;
#             gap: 0.5rem;
#         }

#         .plan-description {
#             font-size: 0.875rem;
#             opacity: 0.7;
#             margin-bottom: 1.5rem;
#         }

#         .plan-stats {
#             display: flex;
#             justify-content: space-around;
#             margin-bottom: 1.5rem;
#             gap: 1rem;
#         }

#         .stat-item {
#             text-align: center;
#         }

#         .stat-label {
#             font-size: 0.75rem;
#             opacity: 0.7;
#             text-transform: uppercase;
#             letter-spacing: 0.05em;
#         }

#         .stat-value {
#             font-size: 1.125rem;
#             font-weight: 600;
#             margin-top: 0.25rem;
#         }

#         .plan-table {
#             width: 100%;
#             border-collapse: separate;
#             border-spacing: 0 0.5rem;
#             margin-top: 1rem;
#         }

#         .plan-table th,
#         .plan-table td {
#             padding: 0.75rem 1rem;
#             text-align: left;
#             font-size: 0.875rem;
#         }

#         .plan-table th {
#             background-color: var(--accent);
#             color: var(--accent-foreground);
#             font-weight: 600;
#             text-transform: uppercase;
#             font-size: 0.75rem;
#             letter-spacing: 0.05em;
#             border-top-left-radius: 8px;
#             border-top-right-radius: 8px;
#         }

#         .plan-table td {
#             background-color: rgba(var(--accent-foreground-rgb, 255, 255, 255), 0.05);
#             border-radius: 0;
#         }

#         .plan-table tr:first-child td {
#             border-top-left-radius: 8px;
#             border-top-right-radius: 8px;
#         }

#         .plan-table tr:last-child td {
#             border-bottom-left-radius: 8px;
#             border-bottom-right-radius: 8px;
#         }

#         .week-divider {
#             height: 1px;
#             background-color: var(--border);
#             margin: 1.5rem 0;
#             opacity: 0.5;
#         }

#         /* ---- Sidebar ---- */
#         .sidebar-content {
#             padding: 1.5rem 1rem;
#         }

#         .sidebar-title {
#             font-size: 1.25rem;
#             font-weight: 700;
#             text-align: center;
#             margin: 0 0 1.5rem 0;
#             color: var(--foreground);
#         }

#         .new-chat-btn {
#             width: 100%;
#             padding: 0.75rem 1rem;
#             background: var(--accent);
#             color: var(--accent-foreground);
#             border: none;
#             border-radius: 10px;
#             font-weight: 600;
#             font-size: 0.875rem;
#             cursor: pointer;
#             transition: opacity 0.2s ease;
#             margin-bottom: 1.5rem;
#         }

#         .new-chat-btn:hover {
#             opacity: 0.9;
#         }

#         .sidebar-section-title {
#             font-size: 0.75rem;
#             font-weight: 700;
#             text-transform: uppercase;
#             letter-spacing: 0.075em;
#             opacity: 0.6;
#             margin: 1.5rem 0 0.5rem 0;
#         }

#         .plan-summary {
#             background-color: var(--card);
#             border: 1px solid var(--border);
#             border-radius: 10px;
#             padding: 1rem;
#         }

#         .plan-summary-title {
#             font-size: 0.875rem;
#             font-weight: 600;
#             margin-bottom: 0.75rem;
#             display: flex;
#             align-items: center;
#             gap: 0.5rem;
#         }

#         .plan-summary-item {
#             display: flex;
#             justify-content: space-between;
#             padding: 0.25rem 0;
#             font-size: 0.875rem;
#         }

#         .plan-summary-label {
#             opacity: 0.7;
#         }

#         .plan-summary-value {
#             font-weight: 500;
#             text-align: right;
#         }

#         /* ---- Chat Input ---- */
#         .stChatInput {
#             border-radius: 12px;
#             border: 1px solid var(--border);
#             background-color: var(--card);
#         }

#         .stChatInput > div > div > textarea {
#             background-color: transparent !important;
#             border: none !important;
#             color: var(--foreground) !important;
#             font-size: 1rem !important;
#             padding: 1rem !important;
#             resize: none !important;
#         }

#         .stChatInput > div > div > textarea:focus {
#             outline: none !important;
#             box-shadow: none !important;
#             border-color: transparent !important;
#         }

#         /* ---- Status & Error Messages ---- */
#         .stAlert {
#             border-radius: 10px;
#         }

#         .stSuccess {
#             background-color: rgba(16, 185, 129, 0.1);
#             border-color: rgba(16, 185, 129, 0.3);
#             color: #10b981;
#         }

#         .stError {
#             background-color: rgba(239, 68, 68, 0.1);
#             border-color: rgba(239, 68, 68, 0.3);
#             color: #ef4444;
#         }

#         .stWarning {
#             background-color: rgba(245, 158, 11, 0.1);
#             border-color: rgba(245, 158, 11, 0.3);
#             color: #f59e0b;
#         }

#         /* ---- Loading Spinner ---- */
#         .stStatus {
#             background-color: transparent !important;
#             border: none !important;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )


# # --- Session state ---------------------------------------------------------
# def init_state() -> None:
#     defaults = {
#         "messages": [],      # chat transcript
#         "plan": None,        # active StudyPlan
#         "plan_request": {},  # inputs used to build the plan
#     }
#     for key, value in defaults.items():
#         st.session_state.setdefault(key, value)


# @st.cache_resource(show_spinner=False)
# def cached_index_status() -> dict:
#     """Check the index and Ollama once per process, not once per rerun."""
#     status: dict = {"index_ok": False, "ollama_ok": False, "models": [],
#                     "index_error": None, "ollama_error": None,
#                     "chunks": 0, "documents": 0}
#     try:
#         store = load_index()
#         status["index_ok"] = True
#         status["chunks"] = store.index.ntotal
#         status["documents"] = len(
#             {d.metadata.get("source") for d in store.docstore._dict.values()}
#         )
#     except Exception as exc:
#         status["index_error"] = str(exc)

#     try:
#         models = check_ollama()
#         status["ollama_ok"] = True
#         status["models"] = models
#     except OllamaUnavailableError as exc:
#         status["ollama_error"] = str(exc)
#     except Exception as exc:
#         status["ollama_error"] = str(exc)
#     return status


# # --- Content ---------------------------------------------------------------
# EXAMPLE_QUESTIONS = [
#     "What is the attendance requirement?",
#     "What documents are required for an internship?",
#     "What are the subjects in Semester 3?",
#     "When are the semester examinations?",
# ]

# PLANNER_PROMPT = (
#     "Create a study plan for me. I have Data Structures, Database Management "
#     "Systems and Operating Systems. My exams are in 20 days and I can study "
#     "3 hours per day."
# )


# def _status_row(ok: bool, icon: str, label: str) -> None:
#     st.markdown(
#         f'<div class="status-row">'
#         f'<span class="status-dot {"dot-ok" if ok else "dot-bad"}"></span>'
#         f'<span>{icon} {label}</span>'
#         f"</div>",
#         unsafe_allow_html=True,
#     )


# def render_sidebar() -> None:
#     with st.sidebar:
#         st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

#         st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
#         st.markdown('<div class="sidebar-title">🎓 College Academic Assistant</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#         if st.button("＋  New Chat", use_container_width=True, type="primary"):
#             st.session_state["messages"] = []
#             st.session_state["plan"] = None
#             st.session_state["plan_request"] = {}
#             st.session_state.pop("pending", None)
#             st.rerun()

#         plan = st.session_state["plan"]
#         if plan is not None:
#             st.markdown('<div class="plan-summary">', unsafe_allow_html=True)
#             st.markdown(
#                 f'<div class="plan-summary-title">📅 Study Plan</div>',
#                 unsafe_allow_html=True,
#             )
#             st.markdown(
#                 f'<div class="plan-summary-item">'
#                 f'<span class="plan-summary-label">Subjects</span>'
#                 f'<span class="plan-summary-value">{len(plan.subjects)}</span>'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )
#             st.markdown(
#                 f'<div class="plan-summary-item">'
#                 f'<span class="plan-summary-label">Hours</span>'
#                 f'<span class="plan-summary-value">{plan.total_hours}</span>'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )
#             st.markdown(
#                 f'<div class="plan-summary-item">'
#                 f'<span class="plan-summary-label">Sessions</span>'
#                 f'<span class="plan-summary-value">{plan.total_sessions}</span>'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )
#             blocked = (
#                 ", ".join(WEEKDAY_NAMES[d] for d in plan.unavailable_days)
#                 if plan.unavailable_days
#                 else "None"
#             )
#             if blocked != "None":
#                 st.markdown(
#                     f'<div class="plan-summary-item">'
#                     f'<span class="plan-summary-label">Excluded</span>'
#                     f'<span class="plan-summary-value">{blocked}</span>'
#                     f'</div>',
#                     unsafe_allow_html=True,
#                 )
#             st.markdown('</div>', unsafe_allow_html=True)

#         st.markdown('</div>', unsafe_allow_html=True)


# def render_header() -> None:
#     st.markdown(
#         '<div class="app-header">'
#         '<div class="app-title">College Academic Assistant</div>'
#         '<div class="app-subtitle">Your AI assistant for college academics</div>'
#         '</div>',
#         unsafe_allow_html=True,
#     )


# def render_welcome() -> None:
#     """Shown instead of an empty chat area before the first question."""
#     st.markdown(
#         '<div class="welcome-container">'
#         '<div class="welcome-title">How can I help you today?</div>'
#         '<div class="welcome-sub">Ask about your syllabus, examinations, attendance, '
#         'internships or regulations. Answers come from your college documents.</div>'
#         '</div>',
#         unsafe_allow_html=True,
#     )

#     for row_start in range(0, len(EXAMPLE_QUESTIONS), 2):
#         cols = st.columns(2)
#         for col, question in zip(cols, EXAMPLE_QUESTIONS[row_start : row_start + 2]):
#             with col:
#                 if st.button(question, key=f"ex_{row_start}_{question[:12]}",
#                              use_container_width=True):
#                     st.session_state["pending"] = question
#                     st.rerun()

#     # Add Create study plan as a separate button below the grid
#     st.markdown('<div class="welcome-actions">', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="welcome-action-btn">'
#         '<span class="action-icon">📚</span>'
#         '<span class="action-text">Create a study plan</span>'
#         '<span class="action-desc">Generate personalized study schedule and reminders</span>'
#         '</div>',
#         unsafe_allow_html=True,
#     )
#     st.markdown('</div>', unsafe_allow_html=True)


# def render_plan(plan) -> None:
#     """Render the study plan as a clean planning interface."""
#     st.markdown('<div class="plan-container">', unsafe_allow_html=True)

#     st.markdown('<div class="plan-header">', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="plan-title">📅 Your study plan</div>',
#         unsafe_allow_html=True,
#     )

#     st.markdown('<div class="plan-stats">', unsafe_allow_html=True)
#     st.markdown(
#         f'<div class="plan-stat"><strong>{len(plan.subjects)}</strong> Subjects</div>',
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         f'<div class="plan-stat"><strong>{plan.total_hours}</strong> Hours</div>',
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         f'<div class="plan-stat"><strong>{plan.total_sessions}</strong> Sessions</div>',
#         unsafe_allow_html=True,
#     )

#     blocked = (
#         ", ".join(WEEKDAY_NAMES[d] for d in plan.unavailable_days)
#         if plan.unavailable_days
#         else "None"
#     )
#     st.markdown(
#         f'<div class="plan-stat">Not scheduled: <strong>{blocked}</strong></div>',
#         unsafe_allow_html=True,
#     )
#     st.markdown('</div>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown('<div class="plan-schedule">', unsafe_allow_html=True)

#     rows = [
#         {
#             "Day": day["weekday"],
#             "Date": day["date"],
#             "Hours": day["hours"],
#             "Sessions": " • ".join(
#                 f"{s['subject']} ({s['minutes']}m)" for s in day["sessions"]
#             ),
#         }
#         for day in plan.schedule
#     ]

#     st.markdown('<table class="schedule-table">', unsafe_allow_html=True)
#     st.markdown('<thead><tr>', unsafe_allow_html=True)
#     st.markdown('<th>Day</th><th>Date</th><th>Hours</th><th>Sessions</th>', unsafe_allow_html=True)
#     st.markdown('</tr></thead>', unsafe_allow_html=True)

#     for row in rows:
#         st.markdown('<tr>', unsafe_allow_html=True)
#         st.markdown(f'<td>{row["Day"]}</td><td>{row["Date"]}</td><td>{row["Hours"]}</td><td>{row["Sessions"]}</td>', unsafe_allow_html=True)
#         st.markdown('</tr>', unsafe_allow_html=True)

#     st.markdown('</table>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)


# def render_citations(sources: list[dict]) -> None:
#     """Citations come from retrieval metadata; the LLM never supplies them."""
#     if not sources:
#         return

#     st.markdown('<div class="sources-container">', unsafe_allow_html=True)
#     st.markdown('<div class="sources-title">Sources</div>', unsafe_allow_html=True)

#     for source in sources:
#         page = source.get("page")
#         location = f"Page {page}" if page not in (None, 0, "") else "Page n/a"
#         # Render as normal text, not with asterisks
#         st.markdown(f'<div class="source-item">{source["source"]} · {location}</div>', unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# def render_history() -> None:
#     for message in st.session_state["messages"]:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             sources = message.get("sources")
#             if sources:
#                 render_citations(sources)
#             elif message.get("unavailable"):
#                 st.markdown(
#                     '<div class="unavailable-message">'
#                     'I couldn\'t find this information in the available college documents.'
#                     '</div>',
#                     unsafe_allow_html=True,
#                 )
#         if message.get("plan"):
#             render_plan(StudyPlan(**message["plan"]))


# def _record_answer(result: dict, answer: str, plan=None) -> None:
#     """Persist this turn, including whether it was grounded."""
#     entry = {
#         "role": "assistant",
#         "content": answer,
#         "sources": result.get("sources", []) if result.get("grounded") else [],
#     }
#     if not entry["sources"]:
#         # A plan turn is not a knowledge-base answer, but it is still an answer,
#         # so it is not labelled as a refusal.
#         entry["unavailable"] = result.get("plan") is None
#     if plan is not None:
#         # Snapshot the dataclass fields only. `StudyPlan.to_dict()` also returns
#         # derived values (total_sessions/total_hours) that are not constructor
#         # arguments, so it cannot be splatted back into StudyPlan(**...).
#         entry["plan"] = {
#             "subjects": plan.subjects,
#             "days_until_exam": plan.days_until_exam,
#             "hours_per_day": plan.hours_per_day,
#             "unavailable_days": plan.unavailable_days,
#             "session_minutes": plan.session_minutes,
#             "schedule": plan.schedule,
#             "subject_hours": plan.subject_hours,
#         }
#     st.session_state["messages"].append(entry)


# def answer_question(question: str) -> None:
#     """Run the assistant for one turn and render the result."""
#     st.session_state["messages"].append({"role": "user", "content": question})

#     with st.chat_message("assistant"):
#         status = st.status("Searching your college documents...", expanded=False)
#         try:
#             with st.spinner("Preparing your answer..."):
#                 result = run_assistant(
#                     question=question,
#                     history=st.session_state["messages"][:-1],
#                     plan=st.session_state["plan"],
#                     plan_request=st.session_state["plan_request"],
#                 )
#         except AppError as exc:
#             status.update(label="Setup required", state="error")
#             st.error(str(exc))
#             return
#         except Exception:
#             logging.exception("Assistant failed")
#             status.update(label="Something went wrong", state="error")
#             st.error(
#                 "The assistant could not respond this time. Please try again."
#             )
#             return

#         if result.get("plan") is not None:
#             st.session_state["plan"] = result["plan"]
#         if result.get("plan_request"):
#             st.session_state["plan_request"] = result["plan_request"]

#         answer = result.get("answer") or "I could not produce an answer."
#         grounded = result.get("grounded", False)
#         show_plan = (
#             result.get("plan") is not None
#             and result.get("intent") in {"PLAN_CREATE", "PLAN_MODIFY"}
#         )

#         if result.get("error"):
#             st.warning(result["error"])

#         st.markdown(answer)

#         if grounded:
#             render_citations(result.get("sources", []))
#             status.update(label="Preparing your answer...", state="complete")
#         elif result.get("plan") is not None:
#             status.update(label="Preparing your answer...", state="complete")
#         else:
#             status.update(label="Preparing your answer...", state="complete")

#         _record_answer(result, answer, plan=result["plan"] if show_plan else None)

#     # main() reruns afterwards so the sidebar picks up the new plan; the panel
#     # itself is re-rendered from the stored message by render_history().
#     if result.get("plan") is not None and result.get("intent") in {
#         "PLAN_CREATE", "PLAN_MODIFY"
#     }:
#         render_plan(result["plan"])


# def main() -> None:
#     init_state()
#     inject_styles()

#     status = cached_index_status()
#     render_header()

#     if not status["index_ok"] or not status["ollama_ok"]:
#         st.warning("Setup is incomplete — the assistant cannot answer yet.")
#         st.caption("Please check the setup and reload the page.")
#         return

#     if not st.session_state["messages"]:
#         render_welcome()

#     render_history()

#     pending = st.session_state.pop("pending", None)
#     typed = st.chat_input(
#         "Ask about your syllabus, exams, attendance, internship, or study plan..."
#     )
#     question = pending or typed
#     if question:
#         answer_question(question)
#         # Answering mutates session state (a plan may be created or updated), so
#         # the sidebar is drawn afterwards. Rendering it earlier would show the
#         # previous run's plan, or none on the turn that creates one.
#         st.rerun()

#     # Drawn last so it reflects any plan produced during this run.
#     render_sidebar()


# if __name__ == "__main__":
#     try:
#         main()
#     except SystemExit:
#         raise
#     except Exception:
#         logging.exception("Fatal error in Streamlit app")
#         st.error("Something went wrong while processing your request. Please try again.")
#         sys.exit(1)





"""Streamlit front end for the AI College Academic Assistant.

Presentation only. Retrieval, planning, routing and citation logic all live in
`src/` and are called exactly as before.
"""

from __future__ import annotations

import logging
import sys

import streamlit as st

from src.config import DATA_DIR, EMBEDDING_MODEL, OLLAMA_MODEL
from src.graph.planner import WEEKDAY_NAMES, StudyPlan, format_plan
from src.graph.workflow import run_assistant
from src.llm.model import check_ollama
from src.rag.vector_store import load_index
from src.utils.helpers import AppError, OllamaUnavailableError

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="College Academic Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Styling ---------------------------------------------------------------
def inject_styles() -> None:
    """Application styles. Modern, polished appearance with a refined
    visual hierarchy, smooth transitions, and a cohesive design language."""
    st.markdown(
        """
        <style>
        /* ---- Design Tokens ---- */
        :root {
            --brand-50:  #eff6ff;
            --brand-100: #dbeafe;
            --brand-200: #bfdbfe;
            --brand-300: #93c5fd;
            --brand-400: #60a5fa;
            --brand-500: #3b82f6;
            --brand-600: #2563eb;
            --brand-700: #1d4ed8;
            --brand-800: #1e40af;
            --brand-900: #1e3a8a;

            --surface:       #ffffff;
            --surface-alt:   #f8fafc;
            --surface-sunk:  #f1f5f9;
            --surface-card:  #ffffff;

            --ink:           #0f172a;
            --ink-muted:     #475569;
            --ink-faint:     #94a3b8;

            --line:          #e2e8f0;
            --line-strong:   #cbd5e1;

            --success:       #10b981;
            --warning:       #f59e0b;
            --danger:        #ef4444;

            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;

            --shadow-xs: 0 1px 2px rgba(15, 23, 42, 0.04);
            --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
            --shadow-md: 0 4px 12px rgba(15, 23, 42, 0.07), 0 2px 4px rgba(15, 23, 42, 0.04);
            --shadow-lg: 0 12px 32px rgba(15, 23, 42, 0.10), 0 4px 8px rgba(15, 23, 42, 0.05);
            --shadow-focus: 0 0 0 4px rgba(37, 99, 235, 0.12);

            --transition-fast: 120ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --surface:       #0b1220;
                --surface-alt:   #111a2e;
                --surface-sunk:  #0f172a;
                --surface-card:  #1e293b;

                --ink:           #f1f5f9;
                --ink-muted:     #94a3b8;
                --ink-faint:     #64748b;

                --line:          #1e293b;
                --line-strong:   #334155;

                --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.3);
                --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.35), 0 1px 2px rgba(0, 0, 0, 0.25);
                --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4), 0 2px 4px rgba(0, 0, 0, 0.25);
                --shadow-lg: 0 12px 32px rgba(0, 0, 0, 0.5), 0 4px 8px rgba(0, 0, 0, 0.3);
                --shadow-focus: 0 0 0 4px rgba(59, 130, 246, 0.2);
            }
        }

        /* ---- Global Reset & Base ---- */
        * {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, 'Helvetica Neue', sans-serif;
        }

        .stApp {
            background-color: var(--surface);
            color: var(--ink);
            background-image:
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(37, 99, 235, 0.04), transparent),
                radial-gradient(ellipse 60% 40% at 80% 110%, rgba(37, 99, 235, 0.03), transparent);
            background-attachment: fixed;
        }

        .main .block-container {
            padding-top: 2.5rem;
            padding-bottom: 5rem;
            max-width: 820px;
            margin: 0 auto;
        }

        #MainMenu, footer, header { visibility: hidden; }

        /* ---- Scrollbar ---- */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: var(--line-strong);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--ink-faint); }

        /* ---- Header ---- */
        .app-header {
            text-align: center;
            padding: 1rem 0 2.5rem 0;
            margin-bottom: 2rem;
            position: relative;
        }

        .app-header::after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 64px;
            height: 3px;
            border-radius: 2px;
            background: linear-gradient(90deg, var(--brand-400), var(--brand-600));
        }

        .app-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.85rem;
            background: linear-gradient(135deg, var(--brand-50), var(--brand-100));
            border: 1px solid var(--brand-200);
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            color: var(--brand-700);
            margin-bottom: 1rem;
            text-transform: uppercase;
        }

        @media (prefers-color-scheme: dark) {
            .app-badge {
                background: rgba(37, 99, 235, 0.15);
                border-color: rgba(59, 130, 246, 0.3);
                color: var(--brand-300);
            }
        }

        .app-title {
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.035em;
            line-height: 1.1;
            margin: 0 0 0.75rem 0;
            background: linear-gradient(135deg, var(--brand-600) 0%, var(--brand-800) 50%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        @media (prefers-color-scheme: dark) {
            .app-title {
                background: linear-gradient(135deg, var(--brand-300) 0%, var(--brand-500) 50%, #a78bfa 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
        }

        .app-subtitle {
            font-size: 1.0625rem;
            font-weight: 400;
            opacity: 0.65;
            margin: 0;
            color: var(--ink);
            letter-spacing: -0.01em;
        }

        /* ---- Welcome Section ---- */
        .welcome-container {
            text-align: center;
            padding: 2.5rem 1rem 1.5rem 1rem;
            max-width: 620px;
            margin: 0 auto;
        }

        .welcome-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 56px;
            height: 56px;
            border-radius: var(--radius-lg);
            background: linear-gradient(135deg, var(--brand-500), var(--brand-700));
            font-size: 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-md);
        }

        .welcome-title {
            font-size: 1.875rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin: 0 0 0.75rem 0;
            color: var(--ink);
        }

        .welcome-sub {
            font-size: 1.0625rem;
            opacity: 0.65;
            margin: 0 0 2.5rem 0;
            color: var(--ink);
            line-height: 1.65;
            max-width: 520px;
            margin-left: auto;
            margin-right: auto;
        }

        /* ---- Example Question Buttons ---- */
        .stButton > button {
            transition: all var(--transition-base);
            font-family: inherit;
        }

        .stButton > button[kind="secondary"] {
            background: var(--surface-card);
            border: 1px solid var(--line);
            border-radius: var(--radius-md);
            color: var(--ink);
            font-weight: 500;
            font-size: 0.9375rem;
            padding: 0.875rem 1.125rem;
            text-align: left;
            box-shadow: var(--shadow-xs);
            height: auto;
            min-height: 56px;
            white-space: normal;
            line-height: 1.4;
        }

        .stButton > button[kind="secondary"]:hover {
            border-color: var(--brand-400);
            background: var(--brand-50);
            color: var(--brand-700);
            box-shadow: var(--shadow-sm);
            transform: translateY(-1px);
        }

        @media (prefers-color-scheme: dark) {
            .stButton > button[kind="secondary"]:hover {
                background: rgba(37, 99, 235, 0.12);
                color: var(--brand-300);
            }
        }

        .stButton > button[kind="secondary"]:active {
            transform: translateY(0);
            box-shadow: var(--shadow-xs);
        }

        .stButton > button[kind="secondary"]:focus-visible {
            outline: none;
            box-shadow: var(--shadow-focus);
            border-color: var(--brand-500);
        }

        /* ---- Chat Messages ---- */
        .stChatMessage {
            padding: 0;
            margin: 0.5rem 0;
            animation: messageIn 300ms cubic-bezier(0.4, 0, 0.2, 1);
        }

        @keyframes messageIn {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageContent-Assistant"]) {
            background-color: var(--surface-card);
            border-radius: var(--radius-lg);
            padding: 1.25rem 1.5rem;
            margin: 0.75rem 0;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-xs);
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageContent-User"]) {
            background: linear-gradient(135deg, var(--brand-600), var(--brand-700));
            border-radius: var(--radius-lg);
            padding: 1.25rem 1.5rem;
            margin: 0.75rem 0;
            border: 1px solid var(--brand-700);
            box-shadow: var(--shadow-md);
        }

        [data-testid="stChatMessageContent-Assistant"] {
            color: var(--ink);
            font-size: 0.9375rem;
            line-height: 1.7;
        }

        [data-testid="stChatMessageContent-User"] {
            color: #ffffff;
            font-weight: 500;
            font-size: 0.9375rem;
            line-height: 1.6;
        }

        [data-testid="stChatMessageAvatar-Assistant"] {
            background: linear-gradient(135deg, var(--brand-500), var(--brand-700));
        }

        [data-testid="stChatMessageAvatar-User"] {
            background: linear-gradient(135deg, #64748b, #475569);
        }

        /* ---- Sources ---- */
        .sources-container {
            margin-top: 1.125rem;
            padding-top: 1rem;
            border-top: 1px dashed var(--line-strong);
        }

        .sources-title {
            font-size: 0.6875rem;
            font-weight: 700;
            color: var(--ink-faint);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.625rem;
            display: flex;
            align-items: center;
            gap: 0.375rem;
        }

        .sources-title::before {
            content: "";
            width: 3px;
            height: 12px;
            border-radius: 2px;
            background: linear-gradient(180deg, var(--brand-400), var(--brand-600));
        }

        .source-item {
            font-size: 0.8125rem;
            color: var(--ink-muted);
            margin: 0.375rem 0;
            padding: 0.5rem 0.75rem;
            background: var(--surface-sunk);
            border-radius: var(--radius-sm);
            border: 1px solid var(--line);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all var(--transition-fast);
        }

        .source-item:hover {
            border-color: var(--brand-300);
            background: var(--brand-50);
        }

        @media (prefers-color-scheme: dark) {
            .source-item:hover {
                background: rgba(37, 99, 235, 0.1);
            }
        }

        .source-item .source-name {
            font-weight: 600;
            color: var(--ink);
        }

        .source-item .source-sep {
            opacity: 0.35;
        }

        .source-item .source-page {
            color: var(--ink-faint);
        }

        /* ---- Unavailable Message ---- */
        .unavailable-message {
            font-size: 0.875rem;
            color: var(--ink-muted);
            font-style: italic;
            margin-top: 1rem;
            padding: 0.875rem 1.125rem;
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(245, 158, 11, 0.04));
            border: 1px solid rgba(245, 158, 11, 0.2);
            border-radius: var(--radius-md);
            display: flex;
            align-items: flex-start;
            gap: 0.625rem;
            line-height: 1.55;
        }

        .unavailable-message::before {
            content: "ℹ";
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--warning);
            color: #fff;
            font-size: 0.6875rem;
            font-weight: 700;
            flex-shrink: 0;
            margin-top: 1px;
        }

        /* ---- Study Plan Panel ---- */
        .plan-container {
            background: var(--surface-card);
            border: 1px solid var(--line);
            border-radius: var(--radius-xl);
            padding: 1.75rem;
            margin: 1.25rem 0;
            box-shadow: var(--shadow-md);
            overflow: hidden;
            position: relative;
        }

        .plan-container::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--brand-400), var(--brand-600), #7c3aed);
        }

        .plan-header {
            margin-bottom: 1.5rem;
        }

        .plan-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0 0 1.25rem 0;
            color: var(--ink);
            display: flex;
            align-items: center;
            gap: 0.625rem;
        }

        .plan-title-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: var(--radius-md);
            background: linear-gradient(135deg, var(--brand-500), var(--brand-700));
            font-size: 1.125rem;
            box-shadow: var(--shadow-sm);
        }

        .plan-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .plan-stat {
            background: var(--surface-sunk);
            border: 1px solid var(--line);
            border-radius: var(--radius-md);
            padding: 0.875rem 1rem;
            font-size: 0.8125rem;
            color: var(--ink-muted);
            text-align: center;
            transition: all var(--transition-fast);
        }

        .plan-stat:hover {
            border-color: var(--brand-300);
            transform: translateY(-1px);
            box-shadow: var(--shadow-sm);
        }

        .plan-stat strong {
            display: block;
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--brand-600);
            margin-bottom: 0.125rem;
            letter-spacing: -0.02em;
        }

        @media (prefers-color-scheme: dark) {
            .plan-stat strong { color: var(--brand-400); }
        }

        .plan-schedule {
            margin-top: 1rem;
        }

        .schedule-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.8125rem;
            border-radius: var(--radius-md);
            overflow: hidden;
            border: 1px solid var(--line);
        }

        .schedule-table thead th {
            background: linear-gradient(135deg, var(--brand-600), var(--brand-700));
            color: #ffffff;
            font-weight: 600;
            font-size: 0.6875rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 0.75rem 1rem;
            text-align: left;
            white-space: nowrap;
        }

        .schedule-table tbody td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--line);
            color: var(--ink);
            vertical-align: top;
            line-height: 1.5;
        }

        .schedule-table tbody tr:last-child td {
            border-bottom: none;
        }

        .schedule-table tbody tr {
            transition: background var(--transition-fast);
        }

        .schedule-table tbody tr:nth-child(even) {
            background: var(--surface-sunk);
        }

        .schedule-table tbody tr:hover {
            background: var(--brand-50);
        }

        @media (prefers-color-scheme: dark) {
            .schedule-table tbody tr:hover {
                background: rgba(37, 99, 235, 0.08);
            }
        }

        .schedule-table tbody td:first-child {
            font-weight: 600;
            color: var(--brand-700);
            white-space: nowrap;
        }

        @media (prefers-color-scheme: dark) {
            .schedule-table tbody td:first-child { color: var(--brand-300); }
        }

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {
            background: var(--surface-alt);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }

        .sidebar-content {
            padding: 0.5rem 0.75rem 1.5rem 0.75rem;
        }

        .sidebar-header {
            text-align: center;
            padding: 0.5rem 0 1.5rem 0;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.25rem;
        }

        .sidebar-brand {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 44px;
            height: 44px;
            border-radius: var(--radius-md);
            background: linear-gradient(135deg, var(--brand-500), var(--brand-700));
            font-size: 1.375rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-sm);
        }

        .sidebar-title {
            font-size: 1.0625rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            color: var(--ink);
        }

        .sidebar-subtitle {
            font-size: 0.75rem;
            color: var(--ink-faint);
            margin-top: 0.25rem;
        }

        /* ---- Sidebar Buttons ---- */
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--brand-600), var(--brand-700));
            border: none;
            border-radius: var(--radius-md);
            color: #ffffff;
            font-weight: 600;
            font-size: 0.875rem;
            padding: 0.75rem 1rem;
            box-shadow: var(--shadow-sm);
            transition: all var(--transition-base);
            width: 100%;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
            filter: brightness(1.05);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"]:active {
            transform: translateY(0);
        }

        /* ---- Sidebar Plan Summary ---- */
        .sidebar-section-title {
            font-size: 0.6875rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--ink-faint);
            margin: 1.5rem 0 0.75rem 0;
            padding-left: 0.25rem;
        }

        .plan-summary {
            background: var(--surface-card);
            border: 1px solid var(--line);
            border-radius: var(--radius-md);
            padding: 1.125rem;
            box-shadow: var(--shadow-xs);
        }

        .plan-summary-title {
            font-size: 0.8125rem;
            font-weight: 700;
            margin-bottom: 0.875rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--ink);
            letter-spacing: -0.01em;
        }

        .plan-summary-title::before {
            content: "";
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--brand-500);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
        }

        .plan-summary-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            font-size: 0.8125rem;
            border-bottom: 1px solid var(--line);
        }

        .plan-summary-item:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }

        .plan-summary-label {
            color: var(--ink-muted);
            font-weight: 500;
        }

        .plan-summary-value {
            font-weight: 700;
            color: var(--brand-600);
            text-align: right;
            letter-spacing: -0.01em;
        }

        @media (prefers-color-scheme: dark) {
            .plan-summary-value { color: var(--brand-400); }
        }

        /* ---- Chat Input ---- */
        [data-testid="stChatInput"] {
            border-radius: var(--radius-md);
            border: 1.5px solid var(--line-strong);
            background: var(--surface-card);
            box-shadow: var(--shadow-sm);
            transition: all var(--transition-base);
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: var(--brand-500);
            box-shadow: var(--shadow-focus), var(--shadow-md);
        }

        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            border: none !important;
            color: var(--ink) !important;
            font-size: 0.9375rem !important;
            padding: 0.875rem 1rem !important;
            resize: none !important;
            font-family: inherit !important;
        }

        [data-testid="stChatInput"] textarea:focus {
            outline: none !important;
            box-shadow: none !important;
            border-color: transparent !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--ink-faint) !important;
        }

        /* ---- Status & Alerts ---- */
        .stAlert {
            border-radius: var(--radius-md);
            border-width: 1px;
            font-size: 0.875rem;
            box-shadow: var(--shadow-xs);
        }

        [data-testid="stAlertContainer"] {
            border-radius: var(--radius-md);
        }

        /* ---- Status Widget ---- */
        [data-testid="stStatusWidget"] {
            background: transparent !important;
            border: none !important;
        }

        [data-testid="stStatusWidget"] > div {
            background: var(--surface-card) !important;
            border: 1px solid var(--line) !important;
            border-radius: var(--radius-md) !important;
            box-shadow: var(--shadow-xs) !important;
        }

        /* ---- Spinner ---- */
        .stSpinner > div {
            border-top-color: var(--brand-500) !important;
        }

        /* ---- Divider ---- */
        hr {
            border-color: var(--line);
            opacity: 0.6;
        }

        /* ---- Responsive ---- */
        @media (max-width: 640px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1.5rem;
            }
            .app-title { font-size: 1.875rem; }
            .app-subtitle { font-size: 0.9375rem; }
            .welcome-title { font-size: 1.5rem; }
            .plan-container { padding: 1.25rem; }
            .schedule-table { font-size: 0.75rem; }
            .schedule-table thead th,
            .schedule-table tbody td { padding: 0.625rem 0.75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Session state ---------------------------------------------------------
def init_state() -> None:
    defaults = {
        "messages": [],      # chat transcript
        "plan": None,        # active StudyPlan
        "plan_request": {},  # inputs used to build the plan
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def cached_index_status() -> dict:
    """Check the index and Ollama once per process, not once per rerun."""
    status: dict = {"index_ok": False, "ollama_ok": False, "models": [],
                    "index_error": None, "ollama_error": None,
                    "chunks": 0, "documents": 0}
    try:
        store = load_index()
        status["index_ok"] = True
        status["chunks"] = store.index.ntotal
        status["documents"] = len(
            {d.metadata.get("source") for d in store.docstore._dict.values()}
        )
    except Exception as exc:
        status["index_error"] = str(exc)

    try:
        models = check_ollama()
        status["ollama_ok"] = True
        status["models"] = models
    except OllamaUnavailableError as exc:
        status["ollama_error"] = str(exc)
    except Exception as exc:
        status["ollama_error"] = str(exc)
    return status


# --- Content ---------------------------------------------------------------
EXAMPLE_QUESTIONS = [
    "What is the attendance requirement?",
    "What documents are required for an internship?",
    "What are the subjects in Semester 3?",
    "When are the semester examinations?",
]

PLANNER_PROMPT = (
    "Create a study plan for me. I have Data Structures, Database Management "
    "Systems and Operating Systems. My exams are in 20 days and I can study "
    "3 hours per day."
)


def _status_row(ok: bool, icon: str, label: str) -> None:
    st.markdown(
        f'<div class="status-row">'
        f'<span class="status-dot {"dot-ok" if ok else "dot-bad"}"></span>'
        f'<span>{icon} {label}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-brand">🎓</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Academic Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Powered by your college documents</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("＋  New Chat", use_container_width=True, type="primary"):
            st.session_state["messages"] = []
            st.session_state["plan"] = None
            st.session_state["plan_request"] = {}
            st.session_state.pop("pending", None)
            st.rerun()

        plan = st.session_state["plan"]
        if plan is not None:
            st.markdown('<div class="sidebar-section-title">Active Plan</div>', unsafe_allow_html=True)
            st.markdown('<div class="plan-summary">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="plan-summary-title">📅 Study Plan</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="plan-summary-item">'
                f'<span class="plan-summary-label">Subjects</span>'
                f'<span class="plan-summary-value">{len(plan.subjects)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="plan-summary-item">'
                f'<span class="plan-summary-label">Hours</span>'
                f'<span class="plan-summary-value">{plan.total_hours}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="plan-summary-item">'
                f'<span class="plan-summary-label">Sessions</span>'
                f'<span class="plan-summary-value">{plan.total_sessions}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            blocked = (
                ", ".join(WEEKDAY_NAMES[d] for d in plan.unavailable_days)
                if plan.unavailable_days
                else "None"
            )
            if blocked != "None":
                st.markdown(
                    f'<div class="plan-summary-item">'
                    f'<span class="plan-summary-label">Excluded</span>'
                    f'<span class="plan-summary-value">{blocked}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        '<div class="app-header">'
        '<div class="app-badge">🎓 AI-Powered</div>'
        '<div class="app-title">College Academic Assistant</div>'
        '<div class="app-subtitle">Your AI assistant for college academics</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    """Shown instead of an empty chat area before the first question."""
    st.markdown(
        '<div class="welcome-container">'
        '<div class="welcome-icon">💬</div>'
        '<div class="welcome-title">How can I help you today?</div>'
        '<div class="welcome-sub">Ask about your syllabus, examinations, attendance, '
        'internships or regulations. Answers come from your college documents.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    for row_start in range(0, len(EXAMPLE_QUESTIONS), 2):
        cols = st.columns(2)
        for col, question in zip(cols, EXAMPLE_QUESTIONS[row_start : row_start + 2]):
            with col:
                if st.button(question, key=f"ex_{row_start}_{question[:12]}",
                             use_container_width=True):
                    st.session_state["pending"] = question
                    st.rerun()


def render_plan(plan) -> None:
    """Render the study plan as a clean planning interface."""
    st.markdown('<div class="plan-container">', unsafe_allow_html=True)

    st.markdown('<div class="plan-header">', unsafe_allow_html=True)
    st.markdown(
        '<div class="plan-title">'
        '<span class="plan-title-icon">📅</span>'
        'Your study plan'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="plan-stats">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="plan-stat"><strong>{len(plan.subjects)}</strong> Subjects</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="plan-stat"><strong>{plan.total_hours}</strong> Hours</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="plan-stat"><strong>{plan.total_sessions}</strong> Sessions</div>',
        unsafe_allow_html=True,
    )

    blocked = (
        ", ".join(WEEKDAY_NAMES[d] for d in plan.unavailable_days)
        if plan.unavailable_days
        else "None"
    )
    st.markdown(
        f'<div class="plan-stat"><strong>{blocked}</strong> Not scheduled</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="plan-schedule">', unsafe_allow_html=True)

    rows = [
        {
            "Day": day["weekday"],
            "Date": day["date"],
            "Hours": day["hours"],
            "Sessions": " • ".join(
                f"{s['subject']} ({s['minutes']}m)" for s in day["sessions"]
            ),
        }
        for day in plan.schedule
    ]

    st.markdown('<table class="schedule-table">', unsafe_allow_html=True)
    st.markdown('<thead><tr>', unsafe_allow_html=True)
    st.markdown('<th>Day</th><th>Date</th><th>Hours</th><th>Sessions</th>', unsafe_allow_html=True)
    st.markdown('</tr></thead>', unsafe_allow_html=True)

    for row in rows:
        st.markdown('<tr>', unsafe_allow_html=True)
        st.markdown(f'<td>{row["Day"]}</td><td>{row["Date"]}</td><td>{row["Hours"]}</td><td>{row["Sessions"]}</td>', unsafe_allow_html=True)
        st.markdown('</tr>', unsafe_allow_html=True)

    st.markdown('</table>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_citations(sources: list[dict]) -> None:
    """Citations come from retrieval metadata; the LLM never supplies them."""
    if not sources:
        return

    st.markdown('<div class="sources-container">', unsafe_allow_html=True)
    st.markdown('<div class="sources-title">Sources</div>', unsafe_allow_html=True)

    for source in sources:
        page = source.get("page")
        location = f"Page {page}" if page not in (None, 0, "") else "Page n/a"
        st.markdown(
            f'<div class="source-item">'
            f'<span class="source-name">{source["source"]}</span>'
            f'<span class="source-sep">·</span>'
            f'<span class="source-page">{location}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


def render_history() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources")
            if sources:
                render_citations(sources)
            elif message.get("unavailable"):
                st.markdown(
                    '<div class="unavailable-message">'
                    'I couldn\'t find this information in the available college documents.'
                    '</div>',
                    unsafe_allow_html=True,
                )
        if message.get("plan"):
            render_plan(StudyPlan(**message["plan"]))


def _record_answer(result: dict, answer: str, plan=None) -> None:
    """Persist this turn, including whether it was grounded."""
    entry = {
        "role": "assistant",
        "content": answer,
        "sources": result.get("sources", []) if result.get("grounded") else [],
    }
    if not entry["sources"]:
        # A plan turn is not a knowledge-base answer, but it is still an answer,
        # so it is not labelled as a refusal.
        entry["unavailable"] = result.get("plan") is None
    if plan is not None:
        # Snapshot the dataclass fields only. `StudyPlan.to_dict()` also returns
        # derived values (total_sessions/total_hours) that are not constructor
        # arguments, so it cannot be splatted back into StudyPlan(**...).
        entry["plan"] = {
            "subjects": plan.subjects,
            "days_until_exam": plan.days_until_exam,
            "hours_per_day": plan.hours_per_day,
            "unavailable_days": plan.unavailable_days,
            "session_minutes": plan.session_minutes,
            "schedule": plan.schedule,
            "subject_hours": plan.subject_hours,
        }
    st.session_state["messages"].append(entry)


def answer_question(question: str) -> None:
    """Run the assistant for one turn and render the result."""
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        status = st.status("Searching your college documents...", expanded=False)
        try:
            with st.spinner("Preparing your answer..."):
                result = run_assistant(
                    question=question,
                    history=st.session_state["messages"][:-1],
                    plan=st.session_state["plan"],
                    plan_request=st.session_state["plan_request"],
                )
        except AppError as exc:
            status.update(label="Setup required", state="error")
            st.error(str(exc))
            return
        except Exception:
            logging.exception("Assistant failed")
            status.update(label="Something went wrong", state="error")
            st.error(
                "The assistant could not respond this time. Please try again."
            )
            return

        if result.get("plan") is not None:
            st.session_state["plan"] = result["plan"]
        if result.get("plan_request"):
            st.session_state["plan_request"] = result["plan_request"]

        answer = result.get("answer") or "I could not produce an answer."
        grounded = result.get("grounded", False)
        show_plan = (
            result.get("plan") is not None
            and result.get("intent") in {"PLAN_CREATE", "PLAN_MODIFY"}
        )

        if result.get("error"):
            st.warning(result["error"])

        st.markdown(answer)

        if grounded:
            render_citations(result.get("sources", []))
            status.update(label="Answer ready", state="complete")
        elif result.get("plan") is not None:
            status.update(label="Plan ready", state="complete")
        else:
            status.update(label="Answer ready", state="complete")

        _record_answer(result, answer, plan=result["plan"] if show_plan else None)

    # main() reruns afterwards so the sidebar picks up the new plan; the panel
    # itself is re-rendered from the stored message by render_history().
    if result.get("plan") is not None and result.get("intent") in {
        "PLAN_CREATE", "PLAN_MODIFY"
    }:
        render_plan(result["plan"])


def main() -> None:
    init_state()
    inject_styles()

    status = cached_index_status()
    render_header()

    if not status["index_ok"] or not status["ollama_ok"]:
        st.warning("Setup is incomplete — the assistant cannot answer yet.")
        st.caption("Please check the setup and reload the page.")
        return

    if not st.session_state["messages"]:
        render_welcome()

    render_history()

    pending = st.session_state.pop("pending", None)
    typed = st.chat_input(
        "Ask about your syllabus, exams, attendance, internship, or study plan..."
    )
    question = pending or typed
    if question:
        answer_question(question)
        # Answering mutates session state (a plan may be created or updated), so
        # the sidebar is drawn afterwards. Rendering it earlier would show the
        # previous run's plan, or none on the turn that creates one.
        st.rerun()

    # Drawn last so it reflects any plan produced during this run.
    render_sidebar()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logging.exception("Fatal error in Streamlit app")
        st.error("Something went wrong while processing your request. Please try again.")
        sys.exit(1)