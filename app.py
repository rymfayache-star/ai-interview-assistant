import asyncio
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from openai import OpenAI, OpenAIError

INTERVIEW_SERVER_PATH = Path(__file__).resolve().parent / "interview_server.py"

ROLES = [
    "Python",
    "Data Science",
    "Web Development",
    "Business / Management",
    "HR Interview",
    "Card Payment",
    "Project Management",
    "Business Analysis",
    "Data Analytics",
]

DIFFICULTIES = ["Easy", "Medium", "Hard"]
QUESTION_COUNTS = [1, 2, 3]

INTERVIEWER_INSTRUCTIONS = """
You are an interviewer conducting a professional mock interview.
Ask exactly ONE interview question.
The question must be relevant to the selected role and difficulty.
Do not provide the answer.
Do not ask multiple questions.
Do not repeat previous questions.
Return only the interview question.
""".strip()

EVALUATION_INSTRUCTIONS = """
You are an interviewer evaluating a candidate's answer to one interview question.
Evaluate ONLY the user's answer to the current question.
Do not generate a new interview question.
Do not provide a model answer.
Do not include any text outside JSON.

Return ONLY valid JSON with exactly these fields:
{
  "score": 0,
  "strength": "...",
  "improvement": "...",
  "feedback": "..."
}

Rules:
- score must be an integer between 0 and 10
- strength must be one short strength
- improvement must be one short improvement suggestion
- feedback must be short overall feedback
""".strip()

FINAL_EVALUATION_INSTRUCTIONS = """
You are an interviewer writing a short final evaluation for a completed mock interview.
Use the full set of questions, answers, and individual evaluations.
Do not ask a new question.
Do not include any text outside JSON.

Return ONLY valid JSON with exactly these fields:
{
  "final_score": 0,
  "overall_feedback": "...",
  "top_strengths": ["...", "...", "..."],
  "areas_for_improvement": ["...", "...", "..."]
}

Rules:
- final_score must be an integer between 0 and 10
- overall_feedback must be short
- top_strengths must contain exactly 3 short strengths
- areas_for_improvement must contain exactly 3 short improvements
""".strip()


def init_session_state():
    """Initialize session keys so the app survives Streamlit reruns."""
    defaults = {
        "interview_started": False,
        "interview_finished": False,
        "selected_role": None,
        "selected_difficulty": None,
        "number_of_questions": None,
        "question_number": 0,
        "current_question": None,
        "user_answer": None,
        "questions": [],
        "answers": [],
        "evaluations": [],
        "final_score": None,
        "final_evaluation": None,
        "generation_error": None,
        "evaluation_error": None,
        "final_error": None,
        "generating_question": False,
        "evaluating_answer": False,
        "generating_final_report": False,
        "session_saved": False,
        "save_error": None,
        "save_message": None,
        "viewing_saved_interview": False,
        "selected_saved_interview_id": None,
        "past_interviews": None,
        "past_interviews_error": None,
        "refresh_past_interviews": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment variables, "
            "then restart the application."
        )
    return OpenAI(api_key=api_key)


def parse_json_object(content):
    """Safely extract a JSON object from a model response."""
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("The model did not return valid JSON.")
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError("The model returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("The model JSON must be an object.")
    return data


def parse_score(value):
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("The score must be an integer between 0 and 10.") from exc
    if score < 0 or score > 10:
        raise ValueError("The score must be an integer between 0 and 10.")
    return score


def parse_text_list(value, field_name):
    if isinstance(value, str):
        items = [value.strip()] if value.strip() else []
    elif isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError(f"The field '{field_name}' must be a list of strings.")
    return items[:3]


def parse_evaluation_json(content):
    data = parse_json_object(content)
    required_fields = ("score", "strength", "improvement", "feedback")
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError("The evaluation JSON is missing: " + ", ".join(missing) + ".")

    return {
        "score": parse_score(data["score"]),
        "strength": str(data["strength"]).strip(),
        "improvement": str(data["improvement"]).strip(),
        "feedback": str(data["feedback"]).strip(),
    }


def parse_final_evaluation_json(content):
    data = parse_json_object(content)
    required_fields = (
        "final_score",
        "overall_feedback",
        "top_strengths",
        "areas_for_improvement",
    )
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError("The final evaluation JSON is missing: " + ", ".join(missing) + ".")

    return {
        "final_score": parse_score(data["final_score"]),
        "overall_feedback": str(data["overall_feedback"]).strip(),
        "top_strengths": parse_text_list(data["top_strengths"], "top_strengths"),
        "areas_for_improvement": parse_text_list(
            data["areas_for_improvement"], "areas_for_improvement"
        ),
    }


def generate_interview_question(role, difficulty, question_number, previous_questions):
    """Ask gpt-4o-mini for one new interview question."""
    client = get_openai_client()
    previous = "\n".join(
        f"{index}. {question}" for index, question in enumerate(previous_questions, 1)
    )
    user_content = (
        f"Selected role: {role}\n"
        f"Selected difficulty: {difficulty}\n"
        f"This is question {question_number}.\n"
    )
    if previous:
        user_content += f"Previous questions (do not repeat them):\n{previous}\n"
    user_content += "Generate exactly one new interview question."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": INTERVIEWER_INSTRUCTIONS},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=300,
    )

    question = (response.choices[0].message.content or "").strip()
    if not question:
        raise RuntimeError("The model returned an empty question. Please try again.")
    return question


def evaluate_answer(question, answer):
    """Evaluate only the user's answer to the current question."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EVALUATION_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Interview question:\n{question}\n\n"
                    f"Candidate's answer:\n{answer}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=400,
    )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("The model returned an empty evaluation. Please try again.")
    return parse_evaluation_json(content)


def calculate_final_score(evaluations):
    """Average of all individual question scores, rounded to 1 decimal."""
    scores = [item["score"] for item in evaluations]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 1)


def format_score(score):
    if score == int(score):
        return str(int(score))
    return f"{score:.1f}"


def build_interview_summary():
    parts = []
    for index, (question, answer, evaluation) in enumerate(
        zip(
            st.session_state.questions,
            st.session_state.answers,
            st.session_state.evaluations,
        ),
        1,
    ):
        parts.append(
            f"Question {index}: {question}\n"
            f"Answer: {answer}\n"
            f"Score: {evaluation['score']}/10\n"
            f"Strength: {evaluation['strength']}\n"
            f"Improvement: {evaluation['improvement']}\n"
            f"Feedback: {evaluation['feedback']}"
        )
    return "\n\n".join(parts)


def generate_final_evaluation():
    """Ask OpenAI for a structured final report, using the calculated score."""
    client = get_openai_client()
    final_score = st.session_state.final_score
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": FINAL_EVALUATION_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Role: {st.session_state.selected_role}\n"
                    f"Difficulty: {st.session_state.selected_difficulty}\n"
                    f"Use this calculated average as final_score: {format_score(final_score)}\n\n"
                    f"{build_interview_summary()}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=700,
    )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("The model returned an empty final evaluation. Please try again.")

    result = parse_final_evaluation_json(content)
    result["final_score"] = final_score
    return result


def _mcp_text_result(result) -> str:
    texts = [
        getattr(block, "text", "")
        for block in (result.content or [])
        if getattr(block, "text", "")
    ]
    return "\n".join(texts).strip()


def get_interview_mcp_server_params():
    """Build stdio connection parameters for interview_server.py."""
    env = get_default_environment()
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        env["DATABASE_URL"] = database_url

    return StdioServerParameters(
        command=sys.executable,
        args=[str(INTERVIEW_SERVER_PATH)],
        env=env,
        cwd=str(INTERVIEW_SERVER_PATH.parent),
    )


@asynccontextmanager
async def connect_to_interview_mcp_server():
    """Connect to interview_server.py using the MCP client over stdio transport."""
    server_params = get_interview_mcp_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_mcp_tool(name, arguments=None):
    """Call one Interview Tools MCP method over stdio."""
    async with connect_to_interview_mcp_server() as session:
        result = await session.call_tool(name, arguments=arguments or {})

    message = _mcp_text_result(result)
    if result.isError:
        raise RuntimeError(message or "The MCP tool returned an error.")
    return message


def build_saved_interview_json():
    evaluation = st.session_state.final_evaluation or {}
    return json.dumps(
        {
            "role": st.session_state.selected_role,
            "difficulty": st.session_state.selected_difficulty,
            "number_of_questions": st.session_state.number_of_questions,
            "final_score": st.session_state.final_score,
            "overall_feedback": evaluation.get("overall_feedback", ""),
            "top_strengths": evaluation.get("top_strengths", []),
            "areas_for_improvement": evaluation.get("areas_for_improvement", []),
            "questions": list(st.session_state.questions),
            "answers": list(st.session_state.answers),
            "evaluations": list(st.session_state.evaluations),
        },
        ensure_ascii=False,
    )


def save_completed_interview_via_mcp():
    evaluation = st.session_state.final_evaluation
    return asyncio.run(
        call_mcp_tool(
            "save_interview_session",
            {
                "role": st.session_state.selected_role,
                "difficulty": st.session_state.selected_difficulty,
                "number_of_questions": int(st.session_state.number_of_questions),
                "final_score": float(st.session_state.final_score),
                "overall_feedback": evaluation["overall_feedback"],
                "top_strengths": list(evaluation["top_strengths"]),
                "areas_for_improvement": list(evaluation["areas_for_improvement"]),
                "interview_json": build_saved_interview_json(),
            },
        )
    )


def list_past_interviews_via_mcp():
    payload = asyncio.run(call_mcp_tool("list_interview_sessions", {}))
    interviews = json.loads(payload or "[]")
    if not isinstance(interviews, list):
        raise RuntimeError("The interview list JSON is invalid.")
    return interviews


def load_past_interview_via_mcp(interview_id):
    payload = asyncio.run(
        call_mcp_tool("get_interview_session", {"interview_id": int(interview_id)})
    )
    interview = json.loads(payload or "{}")
    if not isinstance(interview, dict) or "id" not in interview:
        raise RuntimeError("The saved interview JSON is invalid.")
    return interview


def delete_past_interview_via_mcp(interview_id):
    return asyncio.run(
        call_mcp_tool("delete_interview_session", {"interview_id": int(interview_id)})
    )


def build_final_report_text():
    evaluation = st.session_state.final_evaluation or {}
    lines = [
        "AI Interview Assistant - Final Report",
        "",
        f"Role: {st.session_state.selected_role}",
        f"Difficulty: {st.session_state.selected_difficulty}",
        f"Number of questions: {st.session_state.number_of_questions}",
        f"Final Score: {format_score(st.session_state.final_score)} / 10",
        "",
        "Overall Feedback",
        evaluation.get("overall_feedback", ""),
        "",
        "Top 3 Strengths",
    ]
    for item in evaluation.get("top_strengths", []):
        lines.append(f"- {item}")
    lines.extend(["", "Top 3 Areas for Improvement"])
    for item in evaluation.get("areas_for_improvement", []):
        lines.append(f"- {item}")

    lines.extend(["", "Questions, answers and evaluations", ""])
    for index, (question, answer, item) in enumerate(
        zip(
            st.session_state.questions,
            st.session_state.answers,
            st.session_state.evaluations,
        ),
        1,
    ):
        lines.extend(
            [
                f"Question {index}",
                question,
                "",
                "Answer",
                answer,
                "",
                f"Score: {item['score']} / 10",
                f"Strength: {item['strength']}",
                f"Improvement: {item['improvement']}",
                f"Feedback: {item['feedback']}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def start_interview(role, difficulty, number_of_questions):
    st.session_state.interview_started = True
    st.session_state.interview_finished = False
    st.session_state.selected_role = role
    st.session_state.selected_difficulty = difficulty
    st.session_state.number_of_questions = number_of_questions
    st.session_state.question_number = 1
    st.session_state.current_question = None
    st.session_state.user_answer = None
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.evaluations = []
    st.session_state.final_score = None
    st.session_state.final_evaluation = None
    st.session_state.generation_error = None
    st.session_state.evaluation_error = None
    st.session_state.final_error = None
    st.session_state.generating_question = True
    st.session_state.evaluating_answer = False
    st.session_state.generating_final_report = False
    st.session_state.session_saved = False
    st.session_state.save_error = None
    st.session_state.save_message = None
    st.session_state.viewing_saved_interview = False
    st.session_state.selected_saved_interview_id = None


def handle_openai_call(error_key, failure_message, action):
    try:
        result = action()
        st.session_state[error_key] = None
        return result
    except (json.JSONDecodeError, ValueError) as exc:
        st.session_state[error_key] = f"Could not parse the AI response: {exc}"
    except OpenAIError as exc:
        st.session_state[error_key] = f"The OpenAI API request failed: {exc}"
    except Exception as exc:
        st.session_state[error_key] = f"{failure_message}: {exc}"
    return None


def maybe_generate_question():
    if not st.session_state.generating_question:
        return

    label = (
        "Generating the first question..."
        if st.session_state.question_number == 1
        else "Generating the next question..."
    )
    with st.spinner(label):
        question = handle_openai_call(
            "generation_error",
            "Could not generate the interview question",
            lambda: generate_interview_question(
                st.session_state.selected_role,
                st.session_state.selected_difficulty,
                st.session_state.question_number,
                st.session_state.questions,
            ),
        )
        st.session_state.generating_question = False
        if question:
            st.session_state.current_question = question
            st.session_state.user_answer = None


def maybe_evaluate_answer():
    if not st.session_state.evaluating_answer:
        return

    with st.spinner("Evaluating your answer..."):
        evaluation = handle_openai_call(
            "evaluation_error",
            "Could not evaluate the answer",
            lambda: evaluate_answer(
                st.session_state.current_question,
                st.session_state.user_answer,
            ),
        )
        st.session_state.evaluating_answer = False
        if not evaluation:
            return

        st.session_state.questions.append(st.session_state.current_question)
        st.session_state.answers.append(st.session_state.user_answer)
        st.session_state.evaluations.append(evaluation)

        if len(st.session_state.questions) >= st.session_state.number_of_questions:
            st.session_state.interview_finished = True
            st.session_state.current_question = None
            st.session_state.final_score = calculate_final_score(
                st.session_state.evaluations
            )
            st.session_state.generating_final_report = True
        else:
            st.session_state.question_number += 1
            st.session_state.current_question = None
            st.session_state.user_answer = None
            st.session_state.generating_question = True


def maybe_generate_final_report():
    if not st.session_state.generating_final_report:
        return

    with st.spinner("Preparing the final report..."):
        final_evaluation = handle_openai_call(
            "final_error",
            "Could not generate the final evaluation",
            generate_final_evaluation,
        )
        st.session_state.generating_final_report = False
        if final_evaluation:
            st.session_state.final_evaluation = final_evaluation


def maybe_save_interview_session():
    """Save the completed interview through MCP only after the final evaluation."""
    if st.session_state.session_saved:
        return
    if st.session_state.save_error:
        return
    if not st.session_state.interview_finished:
        return
    if not st.session_state.final_evaluation:
        return

    with st.spinner("Saving the interview session..."):
        try:
            st.session_state.save_message = save_completed_interview_via_mcp()
            st.session_state.session_saved = True
            st.session_state.save_error = None
            st.session_state.refresh_past_interviews = True
        except Exception as exc:
            st.session_state.session_saved = False
            st.session_state.save_error = (
                f"Could not save the interview session: {exc}"
            )


def format_interview_date(value):
    if not value:
        return "Unknown date"
    return str(value).split(".")[0]


def apply_saved_interview(interview):
    """Restore a saved interview in the UI without calling OpenAI."""
    st.session_state.viewing_saved_interview = True
    st.session_state.selected_saved_interview_id = interview["id"]
    st.session_state.interview_started = True
    st.session_state.interview_finished = True
    st.session_state.generating_question = False
    st.session_state.evaluating_answer = False
    st.session_state.generating_final_report = False
    st.session_state.generation_error = None
    st.session_state.evaluation_error = None
    st.session_state.final_error = None
    st.session_state.session_saved = True
    st.session_state.save_error = None
    st.session_state.current_question = None
    st.session_state.user_answer = None
    st.session_state.selected_role = interview.get("role")
    st.session_state.selected_difficulty = interview.get("difficulty")
    st.session_state.number_of_questions = interview.get("number_of_questions")
    st.session_state.final_score = interview.get("final_score")
    st.session_state.questions = list(interview.get("questions") or [])
    st.session_state.answers = list(interview.get("answers") or [])
    st.session_state.evaluations = list(interview.get("evaluations") or [])
    st.session_state.final_evaluation = {
        "final_score": interview.get("final_score"),
        "overall_feedback": interview.get("overall_feedback", ""),
        "top_strengths": interview.get("top_strengths") or [],
        "areas_for_improvement": interview.get("areas_for_improvement") or [],
    }


def maybe_load_past_interviews():
    if not st.session_state.refresh_past_interviews and st.session_state.past_interviews is not None:
        return
    try:
        st.session_state.past_interviews = list_past_interviews_via_mcp()
        st.session_state.past_interviews_error = None
    except Exception as exc:
        st.session_state.past_interviews = []
        st.session_state.past_interviews_error = (
            f"Could not load past interviews: {exc}"
        )
    st.session_state.refresh_past_interviews = False


def open_past_interview(interview_id):
    try:
        interview = load_past_interview_via_mcp(interview_id)
        apply_saved_interview(interview)
        st.session_state.past_interviews_error = None
    except Exception as exc:
        st.session_state.past_interviews_error = (
            f"Could not open the saved interview: {exc}"
        )


def delete_past_interview(interview_id):
    try:
        delete_past_interview_via_mcp(interview_id)
        if st.session_state.selected_saved_interview_id == interview_id:
            st.session_state.viewing_saved_interview = False
            st.session_state.selected_saved_interview_id = None
            st.session_state.interview_started = False
            st.session_state.interview_finished = False
        st.session_state.refresh_past_interviews = True
        st.session_state.past_interviews_error = None
    except Exception as exc:
        st.session_state.past_interviews_error = (
            f"Could not delete the interview: {exc}"
        )


def render_past_interviews_sidebar():
    st.header("My past Interviews")
    maybe_load_past_interviews()

    if st.session_state.past_interviews_error:
        st.error(st.session_state.past_interviews_error)

    interviews = st.session_state.past_interviews or []
    if not interviews and not st.session_state.past_interviews_error:
        st.caption("No saved interviews yet.")
        return

    for item in interviews:
        interview_id = item["id"]
        label = f"{item.get('role', 'Interview')} — {format_interview_date(item.get('created_at'))}"
        is_selected = st.session_state.selected_saved_interview_id == interview_id
        if st.button(
            label,
            key=f"open_interview_{interview_id}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            open_past_interview(interview_id)
            st.rerun()
        if st.button(
            "delete interview",
            key=f"delete_interview_{interview_id}",
            use_container_width=True,
        ):
            delete_past_interview(interview_id)
            st.rerun()


def render_header():
    st.markdown(
        """
        <div class="app-header">
            <h1>AI Interview Assistant</h1>
            <p>Practice your interview with an AI interviewer</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.header("Interview Setup")

        role = st.selectbox("Interview type / Role", ROLES)
        difficulty = st.selectbox("Difficulty", DIFFICULTIES)
        number_of_questions = st.selectbox("Number of questions", QUESTION_COUNTS)

        if st.button("Start Interview", type="primary", use_container_width=True):
            start_interview(role, difficulty, number_of_questions)

        st.divider()
        render_past_interviews_sidebar()


def render_empty_state():
    st.markdown(
        """
        <div class="empty-state">
            <p>Your interview will appear here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_completed_rounds():
    history = zip(
        st.session_state.questions,
        st.session_state.answers,
        st.session_state.evaluations,
    )
    for index, (question, answer, evaluation) in enumerate(history, 1):
        with st.expander(
            f"Question {index} of {st.session_state.number_of_questions} — "
            f"Score {evaluation['score']}/10",
            expanded=st.session_state.interview_finished,
        ):
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Your answer:** {answer}")
            st.markdown(f"**Strength:** {evaluation['strength']}")
            st.markdown(f"**Improvement:** {evaluation['improvement']}")
            st.markdown(f"**Feedback:** {evaluation['feedback']}")


def render_answer_form():
    form_key = f"answer_form_{st.session_state.question_number}"
    with st.form(form_key, clear_on_submit=False):
        answer = st.text_area(
            "Your answer",
            height=160,
            placeholder="Type your answer here...",
        )
        submitted = st.form_submit_button("Submit Answer", type="primary")

    if submitted:
        answer = (answer or "").strip()
        if not answer:
            st.warning("Please enter an answer before submitting.")
            return
        st.session_state.user_answer = answer
        st.session_state.evaluation_error = None
        st.session_state.evaluating_answer = True
        st.rerun()


def render_current_question():
    total = st.session_state.number_of_questions
    current = st.session_state.question_number
    st.subheader(f"Question {current} of {total}")
    st.write(st.session_state.current_question)
    render_answer_form()

    if st.session_state.evaluation_error:
        st.error(st.session_state.evaluation_error)


def render_final_report():
    st.success("Interview Completed")
    st.metric("Final Score", f"{format_score(st.session_state.final_score)} / 10")

    if st.session_state.final_error:
        st.error(st.session_state.final_error)
        if st.button("Retry final report"):
            st.session_state.generating_final_report = True
            st.rerun()
        return

    evaluation = st.session_state.final_evaluation
    if not evaluation:
        return

    st.markdown("### Overall Feedback")
    st.write(evaluation["overall_feedback"])

    st.markdown("### Top 3 Strengths")
    for item in evaluation["top_strengths"]:
        st.markdown(f"- {item}")

    st.markdown("### Top 3 Areas for Improvement")
    for item in evaluation["areas_for_improvement"]:
        st.markdown(f"- {item}")

    st.download_button(
        "Download final report",
        data=build_final_report_text(),
        file_name="interview_report.txt",
        mime="text/plain",
    )

    if st.session_state.viewing_saved_interview:
        return

    if st.session_state.save_error:
        st.error(st.session_state.save_error)
        if st.button("Retry save to database"):
            st.session_state.session_saved = False
            st.session_state.save_error = None
            st.rerun()
    elif st.session_state.session_saved:
        st.success(st.session_state.save_message or "Interview session saved.")


def render_saved_interview():
    st.info("Viewing a saved interview. OpenAI is not called.")
    st.markdown(f"**Role:** {st.session_state.selected_role}")
    st.markdown(f"**Difficulty:** {st.session_state.selected_difficulty}")
    render_completed_rounds()
    render_final_report()


def render_interview_area():
    if st.session_state.viewing_saved_interview:
        render_saved_interview()
        return

    maybe_evaluate_answer()
    maybe_generate_question()
    maybe_generate_final_report()
    maybe_save_interview_session()

    st.success("Interview started")
    st.markdown(f"**Role:** {st.session_state.selected_role}")
    st.markdown(f"**Difficulty:** {st.session_state.selected_difficulty}")
    st.markdown(
        f"**Progress:** {len(st.session_state.questions)} / "
        f"{st.session_state.number_of_questions} answered"
    )

    if st.session_state.generation_error and not st.session_state.current_question:
        st.error(st.session_state.generation_error)
        if st.button("Retry question"):
            st.session_state.generating_question = True
            st.rerun()

    render_completed_rounds()

    if st.session_state.interview_finished:
        render_final_report()
        return

    if st.session_state.current_question:
        render_current_question()


def inject_styles():
    st.markdown(
        """
        <style>
            .app-header {
                padding: 0.5rem 0 1.25rem 0;
                margin-bottom: 1.5rem;
                border-bottom: 1px solid rgba(49, 51, 63, 0.15);
            }
            .app-header h1 {
                margin-bottom: 0.35rem;
                font-weight: 700;
                letter-spacing: -0.02em;
            }
            .app-header p {
                margin: 0;
                color: #5c6270;
                font-size: 1.05rem;
            }
            .empty-state {
                margin-top: 1.5rem;
                padding: 2rem 1.5rem;
                border-radius: 12px;
                font-size: 1.1rem;
                line-height: 1.6;
                text-align: center;
                border: 1px dashed rgba(49, 51, 63, 0.25);
                background: #f7f9fc;
                color: #5c6270;
            }
            section[data-testid="stSidebar"] .stButton button {
                margin-top: 0.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="AI Interview Assistant",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    init_session_state()
    render_header()
    render_sidebar()

    if st.session_state.interview_started:
        render_interview_area()
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
