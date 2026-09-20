import json
import os

import psycopg2
import psycopg2.extras

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as FastMCP

mcp = FastMCP("Interview Tools")

TABLE_NAME = "interview"


def get_database_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add the Supabase connection string "
            "to your environment variables, then restart the MCP server."
        )
    return psycopg2.connect(database_url)


def ensure_interview_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS interview (
            id SERIAL PRIMARY KEY,
            role TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            number_of_questions INT NOT NULL,
            final_score FLOAT NOT NULL,
            overall_feedback TEXT,
            top_strengths TEXT,
            areas_for_improvement TEXT,
            interview_json TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """
    )
    cursor.execute(
        "ALTER TABLE interview ADD COLUMN IF NOT EXISTS interview_json TEXT"
    )


def as_json_list(values):
    items = [str(item).strip() for item in (values or []) if str(item).strip()]
    return json.dumps(items[:3], ensure_ascii=False)


def parse_json_field(value, fallback):
    if value in (None, ""):
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def serialize_row(row):
    payload = parse_json_field(row.get("interview_json"), {})
    if not isinstance(payload, dict):
        payload = {}

    created_at = row.get("created_at")
    return {
        "id": row["id"],
        "role": row["role"],
        "difficulty": row["difficulty"],
        "number_of_questions": row["number_of_questions"],
        "final_score": row["final_score"],
        "overall_feedback": row.get("overall_feedback") or payload.get("overall_feedback", ""),
        "top_strengths": parse_json_field(
            row.get("top_strengths"), payload.get("top_strengths", [])
        ),
        "areas_for_improvement": parse_json_field(
            row.get("areas_for_improvement"),
            payload.get("areas_for_improvement", []),
        ),
        "questions": payload.get("questions", []),
        "answers": payload.get("answers", []),
        "evaluations": payload.get("evaluations", []),
        "created_at": created_at.isoformat(sep=" ") if created_at else None,
    }


def build_interview_json(
    role,
    difficulty,
    number_of_questions,
    final_score,
    overall_feedback,
    top_strengths,
    areas_for_improvement,
    interview_json,
):
    payload = parse_json_field(interview_json, {})
    if not isinstance(payload, dict):
        payload = {}
    payload.update(
        {
            "role": role,
            "difficulty": difficulty,
            "number_of_questions": int(number_of_questions),
            "final_score": float(final_score),
            "overall_feedback": overall_feedback,
            "top_strengths": list(top_strengths or []),
            "areas_for_improvement": list(areas_for_improvement or []),
        }
    )
    payload.setdefault("questions", [])
    payload.setdefault("answers", [])
    payload.setdefault("evaluations", [])
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def save_interview_session(
    role: str,
    difficulty: str,
    number_of_questions: int,
    final_score: float,
    overall_feedback: str,
    top_strengths: list,
    areas_for_improvement: list,
    interview_json: str = "",
) -> str:
    """Save a completed interview session to the Supabase PostgreSQL `interview` table.

    Call this tool ONLY after the final interview evaluation is completed.
    Pass interview_json so the full interview can later be displayed
    without calling OpenAI.

    Do not call this tool before the interview starts, after a single
    question evaluation, or before the final report has been generated.
    """
    payload = build_interview_json(
        role,
        difficulty,
        number_of_questions,
        final_score,
        overall_feedback,
        top_strengths,
        areas_for_improvement,
        interview_json,
    )
    connection = get_database_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                ensure_interview_table(cursor)
                cursor.execute(
                    """
                    INSERT INTO interview (
                        role,
                        difficulty,
                        number_of_questions,
                        final_score,
                        overall_feedback,
                        top_strengths,
                        areas_for_improvement,
                        interview_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (
                        role,
                        difficulty,
                        int(number_of_questions),
                        float(final_score),
                        overall_feedback,
                        as_json_list(top_strengths),
                        as_json_list(areas_for_improvement),
                        payload,
                    ),
                )
                row = cursor.fetchone()
    finally:
        connection.close()

    session_id, created_at = row
    return (
        f"Interview session saved in table '{TABLE_NAME}' "
        f"(id={session_id}, created_at={created_at})."
    )


@mcp.tool()
def list_interview_sessions() -> str:
    """List saved interviews from the `interview` table.

    Runs SELECT id, role, created_at, ... FROM interview ORDER BY created_at DESC.
    Use this to show the My past Interviews section. Do not call OpenAI.
    """
    connection = get_database_connection()
    try:
        with connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                ensure_interview_table(cursor)
                cursor.execute(
                    """
                    SELECT id, role, difficulty, number_of_questions,
                           final_score, created_at
                    FROM interview
                    ORDER BY created_at DESC
                    """
                )
                rows = cursor.fetchall()
    finally:
        connection.close()

    interviews = []
    for row in rows:
        created_at = row["created_at"]
        interviews.append(
            {
                "id": row["id"],
                "role": row["role"],
                "difficulty": row["difficulty"],
                "number_of_questions": row["number_of_questions"],
                "final_score": row["final_score"],
                "created_at": created_at.isoformat(sep=" ") if created_at else None,
            }
        )
    return json.dumps(interviews, ensure_ascii=False)


@mcp.tool()
def get_interview_session(interview_id: int) -> str:
    """Load one saved interview JSON by id from the `interview` table.

    Use this when the user clicks a past interview. Display the saved
    interview again without calling OpenAI.
    """
    connection = get_database_connection()
    try:
        with connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                ensure_interview_table(cursor)
                cursor.execute(
                    """
                    SELECT id, role, difficulty, number_of_questions, final_score,
                           overall_feedback, top_strengths, areas_for_improvement,
                           interview_json, created_at
                    FROM interview
                    WHERE id = %s
                    """,
                    (int(interview_id),),
                )
                row = cursor.fetchone()
    finally:
        connection.close()

    if not row:
        raise RuntimeError(f"No interview found with id={interview_id}.")
    return json.dumps(serialize_row(row), ensure_ascii=False)


@mcp.tool()
def delete_interview_session(interview_id: int) -> str:
    """Delete one saved interview from the `interview` table by id.

    Runs DELETE FROM interview WHERE id = interview_id.
    Call this when the user clicks the delete interview button.
    """
    connection = get_database_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                ensure_interview_table(cursor)
                cursor.execute(
                    "DELETE FROM interview WHERE id = %s RETURNING id",
                    (int(interview_id),),
                )
                row = cursor.fetchone()
    finally:
        connection.close()

    if not row:
        raise RuntimeError(f"No interview found with id={interview_id}.")
    return json.dumps({"deleted_id": row[0]})


if __name__ == "__main__":
    mcp.run(transport="stdio")
