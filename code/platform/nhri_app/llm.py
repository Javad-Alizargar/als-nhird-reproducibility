from __future__ import annotations

from openai import OpenAI


MAX_QA_CONTEXT_CHARS = 12_000
MAX_PLAN_PROJECT_CHARS = 8_000
MAX_PLAN_SKELETON_CHARS = 28_000
MAX_PLAN_CONTEXT_CHARS = 18_000


def limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED to avoid an oversized API request.]"


SYSTEM_PROMPT = """You are an NHRI application planning assistant.
Answer only in the requested language unless the user explicitly asks for bilingual output.
Use only the provided context for NHRI rules, cost, application process, dataset years, and fields.
If evidence is missing, say what must be verified in the official NHRI system.
Be practical: separate applications when linkage identifiers, disease cohorts, or expansion costs make one large application risky.
"""


def answer_with_deepseek(api_key: str, question: str, context: str, model: str = "deepseek-chat") -> str:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    context = limit_text(context, MAX_QA_CONTEXT_CHARS)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


PLAN_PROMPT = """You are preparing NHRI application materials.
Write a polished staged application plan in the requested output language only.

Hard constraints:
- Do not invent datasets, years, variables, costs, or rules.
- Preserve the dataset order and variable order exactly as provided in the deterministic skeleton.
- Treat the skeleton as the source of truth for dataset/variable/year choices.
- Respect rule notes, including per-field-per-year fees, fixed annual-fee datasets, triple-count fields, Health-01/02/03 over-half-field briefing threshold, compatibility restrictions, and 醫別 selectors.
- If Health-01 full visit date or Health-02 full admission date is needed, explicitly state that page 8 "特殊需求申請(選填)" must be marked "是" and "需使用完整日期欄位（含年月日）"; the special fields must be checked and justified.
- For Health-02 APPL_S_DATE 申報期間-起, justify it as cross-month/cross-claim inpatient event continuity and consistency checking with admission date, discharge date, and length of stay. Do not use a vague data-quality-only reason.
- Use the retrieved rules/context only to justify review risk, minimum necessary fields, cost-effectiveness, and application separation.
- If something must be checked in the official NHRI application system, say so clearly.

Output sections:
1. Project interpretation
2. Cost-effective application strategy
3. Phase 1 minimum publishable plan
4. Phase 2 expansion plan
5. Phase 3 policy/cost-effectiveness plan
6. Questions to answer before paying for expanded datasets
7. Application material checklist
"""


def generate_plan_with_deepseek(
    api_key: str,
    project_title: str,
    project_text: str,
    deterministic_skeleton: str,
    retrieved_context: str,
    model: str = "deepseek-chat",
    output_language: str = "English",
) -> str:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    project_text = limit_text(project_text, MAX_PLAN_PROJECT_CHARS)
    deterministic_skeleton = limit_text(deterministic_skeleton, MAX_PLAN_SKELETON_CHARS)
    retrieved_context = limit_text(retrieved_context, MAX_PLAN_CONTEXT_CHARS)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLAN_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Output language: {output_language}. Do not mix languages unless an official dataset field or rule is quoted.\n\n"
                    f"Project title:\n{project_title}\n\n"
                    f"Project description:\n{project_text}\n\n"
                    f"Deterministic NHRI-ordered skeleton:\n{deterministic_skeleton}\n\n"
                    f"Retrieved NHRI rules/context:\n{retrieved_context}"
                ),
            },
        ],
        temperature=0.15,
    )
    return response.choices[0].message.content or ""
