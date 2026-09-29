from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
import re

from .catalog import DatasetDef, FieldDef, order_datasets, relevant_fields, search_datasets
from .review_guidance import special_full_date_purpose
from .rules import compatibility_notes, dataset_processing_fee, dataset_rule_notes, needs_half_field_briefing


CORE_FIELD_KEYWORDS = [
    "id", "birth", "sex", "date", "fee_ym", "icd", "hosp", "drug", "death",
    "diagnosis", "cost", "procedure", "order", "visit", "admission",
    "\u8eab\u5206", "\u51fa\u751f", "\u6027\u5225", "\u65e5\u671f", "\u5e74\u6708",
    "\u8a3a\u65b7", "\u6b7b\u4ea1", "\u91ab\u7642\u6a5f\u69cb", "\u8cbb\u7528",
    "\u9ede\u6578", "\u85e5", "\u8655\u7f6e", "\u624b\u8853", "\u91cd\u5927\u50b7\u75c5",
    "\u985e\u5225", "\u6838\u5b9a", "\u8f49\u6b78", "\u5c31\u91ab", "\u4f4f\u9662",
    "\u9580\u6025\u8a3a",
]

COST_STRATEGIES = {
    "Lean / gated": {
        "field_limits": {1: 10, 2: 14, 3: 18},
        "phase_years": {1: (3, 2022), 2: (5, 2020), 3: (10, 2015)},
        "phase2_extra": False,
        "phase3_extra": True,
    },
    "Balanced": {
        "field_limits": {1: 14, 2: 18, 3: 24},
        "phase_years": {1: (5, 2020), 2: (7, 2018), 3: (15, 2010)},
        "phase2_extra": True,
        "phase3_extra": True,
    },
    "Comprehensive": {
        "field_limits": {1: 14, 2: 22, 3: 30},
        "phase_years": {1: (5, 2020), 2: (10, 2015), 3: (24, 2001)},
        "phase2_extra": True,
        "phase3_extra": True,
    },
}


def cost_strategy_names() -> list[str]:
    return list(COST_STRATEGIES)


def _strategy(name: str) -> dict:
    return COST_STRATEGIES.get(name, COST_STRATEGIES["Lean / gated"])


def field_purpose_reason(phase_name: str, item: "PhaseDataset", field_def: FieldDef) -> str:
    if field_def.purpose:
        return field_def.purpose
    special_purpose = special_full_date_purpose(field_def.name)
    if special_purpose:
        return special_purpose
    if item.dataset.code == "Health-01" and field_def.name == "FUNC_DATE":
        return (
            "本研究需申請完整就醫日期。此欄位用於建立 ALS 與 Wilson’s disease 患者之門急診就醫時間軸，"
            "計算首次相關就醫、首次疑似診斷、確診前後就醫頻率，以及與重大傷病申請/核定日期之時間間隔，"
            "以評估診斷延遲與照護流程。完整日期僅用於安全環境內進行時間間隔與事件順序分析，"
            "不會用於辨識個人，研究結果亦僅以彙整統計呈現。"
        )
    if item.dataset.code == "Health-02" and field_def.name == "IN_DATE":
        return (
            "本研究需申請完整入院日期。此欄位用於建立 ALS 與 Wilson’s disease 患者之住院照護時間軸，"
            "計算首次住院、診斷前後住院頻率、住院與門急診就醫之先後順序，以及與重大傷病申請/核定日期之"
            "時間間隔，以評估疾病嚴重度、診斷延遲、照護流程與醫療資源使用情形。完整日期僅用於安全環境內"
            "進行時間間隔與事件順序分析，不會用於辨識個人，研究結果亦僅以彙整統計呈現。"
        )
    text = f"{field_def.name} {field_def.description}".lower()
    dataset = item.dataset
    if any(token in text for token in ["id", "\u8eab\u5206"]):
        return "\u7528\u65bc\u500b\u6848\u52a0\u5bc6\u4e32\u9023\u3001\u53bb\u91cd\u8907\u8207\u8de8\u6a94\u8ffd\u8e64\uff0c\u4e0d\u7528\u65bc\u8b58\u5225\u7279\u5b9a\u500b\u4eba\u3002"
    if any(token in text for token in ["icd", "\u8a3a\u65b7", "\u50b7\u75c5"]):
        return "\u7528\u65bc\u5b9a\u7fa9\u7814\u7a76\u4e16\u4ee3\u3001\u8fa8\u8b58\u76ee\u6a19\u75be\u75c5\u8207\u4f30\u8a08\u8a3a\u65b7\u5ef6\u9072\u6216\u75c5\u7a0b\u6307\u6a19\u3002"
    if any(token in text for token in ["date", "\u65e5\u671f", "\u5e74\u6708", "ym"]):
        return "\u7528\u65bc\u5efa\u7acb\u5c31\u91ab\u3001\u7533\u5831\u3001\u8a3a\u65b7\u6216\u6b7b\u4ea1\u4e4b\u6642\u9593\u8ef8\uff0c\u4ee5\u652f\u6301\u968e\u6bb5\u6027\u8207\u7e31\u8cab\u5206\u6790\u3002"
    if any(token in text for token in ["sex", "\u6027\u5225", "birth", "\u51fa\u751f", "\u5e74\u9f61", "age"]):
        return "\u7528\u65bc\u63cf\u8ff0\u7814\u7a76\u5c0d\u8c61\u7279\u5fb5\u3001\u98a8\u96aa\u8abf\u6574\u8207\u5206\u5c64\u5206\u6790\u3002"
    if any(token in text for token in ["hosp", "\u91ab\u7642\u6a5f\u69cb", "\u91ab\u4e8b\u6a5f\u69cb"]):
        return "\u7528\u65bc\u8a55\u4f30\u5c31\u91ab\u5834\u57df\u3001\u8f49\u8a3a\u8def\u5f91\u3001\u7167\u8b77\u9023\u7e8c\u6027\u8207\u8de8\u9662\u6240\u5229\u7528\u60c5\u5f62\u3002"
    if any(token in text for token in ["drug", "\u85e5", "\u7528\u85e5"]):
        return "\u7528\u65bc\u8a55\u4f30\u6cbb\u7642\u5167\u5bb9\u3001\u7528\u85e5\u53ef\u8fd1\u6027\u3001\u6cbb\u7642\u6301\u7e8c\u6027\u8207\u91ab\u7642\u6210\u672c\u6548\u76ca\u3002"
    if any(token in text for token in ["op_", "\u624b\u8853", "\u8655\u7f6e", "\u91ab\u4ee4", "order"]):
        return "\u7528\u65bc\u8fa8\u8b58\u6cbb\u7642\u8655\u7f6e\u3001\u91ab\u4ee4\u5167\u5bb9\u8207\u7167\u8b77\u6d41\u7a0b\u6307\u6a19\u3002"
    if any(token in text for token in ["dot", "\u9ede\u6578", "\u8cbb\u7528", "cost"]):
        return "\u7528\u65bc\u4f30\u7b97\u91ab\u7642\u5229\u7528\u3001\u8cc7\u6e90\u8017\u7528\u8207\u6210\u672c\u6548\u76ca\u5206\u6790\u3002"
    if any(token in text for token in ["death", "\u6b7b\u4ea1", "\u6b7b\u56e0"]):
        return "\u7528\u65bc\u8a55\u4f30\u6b7b\u4ea1\u7d50\u679c\u3001\u5b58\u6d3b\u5206\u6790\u8207\u653f\u7b56\u6210\u6548\u6307\u6a19\u3002"
    return f"\u7528\u65bc {phase_name} \u4e4b {dataset.code} \u5206\u6790\uff0c\u652f\u6301\u7814\u7a76\u76ee\u7684\u3001\u6700\u5c0f\u5fc5\u8981\u6b04\u4f4d\u539f\u5247\u8207\u7533\u8acb\u5408\u7406\u6027\u8aaa\u660e\u3002"


@dataclass
class PhaseDataset:
    dataset: DatasetDef
    years: list[int] | str
    fields: list[FieldDef]
    rationale_en: str
    rationale_zh: str
    medical_type: str = ""

    @property
    def field_count(self) -> int:
        return len(self.fields) or min(self.dataset.total_fields or 0, 8)

    @property
    def year_count(self) -> int:
        if isinstance(self.years, str):
            return 1
        return max(1, len(self.years))


@dataclass
class PhasePlan:
    name_en: str
    name_zh: str
    application_strategy_en: str
    application_strategy_zh: str
    datasets: list[PhaseDataset]
    decision_gate_en: str
    decision_gate_zh: str

    @property
    def estimated_cost(self) -> int:
        file_count = len(self.datasets)
        return sum(
            dataset_processing_fee(item.dataset, item.fields, item.year_count, file_count)
            for item in self.datasets
        )

    @property
    def rule_notes(self) -> list[str]:
        notes: list[str] = []
        for item in self.datasets:
            for note in dataset_rule_notes(item.dataset, item.fields):
                notes.append(f"{item.dataset.code}: {note}")
        notes.extend(compatibility_notes([item.dataset for item in self.datasets]))
        return notes


def _by_code(datasets: list[DatasetDef], code: str) -> DatasetDef | None:
    for ds in datasets:
        if ds.code.lower() == code.lower():
            return ds
    return None


def _find(datasets: list[DatasetDef], *needles: str) -> DatasetDef | None:
    lower_needles = [n.lower() for n in needles]
    for ds in datasets:
        text = f"{ds.code} {ds.name}".lower()
        if any(needle in text for needle in lower_needles):
            return ds
    return None


def _last_years(ds: DatasetDef, n: int, start: int | None = None) -> list[int] | str:
    if not ds.years:
        return "\u4e0d\u5206\u5e74\u5ea6"
    years = ds.years
    if start:
        years = [year for year in years if year >= start]
    return years[-n:]


def _field_limit(ds: DatasetDef, phase_level: int, cost_strategy: str) -> int:
    if not ds.fields:
        return 0
    base = _strategy(cost_strategy)["field_limits"].get(phase_level, 16)
    if ds.code in {"Health-01", "Health-02", "Health-03"} and ds.total_fields:
        # Keep below the "over half fields requires briefing" threshold unless a human explicitly decides otherwise.
        base = min(base, max(1, ds.total_fields // 2))
    return min(base, len(ds.fields))


def _fields(ds: DatasetDef, extra_keywords: list[str] | None = None, phase_level: int = 1, cost_strategy: str = "Lean / gated") -> list[FieldDef]:
    limit = _field_limit(ds, phase_level, cost_strategy)
    fields = relevant_fields(ds, CORE_FIELD_KEYWORDS + (extra_keywords or []))
    if len(fields) < limit:
        seen = {field.number for field in fields}
        for field_def in ds.fields:
            if field_def.number not in seen:
                fields.append(field_def)
                seen.add(field_def.number)
            if len(fields) >= limit:
                break
    return sorted(fields[:limit], key=lambda field_def: field_def.number)


def _medical_type(project_text: str, ds: DatasetDef) -> str:
    if not ds.medical_types:
        return ""
    text = project_text.lower()
    if "牙" in text or "dental" in text:
        return "牙醫" if "牙醫" in ds.medical_types else ds.medical_types[0]
    if "中醫" in text or "chinese medicine" in text or "traditional chinese" in text:
        return "中醫" if "中醫" in ds.medical_types else ds.medical_types[0]
    return "西醫" if "西醫" in ds.medical_types else ds.medical_types[0]


def _wrap(
    source_list: list[DatasetDef],
    year_n: int,
    start: int | None,
    purpose: str,
    keywords: list[str],
    project_text: str,
    phase_level: int,
    cost_strategy: str,
) -> list[PhaseDataset]:
    wrapped: list[PhaseDataset] = []
    for ds in order_datasets(source_list):
        wrapped.append(
            PhaseDataset(
                dataset=ds,
                years=_last_years(ds, year_n, start),
                fields=_fields(ds, keywords, phase_level=phase_level, cost_strategy=cost_strategy),
                rationale_en=(
                    f"Cost-aware useful field pack for {purpose}: include all plausibly useful core fields, "
                    "but avoid unnecessary fields because most files are billed by selected field x year."
                ),
                rationale_zh=(
                    f"\u7528\u65bc {purpose} \u4e4b\u6210\u672c\u6548\u76ca\u6b04\u4f4d\u7d44\uff1a"
                    "\u7d0d\u5165\u53ef\u80fd\u6709\u7528\u7684\u6838\u5fc3\u6b04\u4f4d\uff0c"
                    "\u4f46\u907f\u514d\u4e0d\u5fc5\u8981\u6b04\u4f4d\uff0c\u56e0\u591a\u6578\u6a94\u6848\u4f9d\u6b04\u4f4d x \u5e74\u5ea6\u8a08\u8cbb\u3002"
                ),
                medical_type=_medical_type(project_text, ds),
            )
        )
    return wrapped


def _project_keywords(project_text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", project_text)
    stop = {"study", "data", "policy", "taiwan", "analysis", "project", "\u7814\u7a76", "\u8a08\u756b", "\u8cc7\u6599", "\u5206\u6790", "\u53f0\u7063", "\u63d0\u5347"}
    return [word for word in words if word.lower() not in stop][:30]


def build_rare_disease_plan(datasets: list[DatasetDef], project_text: str = "", cost_strategy: str = "Lean / gated") -> list[PhasePlan]:
    outpatient = _by_code(datasets, "Health-01")
    inpatient = _by_code(datasets, "Health-02")
    pharmacy = _by_code(datasets, "Health-03")
    outpatient_orders = _by_code(datasets, "Health-04")
    inpatient_orders = _by_code(datasets, "Health-05")
    enrollment = _by_code(datasets, "Health-07")
    catastrophic = _by_code(datasets, "Health-08")
    death = _by_code(datasets, "Health-10")
    registry = _by_code(datasets, "Health-51") or _find(datasets, "\u7f55\u898b\u75be\u75c5")

    strategy = _strategy(cost_strategy)
    y1, s1 = strategy["phase_years"][1]
    y2, s2 = strategy["phase_years"][2]
    y3, s3 = strategy["phase_years"][3]

    phase1_sources = [ds for ds in [outpatient, inpatient, catastrophic, death, registry] if ds]
    if strategy["phase2_extra"]:
        phase2_sources = [ds for ds in [outpatient, inpatient, pharmacy, outpatient_orders, inpatient_orders, catastrophic, death, registry] if ds]
    else:
        phase2_sources = [ds for ds in [outpatient, inpatient, catastrophic, death, registry] if ds]
    phase3_optional = [pharmacy, outpatient_orders, inpatient_orders, enrollment] if strategy["phase3_extra"] else []
    phase3_sources = [ds for ds in [outpatient, inpatient, *phase3_optional, catastrophic, death, registry] if ds]
    keywords = ["ALS", "Wilson", "\u7f55\u898b", "\u91cd\u5927\u50b7\u75c5", "ICD10", "\u8cbb\u7528", "\u8f49\u8a3a", "\u8ffd\u8e64"]

    return [
        PhasePlan(
            name_en="Phase 1 - Minimum Publishable Feasibility",
            name_zh="\u7b2c\u4e00\u968e\u6bb5 - \u6700\u5c0f\u53ef\u767c\u8868\u53ef\u884c\u6027\u5206\u6790",
            application_strategy_en="Submit the smallest separate application that can verify prevalence, cohort size, and linkage feasibility.",
            application_strategy_zh="\u4ee5\u6700\u5c0f\u7bc4\u570d\u7368\u7acb\u7533\u8acb\uff0c\u5148\u78ba\u8a8d\u76db\u884c\u7387\u3001\u500b\u6848\u6578\u8207\u4e32\u9023\u53ef\u884c\u6027\u3002",
            datasets=_wrap(phase1_sources, y1, s1, "prevalence, case validation, diagnostic delay, and mortality baseline", keywords, project_text, 1, cost_strategy),
            decision_gate_en="Continue only if cohort counts, event dates, and linkage quality support stable estimates.",
            decision_gate_zh="\u50c5\u5728\u500b\u6848\u6578\u3001\u4e8b\u4ef6\u65e5\u671f\u8207\u4e32\u9023\u54c1\u8cea\u8db3\u4ee5\u7522\u751f\u7a69\u5b9a\u4f30\u8a08\u6642\u9032\u5165\u4e0b\u4e00\u968e\u6bb5\u3002",
        ),
        PhasePlan(
            name_en="Phase 2 - Care Pathway and Service Use",
            name_zh="\u7b2c\u4e8c\u968e\u6bb5 - \u7167\u8b77\u6d41\u7a0b\u8207\u91ab\u7642\u5229\u7528",
            application_strategy_en="Use a gated expansion. In lean mode, keep Phase 2 to confirmed cohort, utilization, and outcomes; add pharmacy/orders only if Phase 1 shows enough cases and a treatment question that changes decisions.",
            application_strategy_zh="\u63a1\u968e\u6bb5\u95dc\u5361\u64f4\u5145\u3002\u7cbe\u7c21\u6a21\u5f0f\u4e0b\uff0c\u7b2c\u4e8c\u968e\u6bb5\u5148\u4fdd\u7559\u78ba\u8a8d\u4e16\u4ee3\u3001\u91ab\u7642\u5229\u7528\u8207\u7d50\u679c\uff1b\u53ea\u5728\u7b2c\u4e00\u968e\u6bb5\u8b49\u5be6\u500b\u6848\u6578\u8db3\u5920\u4e14\u7528\u85e5/\u91ab\u4ee4\u554f\u984c\u6703\u6539\u8b8a\u6c7a\u7b56\u6642\u518d\u52a0\u5165\u3002",
            datasets=_wrap(phase2_sources, y2, s2, "referral delay, treatment pathway, follow-up, and utilization", keywords, project_text, 2, cost_strategy),
            decision_gate_en="Continue if pathway indicators are measurable and intervention targets are clear.",
            decision_gate_zh="\u82e5\u8f49\u8a3a\u3001\u8ffd\u8e64\u8207\u6cbb\u7642\u8def\u5f91\u6307\u6a19\u53ef\u91cf\u6e2c\u4e14\u4ecb\u5165\u76ee\u6a19\u660e\u78ba\uff0c\u518d\u9032\u5165\u7b2c\u4e09\u968e\u6bb5\u3002",
        ),
        PhasePlan(
            name_en="Phase 3 - Policy Translation and Cost-Effectiveness",
            name_zh="\u7b2c\u4e09\u968e\u6bb5 - \u653f\u7b56\u8f49\u8b6f\u8207\u6210\u672c\u6548\u76ca",
            application_strategy_en="Expand years and add registry/enrollment context only for cost-effectiveness and policy simulation outputs.",
            application_strategy_zh="\u50c5\u70ba\u6210\u672c\u6548\u76ca\u3001\u653f\u7b56\u6a21\u64ec\u8207\u767d\u76ae\u66f8\u6210\u679c\uff0c\u624d\u64f4\u5145\u5e74\u5ea6\u4e26\u52a0\u5165\u901a\u5831\u6216\u627f\u4fdd\u80cc\u666f\u8cc7\u6599\u3002",
            datasets=_wrap(phase3_sources, y3, s3, "longitudinal outcomes, cost-effectiveness, and policy simulation", keywords, project_text, 3, cost_strategy),
            decision_gate_en="Stop expanding if added datasets do not change policy conclusions enough to justify cost.",
            decision_gate_zh="\u82e5\u65b0\u589e\u8cc7\u6599\u7121\u6cd5\u660e\u986f\u6539\u8b8a\u653f\u7b56\u7d50\u8ad6\uff0c\u61c9\u505c\u6b62\u64f4\u5145\u4ee5\u63a7\u5236\u6210\u672c\u3002",
        ),
    ]


def build_project_plan(project_title: str, project_text: str, datasets: list[DatasetDef], cost_strategy: str = "Lean / gated") -> list[PhasePlan]:
    text = f"{project_title}\n{project_text}".lower()
    if any(term in text for term in ["als", "wilson", "\u7f55\u898b", "\u808c\u840e\u7e2e", "\u5a01\u723e\u68ee"]):
        return build_rare_disease_plan(datasets, project_text, cost_strategy)

    base_codes = ["Health-01", "Health-02", "Health-08", "Health-10"]
    if any(term in text for term in ["drug", "pharmacy", "medication", "\u85e5", "\u7528\u85e5"]):
        base_codes.extend(["Health-03", "Health-04", "Health-05"])
    if any(term in text for term in ["cost", "utilization", "\u8cbb\u7528", "\u6210\u672c", "\u91ab\u7642\u5229\u7528"]):
        base_codes.extend(["Health-04", "Health-05", "Health-07"])

    selected = [ds for code in base_codes if (ds := _by_code(datasets, code))]
    query_matches = search_datasets(datasets, " ".join(_project_keywords(project_text)), limit=6)
    for ds in query_matches:
        if ds not in selected and len(selected) < 9:
            selected.append(ds)

    selected = order_datasets(selected)
    keywords = _project_keywords(project_text)
    strategy = _strategy(cost_strategy)
    y1, s1 = strategy["phase_years"][1]
    y2, s2 = strategy["phase_years"][2]
    y3, s3 = strategy["phase_years"][3]
    return [
        PhasePlan(
            name_en="Phase 1 - Minimum Publishable Feasibility",
            name_zh="\u7b2c\u4e00\u968e\u6bb5 - \u6700\u5c0f\u53ef\u767c\u8868\u53ef\u884c\u6027\u5206\u6790",
            application_strategy_en="Start with the smallest application that can define the cohort and answer one publishable baseline question.",
            application_strategy_zh="\u5148\u4ee5\u6700\u5c0f\u7533\u8acb\u7bc4\u570d\u5b9a\u7fa9\u7814\u7a76\u4e16\u4ee3\uff0c\u56de\u7b54\u4e00\u500b\u53ef\u767c\u8868\u7684\u57fa\u7dda\u554f\u984c\u3002",
            datasets=_wrap(selected[:4], y1, s1, "cohort definition and baseline feasibility", keywords, project_text, 1, cost_strategy),
            decision_gate_en="Expand only if case count, variable completeness, and linkage feasibility are adequate.",
            decision_gate_zh="\u50c5\u5728\u500b\u6848\u6578\u3001\u6b04\u4f4d\u5b8c\u6574\u6027\u8207\u4e32\u9023\u53ef\u884c\u6027\u8db3\u5920\u6642\u64f4\u5145\u3002",
        ),
        PhasePlan(
            name_en="Phase 2 - Mechanism and Care Pathway",
            name_zh="\u7b2c\u4e8c\u968e\u6bb5 - \u6a5f\u8f49\u3001\u7167\u8b77\u6d41\u7a0b\u8207\u5229\u7528",
            application_strategy_en="Add detailed utilization or mechanism datasets after Phase 1 proves the cohort is worth expanding.",
            application_strategy_zh="\u7b2c\u4e00\u968e\u6bb5\u78ba\u8a8d\u4e16\u4ee3\u503c\u5f97\u64f4\u5145\u5f8c\uff0c\u518d\u52a0\u5165\u8f03\u7d30\u7684\u5229\u7528\u3001\u91ab\u4ee4\u6216\u6a5f\u8f49\u8cc7\u6599\u3002",
            datasets=_wrap(selected[:7], y2, s2, "care pathway and service-use analysis", keywords, project_text, 2, cost_strategy),
            decision_gate_en="Continue if the added variables materially improve interpretation or intervention design.",
            decision_gate_zh="\u82e5\u65b0\u589e\u6b04\u4f4d\u80fd\u5be6\u8cea\u6539\u5584\u89e3\u91cb\u6216\u4ecb\u5165\u8a2d\u8a08\uff0c\u518d\u9032\u5165\u4e0b\u4e00\u968e\u6bb5\u3002",
        ),
        PhasePlan(
            name_en="Phase 3 - Translation and Robustness",
            name_zh="\u7b2c\u4e09\u968e\u6bb5 - \u8f49\u8b6f\u3001\u654f\u611f\u5ea6\u5206\u6790\u8207\u653f\u7b56\u61c9\u7528",
            application_strategy_en="Use the broadest application only for policy translation, robustness checks, and final deliverables.",
            application_strategy_zh="\u6700\u5927\u7bc4\u570d\u7533\u8acb\u50c5\u7528\u65bc\u653f\u7b56\u8f49\u8b6f\u3001\u654f\u611f\u5ea6\u5206\u6790\u8207\u6700\u7d42\u6210\u679c\u3002",
            datasets=_wrap(selected, y3, s3, "longitudinal robustness and policy translation", keywords, project_text, 3, cost_strategy),
            decision_gate_en="Do not add more datasets unless they change decisions, precision, or policy usefulness.",
            decision_gate_zh="\u9664\u975e\u65b0\u589e\u8cc7\u6599\u6703\u6539\u8b8a\u6c7a\u7b56\u3001\u7cbe\u6e96\u5ea6\u6216\u653f\u7b56\u7528\u9014\uff0c\u5426\u5247\u4e0d\u518d\u64f4\u5145\u3002",
        ),
    ]


def plan_to_markdown(phases: list[PhasePlan]) -> str:
    lines: list[str] = []
    for phase in phases:
        lines.append(f"## {phase.name_en} / {phase.name_zh}")
        lines.append(f"Estimated processing fee: NT${phase.estimated_cost:,}")
        lines.append(f"Strategy EN: {phase.application_strategy_en}")
        lines.append(f"Strategy ZH: {phase.application_strategy_zh}")
        for item in sorted(phase.datasets, key=lambda phase_dataset: phase_dataset.dataset.order):
            years = item.years if isinstance(item.years, str) else ", ".join(map(str, item.years))
            lines.append(f"- Dataset: {item.dataset.label}")
            lines.append(f"  Years: {years}")
            if item.medical_type:
                lines.append(f"  Medical type selection: {item.medical_type}")
            lines.append("  Variables in original NHRI order:")
            for field in sorted(item.fields, key=lambda field_def: field_def.number):
                note = f" ({field.note})" if field.note else ""
                reason = field_purpose_reason(phase.name_zh, item, field)
                lines.append(f"  - {field.number}. {field.name}: {field.description}{note}")
                lines.append(f"    資料欄位需求之緣由或目的: {reason}")
            for note in dataset_rule_notes(item.dataset, item.fields):
                lines.append(f"  Rule note: {note}")
        lines.append(f"Decision gate EN: {phase.decision_gate_en}")
        lines.append(f"Decision gate ZH: {phase.decision_gate_zh}")
        if phase.rule_notes:
            lines.append("Phase rule checks:")
            for note in phase.rule_notes:
                lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def plan_to_html(project_title: str, project_text: str, phases: list[PhasePlan], output_dir: Path, ai_report: str = "") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if ai_report.strip():
        rows.append("<h2>DeepSeek Draft Application Narrative</h2>")
        rows.append(f"<div class='project'>{escape(ai_report)}</div>")
    for phase in phases:
        rows.append(f"<h2>{escape(phase.name_en)}<br>{escape(phase.name_zh)}</h2>")
        rows.append(f"<p><b>Strategy:</b> {escape(phase.application_strategy_en)}<br>{escape(phase.application_strategy_zh)}</p>")
        rows.append(f"<p><b>Estimated processing fee:</b> NT${phase.estimated_cost:,} (rough estimate from selected fields x years)</p>")
        rows.append("<table><thead><tr><th>Dataset</th><th>Years / Selectors</th><th>Variables in NHRI Order</th><th>Justification / Rule Checks</th></tr></thead><tbody>")
        for item in sorted(phase.datasets, key=lambda phase_dataset: phase_dataset.dataset.order):
            field_rows = []
            for field in sorted(item.fields, key=lambda field_def: field_def.number):
                note = f" ({field.note})" if field.note else ""
                reason = field_purpose_reason(phase.name_zh, item, field)
                field_rows.append(
                    "<tr>"
                    f"<td>{field.number}</td>"
                    f"<td>{escape(field.name)}</td>"
                    f"<td>{escape(field.description + note)}</td>"
                    f"<td>{escape(reason)}</td>"
                    "</tr>"
                )
            field_text = (
                "<table><thead><tr><th>No.</th><th>Variable</th><th>Description</th>"
                "<th>資料欄位需求之緣由或目的</th></tr></thead><tbody>"
                + "".join(field_rows)
                + "</tbody></table>"
            ) if field_rows else "To be finalized in NHRI field table"
            years = item.years if isinstance(item.years, str) else ", ".join(map(str, item.years))
            selector = f"<br><b>醫別:</b> {escape(item.medical_type)}" if item.medical_type else ""
            rule_text = "<br>".join(escape(note) for note in dataset_rule_notes(item.dataset, item.fields))
            warning = " Briefing threshold reached." if needs_half_field_briefing(item.dataset, item.fields) else ""
            rows.append(
                "<tr>"
                f"<td>{escape(item.dataset.label)}</td>"
                f"<td>{escape(str(years))}{selector}</td>"
                f"<td>{field_text}</td>"
                f"<td>{escape(item.rationale_en)}<br>{escape(item.rationale_zh)}{escape(warning)}<br>{rule_text}</td>"
                "</tr>"
            )
        rows.append("</tbody></table>")
        rows.append(f"<p><b>Decision gate:</b> {escape(phase.decision_gate_en)}<br>{escape(phase.decision_gate_zh)}</p>")
        if phase.rule_notes:
            rows.append("<ul>")
            for note in phase.rule_notes:
                rows.append(f"<li>{escape(note)}</li>")
            rows.append("</ul>")

    safe_title = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", project_title).strip("_")[:80] or "nhri_project"
    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>{escape(project_title)} - NHRI staged application plan</title>
<style>
body {{ font-family: Arial, "Microsoft JhengHei", sans-serif; margin: 36px; line-height: 1.55; color: #18212f; }}
h1 {{ color: #123b5d; }}
h2 {{ margin-top: 32px; border-bottom: 2px solid #d7e3ef; padding-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 14px 0 24px; font-size: 14px; }}
th, td {{ border: 1px solid #ccd7e2; padding: 8px; vertical-align: top; }}
th {{ background: #eef5fb; }}
.project {{ white-space: pre-wrap; background: #f8fafc; padding: 14px; border-left: 4px solid #2f6f9f; }}
</style>
</head>
<body>
<h1>{escape(project_title)}</h1>
<p>Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
<div class="project">{escape(project_text)}</div>
{''.join(rows)}
</body></html>"""
    path = output_dir / f"{safe_title}_nhri_staged_application_plan.html"
    path.write_text(html, encoding="utf-8")
    return path
