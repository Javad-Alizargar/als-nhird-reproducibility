from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


DATASET_RE = re.compile(r"^([A-Za-z]+-\d+(?:-\d+)?)\s+(.+?)(?:\(([^)]*)\))?\s*$")
YEAR_RE = re.compile(r"^(19|20)\d{2}$")

TOTAL_FIELDS_LABEL = "\u7e3d\u6b04\u4f4d\u6578"
FIELD_HEADER_LABEL = "\u6b04\u4f4d\u5e8f\u865f"
MEDICAL_TYPE_LABEL = "::\u91ab\u5225"
WESTERN_MEDICINE = "\u897f\u91ab"
CHINESE_MEDICINE = "\u4e2d\u91ab"
DENTAL_MEDICINE = "\u7259\u91ab"
MEDICAL_TYPE_VALUES = {WESTERN_MEDICINE, CHINESE_MEDICINE, DENTAL_MEDICINE}


@dataclass
class FieldDef:
    number: int
    name: str
    length: str
    description: str
    note: str = ""
    purpose: str = ""


@dataclass
class DatasetDef:
    code: str
    name: str
    order: int
    year_note: str = ""
    total_fields: int | None = None
    years: list[int] = field(default_factory=list)
    fields: list[FieldDef] = field(default_factory=list)
    medical_types: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def label(self) -> str:
        return f"{self.code} {self.name}"


def parse_catalog(path: Path) -> list[DatasetDef]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    datasets: list[DatasetDef] = []
    current: DatasetDef | None = None
    in_fields = False
    in_medical_types = False
    order = 0

    for line in lines:
        stripped = line.strip()
        match = DATASET_RE.match(stripped)
        if match and TOTAL_FIELDS_LABEL not in stripped and FIELD_HEADER_LABEL not in stripped:
            if current:
                datasets.append(current)
            current = DatasetDef(
                code=match.group(1),
                name=match.group(2).strip(),
                year_note=(match.group(3) or "").strip(),
                order=order,
            )
            order += 1
            in_fields = False
            in_medical_types = False
            continue

        if not current:
            continue

        current.raw += line + "\n"

        if stripped.startswith(TOTAL_FIELDS_LABEL):
            nums = re.findall(r"\d+", stripped)
            if nums:
                current.total_fields = int(nums[0])
            continue

        if stripped == MEDICAL_TYPE_LABEL:
            in_medical_types = True
            in_fields = False
            continue

        if in_medical_types:
            if stripped in MEDICAL_TYPE_VALUES and stripped not in current.medical_types:
                current.medical_types.append(stripped)
                continue
            if stripped.startswith(FIELD_HEADER_LABEL) or stripped.startswith("::"):
                in_medical_types = False

        if YEAR_RE.match(stripped):
            year = int(stripped)
            if year not in current.years:
                current.years.append(year)
            continue

        if stripped.startswith(FIELD_HEADER_LABEL):
            in_fields = True
            continue

        if in_fields:
            parts = line.split("\t")
            if len(parts) >= 4 and parts[0].strip().isdigit():
                current.fields.append(
                    FieldDef(
                        number=int(parts[0].strip()),
                        name=parts[1].strip(),
                        length=parts[2].strip(),
                        description=parts[3].strip(),
                        note=parts[4].strip() if len(parts) > 4 else "",
                        purpose=parts[5].strip() if len(parts) > 5 else "",
                    )
                )
            elif stripped.startswith("::") or DATASET_RE.match(stripped):
                in_fields = False

    if current:
        datasets.append(current)

    for ds in datasets:
        ds.years = sorted(ds.years)
        ds.fields = sorted(ds.fields, key=lambda field_def: field_def.number)
    return datasets


def order_datasets(datasets: list[DatasetDef]) -> list[DatasetDef]:
    return sorted(datasets, key=lambda ds: ds.order)


def search_datasets(datasets: list[DatasetDef], query: str, limit: int = 20) -> list[DatasetDef]:
    terms = [term.lower() for term in re.split(r"\s+", query) if term.strip()]
    scored: list[tuple[int, int, DatasetDef]] = []
    for ds in datasets:
        haystack = f"{ds.code} {ds.name} {ds.raw}".lower()
        score = sum(haystack.count(term) for term in terms)
        if score:
            scored.append((score, -ds.order, ds))
    ranked = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)
    return [ds for _, _, ds in ranked[:limit]]


def relevant_fields(ds: DatasetDef, keywords: list[str]) -> list[FieldDef]:
    selected = []
    lowered = [keyword.lower() for keyword in keywords]
    for field_def in ds.fields:
        text = f"{field_def.name} {field_def.description} {field_def.note} {field_def.purpose}".lower()
        if any(keyword in text for keyword in lowered):
            selected.append(field_def)
    return sorted(selected, key=lambda field_def: field_def.number)
