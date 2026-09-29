from __future__ import annotations

from .catalog import DatasetDef, FieldDef


STANDARD_FIELD_FEE = 210
HIGH_FILE_COUNT_FIELD_FEE = 240
FIXED_ANNUAL_FEE = 4200
FIXED_ANNUAL_CODES = {
    "Health-12",
    "Health-13",
    "Health-48",
    "Health-49",
    "Society-10",
    "Society-12",
    "Welfare-4",
    "Welfare-5",
    "Welfare-6",
    "Welfare-7",
    "Welfare-8",
}
HALF_FIELD_BRIEFING_CODES = {"Health-01", "Health-02", "Health-03"}


def field_billing_units(dataset: DatasetDef, field_def: FieldDef) -> int:
    if dataset.code == "Health-30" and field_def.number == 15:
        return 3
    if dataset.code == "Health-09" and 25 <= field_def.number <= 30:
        return 3
    return 1


def billable_field_units(dataset: DatasetDef, fields: list[FieldDef]) -> int:
    return sum(field_billing_units(dataset, field_def) for field_def in fields)


def dataset_processing_fee(dataset: DatasetDef, fields: list[FieldDef], year_count: int, file_count: int) -> int:
    if dataset.code in FIXED_ANNUAL_CODES:
        return year_count * FIXED_ANNUAL_FEE
    unit = HIGH_FILE_COUNT_FIELD_FEE if file_count >= 12 else STANDARD_FIELD_FEE
    return billable_field_units(dataset, fields) * year_count * unit


def selected_field_ratio(dataset: DatasetDef, fields: list[FieldDef]) -> float:
    if not dataset.total_fields:
        return 0.0
    return len(fields) / dataset.total_fields


def needs_half_field_briefing(dataset: DatasetDef, fields: list[FieldDef]) -> bool:
    return dataset.code in HALF_FIELD_BRIEFING_CODES and selected_field_ratio(dataset, fields) > 0.5


def dataset_rule_notes(dataset: DatasetDef, fields: list[FieldDef]) -> list[str]:
    notes: list[str] = []
    if dataset.code in FIXED_ANNUAL_CODES:
        notes.append("Fee rule: this dataset is charged NT$4,200 per year in the application note.")
    elif fields:
        notes.append("Fee rule: standard health/social welfare files are charged by selected field x selected year.")
    if needs_half_field_briefing(dataset, fields):
        notes.append("Review/process rule: Health-01/02/03 uses more than half of fields; briefing at the center is required.")
    if dataset.code == "Health-09" and any(25 <= field_def.number <= 30 for field_def in fields):
        notes.append("Billing rule: Health-09 fields 25-30 count as 3 fields each.")
    if dataset.code == "Health-30" and any(field_def.number == 15 for field_def in fields):
        notes.append("Billing rule: Health-30 field 15 counts as 3 fields.")
    if "需提供" in dataset.name or "授權" in dataset.name:
        notes.append("Authorization rule: this file name indicates additional provider authorization documents may be required.")
    if dataset.medical_types:
        notes.append(f"Medical-type selector available: {', '.join(dataset.medical_types)}.")
    return notes


def compatibility_notes(datasets: list[DatasetDef]) -> list[str]:
    codes = {dataset.code for dataset in datasets}
    notes: list[str] = []
    if "Society-10" in codes:
        notes.append("Society-10 years 2001, 2005, 2009, 2013, 2017 may only be linked with NHI files and cause-of-death files, not other files in the same application.")
    if "Health-48" in codes:
        notes.append("Health-48 may only be linked with NHI years from the survey year onward; 94-106, 111, 112, 113 survey years require separate applications when NHI data are used.")
    if "Society-12" in codes:
        notes.append("Society-12 and Society-12-1 may only be linked with cause-of-death and specific NHI years, not other files in the same application.")
    if "Society-17" in codes:
        notes.append("Society-17 may only be linked with NHI files, cancer registry, and cause-of-death files, not other files in the same application.")
    return notes
