from __future__ import annotations

from dataclasses import dataclass


PURPOSE_COL = "資料欄位需求之緣由或目的"


@dataclass(frozen=True)
class SpecialFieldNeed:
    dataset_code: str
    dataset_name: str
    field_number: int
    field_name: str
    field_description: str
    purpose: str
    reviewer_note: str = ""

    @property
    def label(self) -> str:
        return f"{self.dataset_code} 欄位{self.field_number} {self.field_name} {self.field_description}"


SPECIAL_FULL_DATE_NEEDS: list[SpecialFieldNeed] = [
    SpecialFieldNeed(
        dataset_code="Health-01",
        dataset_name="全民健保處方及治療明細檔_門急診",
        field_number=52,
        field_name="FUNC_DATE_OLD",
        field_description="就醫日期",
        purpose=(
            "本研究需申請完整就醫日期，用於建立 ALS 與 Wilson’s disease 患者之門急診就醫時間軸，"
            "計算首次相關就醫、首次疑似診斷、確診前後就醫頻率，以及與重大傷病申請/核定日期之時間間隔，"
            "以評估診斷延遲與照護流程；完整日期僅於安全環境內進行時間間隔與事件順序分析，"
            "研究結果僅以彙整統計呈現。"
        ),
        reviewer_note="第八頁特殊需求需勾選「是」與「需使用完整日期欄位（含年月日）」。",
    ),
    SpecialFieldNeed(
        dataset_code="Health-02",
        dataset_name="全民健保處方及治療明細檔_住院",
        field_number=81,
        field_name="OUT_DATE",
        field_description="出院年月日",
        purpose=(
            "本研究需申請完整出院日期，用於計算 ALS 與 Wilson’s disease 患者每次住院之住院天數、"
            "出院後再就醫或再住院時間間隔，以及住院照護結束時間點，以評估疾病嚴重度、"
            "照護連續性與醫療資源使用情形；完整日期僅用於時間間隔分析，不會用於辨識個人。"
        ),
    ),
    SpecialFieldNeed(
        dataset_code="Health-02",
        dataset_name="全民健保處方及治療明細檔_住院",
        field_number=82,
        field_name="APPL_S_DATE",
        field_description="申報期間-起",
        purpose=(
            "本研究需申請完整申報期間起始日期，用於確認住院申報資料之申報起始時間，"
            "輔助判定同一次住院事件於不同申報期間或跨月申報時之資料連續性，並與入院日期、"
            "出院日期及住院天數進行一致性檢核，以避免將同一次住院誤判為多次住院或錯估住院期間。"
            "本欄位僅用於住院事件整理、資料品質檢核與時間序列分析，不會用於辨識個人。"
        ),
        reviewer_note="此欄位曾被審查提醒確認，建議使用此版較明確的說明。",
    ),
    SpecialFieldNeed(
        dataset_code="Health-02",
        dataset_name="全民健保處方及治療明細檔_住院",
        field_number=83,
        field_name="APPL_E_DATE",
        field_description="申報期間-迄",
        purpose=(
            "本研究需申請完整申報期間結束日期，用於確認住院申報資料所涵蓋之結束時間，"
            "輔助檢核住院事件、入院日期、出院日期及申報期間之一致性，避免跨月或跨期申報造成"
            "住院次數與住院天數估計錯誤；本欄位僅用於資料品質檢核與時間序列分析。"
        ),
    ),
    SpecialFieldNeed(
        dataset_code="Health-02",
        dataset_name="全民健保處方及治療明細檔_住院",
        field_number=84,
        field_name="IN_DATE_OLD",
        field_description="入院日期",
        purpose=(
            "本研究需申請完整入院日期，用於建立 ALS 與 Wilson’s disease 患者之住院照護時間軸，"
            "計算首次住院、診斷前後住院頻率、住院與門急診就醫之先後順序，以及與重大傷病申請/核定日期"
            "之時間間隔，以評估診斷延遲、照護流程與醫療資源使用情形；完整日期僅用於安全環境內進行"
            "時間間隔與事件順序分析。"
        ),
        reviewer_note="審查意見明確要求 Health-02「入院日期」完整日期需在第八頁特殊需求勾選。",
    ),
]


def special_full_date_purpose(field_name: str) -> str:
    normalized = field_name.strip().upper()
    for need in SPECIAL_FULL_DATE_NEEDS:
        if need.field_name.upper() == normalized:
            return need.purpose
    return ""


def special_needs_markdown() -> str:
    lines = [
        "### 第八頁特殊需求申請",
        "",
        "請勾選：`是`、`需使用完整日期欄位（含年月日）`，並在下列欄位填寫需求說明。",
        "",
    ]
    for need in SPECIAL_FULL_DATE_NEEDS:
        lines.append(f"**{need.label}**")
        lines.append("")
        lines.append("```text")
        lines.append(need.purpose)
        lines.append("```")
        if need.reviewer_note:
            lines.append(f"注意：{need.reviewer_note}")
        lines.append("")
    return "\n".join(lines)


def reviewer_feedback_for_special_needs() -> str:
    return (
        "已了解，感謝審查委員及承辦窗口提醒。本次已依審查意見於第八頁「特殊需求申請(選填)」"
        "勾選「是」，並勾選「需使用完整日期欄位（含年月日）」。另已於特殊需求欄位中勾選 "
        "Health-01「就醫日期」及 Health-02「入院日期」之完整日期欄位，並補充填寫各欄位之"
        "資料欄位需求緣由或目的，以利後續審核。再次感謝審查協助。"
    )


def reviewer_feedback_for_appl_s_date_revision() -> str:
    return (
        "已了解，感謝審查委員及承辦窗口提醒。本次已確認並修正 Health-02 欄位"
        "「APPL_S_DATE 申報期間-起」之特殊需求說明，使其更明確說明本欄位用於判定住院申報期間、"
        "跨月申報資料連續性、住院事件整理，以及與入院日期、出院日期及住院天數之一致性檢核；"
        "本欄位僅作為資料整理、品質檢核與時間序列分析使用，不會用於辨識個人。再次感謝審查協助。"
    )


def application_review_lessons() -> list[str]:
    return [
        "Health-01/Health-02 若需要完整日期，不能只在主檔欄位填寫理由，還必須到第八頁特殊需求申請勾選完整日期欄位。",
        "Health-02 APPL_S_DATE 的理由需明確說明跨月或不同申報期間下的住院事件連續性與一致性檢核。",
        "審查回覆應只說明已完成的修改，不要把「欄位敘述請確認/修正」等內部提醒文字留在欄位需求說明中。",
        "Health-10 欄位若與申請年度無交集，應取消勾選，而不是用文字補充說明硬保留。",
        "Health-51 罕見疾病通報資料庫是 rare disease 類研究的核心可行性資料，Phase 1 應納入以確認通報與健保資料串連可行性。",
    ]
