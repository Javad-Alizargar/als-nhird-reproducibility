from __future__ import annotations

from pathlib import Path
import os
import re


DEFAULT_SOURCE_DIR = Path(os.getenv("NHRI_SOURCE_DIR", r"D:\NHRI_au"))


SOURCE_FILES = {
    "catalog": "nhri_dataset_categories_expanded.txt",
    "datasets_pdf": "datasets_not expanded.pdf",
    "regulation_pdf": "衛生福利資料科學中心資料申請應用規定及申請文件_上網公告版_20260528.pdf",
    "money": "Money regulation.txt",
    "review": "review rules.txt",
    "qa": "QA.txt",
    "rare_plan": "rare disease plan.txt",
    "api_keys": "API_key.txt",
}


def read_key_file(source_dir: Path) -> dict[str, str]:
    path = source_dir / SOURCE_FILES["api_keys"]
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    keys: dict[str, str] = {}
    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        if "=" in clean:
            name, value = clean.split("=", 1)
            keys[name.strip().upper()] = value.strip().strip('"')
        elif ":" in clean:
            name, value = clean.split(":", 1)
            keys[name.strip().upper()] = value.strip().strip('"')
    for match in re.finditer(r"(sk-[A-Za-z0-9_\-]{20,})", text):
        keys.setdefault("DEEPSEEK_API_KEY", match.group(1))
    return keys


def get_deepseek_key(source_dir: Path) -> str | None:
    env_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
    if env_key:
        return env_key
    keys = read_key_file(source_dir)
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY", "API_KEY", "OPENAI_API_KEY"):
        if keys.get(name):
            return keys[name]
    return None
