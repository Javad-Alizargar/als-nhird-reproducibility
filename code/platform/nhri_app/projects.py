from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import re


DEFAULT_PROJECTS_DIR = Path("projects")


@dataclass
class Project:
    slug: str
    title: str
    description: str
    export_dir: str
    created_at: str
    updated_at: str
    ai_plan_markdown: str = ""


def slugify(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", title).strip("_").lower()
    return slug[:80] or "untitled_project"


def project_dir(projects_dir: Path, slug: str) -> Path:
    return projects_dir / slug


def project_file(projects_dir: Path, slug: str) -> Path:
    return project_dir(projects_dir, slug) / "project.json"


def list_projects(projects_dir: Path = DEFAULT_PROJECTS_DIR) -> list[Project]:
    projects: list[Project] = []
    if not projects_dir.exists():
        return projects
    for path in sorted(projects_dir.glob("*/project.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("ai_plan_markdown", "")
            projects.append(Project(**data))
        except Exception:
            continue
    return sorted(projects, key=lambda project: project.updated_at, reverse=True)


def save_project(project: Project, projects_dir: Path = DEFAULT_PROJECTS_DIR) -> Project:
    now = datetime.now().isoformat(timespec="seconds")
    if not project.created_at:
        project.created_at = now
    project.updated_at = now
    folder = project_dir(projects_dir, project.slug)
    folder.mkdir(parents=True, exist_ok=True)
    project_file(projects_dir, project.slug).write_text(
        json.dumps(asdict(project), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(project.export_dir).mkdir(parents=True, exist_ok=True)
    return project


def ensure_rare_project(title: str, description: str, projects_dir: Path = DEFAULT_PROJECTS_DIR) -> Project:
    slug = "rare_disease_als_wilson"
    existing = project_file(projects_dir, slug)
    if existing.exists():
        data = json.loads(existing.read_text(encoding="utf-8"))
        data.setdefault("ai_plan_markdown", "")
        return Project(**data)
    project = Project(
        slug=slug,
        title=title,
        description=description,
        export_dir=str((project_dir(projects_dir, slug) / "exports").resolve()),
        created_at="",
        updated_at="",
    )
    return save_project(project, projects_dir)
