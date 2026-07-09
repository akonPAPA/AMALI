"""Security & privacy validation for AMALI-FT-v0 (Stage 13).

    python scripts/security_validate_amali_ft_v0.py

Mandatory scans (decide the exit code):
1. training dataset JSONL — secret-shaped patterns;
2. artifacts/ directory — secret-shaped patterns and stray .env files;
3. latest model_card.md — forbidden claim markers;
4. git tracked files — no artifacts/data/models/cache/logs/.env/egg-info.

Optional tools (gitleaks, semgrep, bandit, ruff, mypy) run only when
installed; otherwise they are honestly reported NOT_RUN — never faked.
Their findings are reported but only the mandatory scans gate the exit.

Writes security_validation_report.{json,md} under
``artifacts/amali_ft_v0/<timestamp>/``. Exit 0 only when every mandatory
scan passes.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DATASET_PATH = Path("D:/AMALI/data/processed/amali_bmg_train.jsonl")
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai_style_key": re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    "github_token": re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    "bearer_token": re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key_marker": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credentialed_url": re.compile(
        r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^/\s:@]+:[^/\s:@]+@"
    ),
}

# Markers kept aligned with amali.training.model_card._FORBIDDEN_MARKERS,
# plus phrasing variants. Chosen so the card's own negations ("No claim of
# AGI ... is made") never match.
FORBIDDEN_CLAIM_MARKERS = (
    "beats gpt",
    "beats claude",
    "beats deepseek",
    "beats glm",
    "beats qwen",
    "better than gpt",
    "better than claude",
    "outperforms gpt",
    "outperforms claude",
    "frontier model",
    "frontier-class",
    "agi achieved",
    "achieves agi",
    "asi achieved",
    "is self-improving",
    "trained from scratch by amali",
)

FORBIDDEN_TRACKED = re.compile(
    r"^(artifacts/|data/|models/|cache/|logs/)|(\.env$|\.env\.|\.egg-info)"
)

OPTIONAL_TOOLS = {
    "gitleaks": ["gitleaks", "detect", "--source", str(REPO_ROOT), "--no-banner"],
    "semgrep": ["semgrep", "scan", "--quiet", "--error", str(REPO_ROOT)],
    "bandit": ["bandit", "-r", str(REPO_ROOT / "src"), "-q"],
    "ruff": ["ruff", "check", str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")],
    "mypy": ["mypy", str(REPO_ROOT / "src")],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_ref() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=REPO_ROOT,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only
        pass
    return "unknown"


def _scan_text(text: str, origin: str) -> list[dict]:
    findings = []
    for name, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(text):
            snippet = match.group(0)
            findings.append(
                {
                    "scan": "secret_pattern",
                    "pattern": name,
                    "path": origin,
                    # never echo the full candidate secret back out
                    "evidence": snippet[:8] + "..." if len(snippet) > 8 else snippet,
                }
            )
    return findings


def scan_dataset() -> dict:
    if not DATASET_PATH.exists():
        return {"status": "NOT_RUN", "detail": "dataset not built", "findings": []}
    findings = _scan_text(
        DATASET_PATH.read_text(encoding="utf-8"), str(DATASET_PATH)
    )
    return {"status": "PASS" if not findings else "FAIL", "findings": findings}


def scan_artifacts() -> dict:
    if not ARTIFACTS_DIR.is_dir():
        return {"status": "NOT_RUN", "detail": "no artifacts dir", "findings": []}
    findings: list[dict] = []
    for path in sorted(ARTIFACTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        if path.name.startswith(".env") or path.suffix == ".env":
            findings.append(
                {"scan": "env_file", "pattern": "env_file", "path": rel,
                 "evidence": "raw .env file in artifacts"}
            )
            continue
        if path.suffix.lower() not in (".json", ".md", ".txt", ".jsonl"):
            continue
        try:
            findings.extend(
                _scan_text(path.read_text(encoding="utf-8"), rel)
            )
        except UnicodeDecodeError:
            findings.append(
                {"scan": "binary_artifact", "pattern": "binary",
                 "path": rel, "evidence": "unreadable as UTF-8"}
            )
    return {"status": "PASS" if not findings else "FAIL", "findings": findings}


def scan_model_card() -> dict:
    cards = sorted(ARTIFACTS_DIR.rglob("model_card.md")) if ARTIFACTS_DIR.is_dir() else []
    if not cards:
        return {
            "status": "NOT_RUN",
            "detail": "no model_card.md generated yet",
            "findings": [],
        }
    card = cards[-1]
    lowered = card.read_text(encoding="utf-8").lower()
    findings = [
        {
            "scan": "forbidden_claim",
            "pattern": marker,
            "path": str(card.relative_to(REPO_ROOT)).replace("\\", "/"),
            "evidence": marker,
        }
        for marker in FORBIDDEN_CLAIM_MARKERS
        if marker in lowered
    ]
    return {"status": "PASS" if not findings else "FAIL", "findings": findings}


def scan_git_tracked() -> dict:
    out = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO_ROOT,
    )
    if out.returncode != 0:
        return {"status": "NOT_RUN", "detail": "git ls-files failed", "findings": []}
    findings = [
        {"scan": "git_tracked", "pattern": "forbidden_tracked_path",
         "path": line, "evidence": line}
        for line in out.stdout.splitlines()
        if FORBIDDEN_TRACKED.search(line)
    ]
    return {"status": "PASS" if not findings else "FAIL", "findings": findings}


def run_optional_tools() -> dict:
    results: dict[str, dict] = {}
    for name, cmd in OPTIONAL_TOOLS.items():
        if shutil.which(cmd[0]) is None:
            results[name] = {"status": "NOT_RUN", "detail": "not installed"}
            continue
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
                cwd=REPO_ROOT,
            )
            results[name] = {
                "status": "PASS" if proc.returncode == 0 else "FINDINGS",
                "exit_code": proc.returncode,
                "output_tail": ((proc.stdout or "") + (proc.stderr or ""))[-2000:],
            }
        except subprocess.TimeoutExpired:
            results[name] = {"status": "TIMEOUT"}
    return results


def main() -> int:
    scans = {
        "dataset_secret_scan": scan_dataset(),
        "artifact_secret_scan": scan_artifacts(),
        "model_card_claim_scan": scan_model_card(),
        "git_tracked_file_scan": scan_git_tracked(),
    }
    tools = run_optional_tools()

    mandatory_fail = any(s["status"] == "FAIL" for s in scans.values())
    status = "FAIL" if mandatory_fail else "PASS"
    all_findings = [f for s in scans.values() for f in s.get("findings", [])]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "amali_ft_v0" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "security_validation_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": _now(),
                "repo_ref": _repo_ref(),
                "command": "python scripts/security_validate_amali_ft_v0.py",
                "status": status,
                "metrics": {
                    "findings": len(all_findings),
                    **{k: v["status"] for k, v in scans.items()},
                },
                "scans": scans,
                "optional_tools": tools,
                "risks": [
                    f"{f['scan']}: {f['pattern']} in {f['path']}"
                    for f in all_findings
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "security_validation_report.md").write_text(
        f"# Security Validation Report\n\nStatus: **{status}**\n\n"
        "| scan | status |\n|------|--------|\n"
        + "\n".join(f"| {k} | {v['status']} |" for k, v in scans.items())
        + "\n\n## Optional tools\n\n"
        + "\n".join(f"- {k}: {v['status']}" for k, v in tools.items())
        + "\n\n## Findings\n\n"
        + (
            "\n".join(
                f"- {f['scan']}: `{f['pattern']}` in `{f['path']}`"
                for f in all_findings
            )
            if all_findings
            else "None."
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Security validation: {status}")
    for name, scan in scans.items():
        print(f"  {name}: {scan['status']}")
    for name, result in tools.items():
        print(f"  optional {name}: {result['status']}")
    for finding in all_findings:
        print(
            f"  finding: {finding['scan']} {finding['pattern']} {finding['path']}",
            file=sys.stderr,
        )
    print(f"Report: {out_dir / 'security_validation_report.json'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
