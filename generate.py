# -*- coding: utf-8 -*-
"""
Generate assistant answers (via Metis) and build Verification Prompts (Top1/Top3/Top5),
while REDACTING DxBench ID and GT from the content sent to the API.

— How it works —
1) Reads an input prompts file containing multiple cases separated by a line of 22 '=' characters.
   - Typical block header:
       Line 1: DxBench_<id>   (e.g., DxBench_481)
       Line 2: <GROUND TRUTH DIAGNOSIS>
       Then:   Patient Symptoms / Clinical Notes / ...
2) For API: removes the ID line and the first non-empty header line after it (GT).
3) Calls Metis (unless DRY_RUN=True) and stores raw assistant outputs in <OUTPUT_ROOT>/results/<METHOD>.jsonl
4) Builds a Verification Prompt per case at:
       <OUTPUT_ROOT>/results/verification/<METHOD>/<dxbench_id>.txt
   and also appends all cases to:
       <OUTPUT_ROOT>/results/verification/<METHOD>.all.txt

Test offline (no API calls):
    DRY_RUN = True
Run online:
    DRY_RUN = False and set METIS_API_KEY / METIS_BOT_ID constants below.

Author: you :)
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests


# =============================================================================
#                       ⚙️  EDITABLE CONSTANTS (TOP-OF-FILE)
# =============================================================================

# --- I/O paths ---
INPUT_FILE   = "F:\\A-project\\Final\\run\\GPT-4o - full dataset\\prompts_zero_shot_direct.txt"   # مسیر فایل ورودی پرامپت‌ها
OUTPUT_ROOT  = "F:\\A-project\\Final\\run\\GPT-4o - full dataset\\run_out"                      # ریشهٔ خروجی‌ها (پوشهٔ نتایج)
METHOD       = "zero_shot_direct"                # نام روش: zero_shot_direct | single_step_cot | least_to_most

# --- Metis API config ---
API_BASE        = "https://api.metisai.ir"
METIS_API_KEY   = ""   # اگر خالی بماند و DRY_RUN=False، اسکریپت خطا می‌دهد
METIS_BOT_ID    = ""   # اگر خالی بماند و DRY_RUN=False، اسکریپت خطا می‌دهد

# --- Runtime toggles ---
DRY_RUN            = False   # True → بدون تماس با API (فقط ساخت فایل‌ها)
CONNECT_TIMEOUT    = 10
READ_TIMEOUT       = 120
MAX_RETRIES        = 5
SLEEP_BETWEEN_MSGS = 0.2

# --- Input block separator (exactly 22 '=' signs on a line) ---
SEP = "=" * 22


# =============================================================================
#                                 Helpers
# =============================================================================

def split_blocks(text: str) -> List[str]:
    """Split by a line that is exactly 22 '=' characters."""
    return [b for b in re.split(rf"(?m)^{re.escape(SEP)}\s*$", text) if b.strip()]

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def append_jsonl(path: Path, obj: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()

def load_done_ids(jsonl_path: Path) -> set:
    done = set()
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    if "id" in o:
                        done.add(str(o["id"]))
                except Exception:
                    pass
    return done


# =============================================================================
#                        Parse ID / GT, and REDACT for API
# =============================================================================

ID_LINE = re.compile(r"^\s*(dxbench|DxBench)[_\- ]?(\d+)\s*$", re.IGNORECASE)

STOP_HEADERS = [
    r"^\s*Patient\s+Symptoms\s*:?",
    r"^\s*Explicit\s+Symptoms\s*:?",
    r"^\s*Implicit\s+Symptoms\s*:?",
    r"^\s*Symptoms\s*:?",
    r"^\s*Clinical\s+Notes\s*:?",
    r"^\s*Case\s*:?",
]
STOP_RE = re.compile("|".join(STOP_HEADERS), re.IGNORECASE)

def extract_id(block: str) -> Optional[str]:
    for ln in block.splitlines():
        m = ID_LINE.match(ln.strip())
        if m:
            return f"dxbench_{m.group(2)}"
    return None

def extract_gt(block: str) -> str:
    """
    In the top header (before Symptoms/Case headers), return the first non-empty
    line that is NOT an ID → considered GT.
    """
    lines = [ln.rstrip() for ln in block.splitlines()]

    stop_idx = None
    for i, ln in enumerate(lines):
        if STOP_RE.search(ln):
            stop_idx = i
            break

    header = lines[:stop_idx] if stop_idx is not None else lines[:5]  # small buffer if no headers
    for ln in header:
        s = ln.strip()
        if not s:
            continue
        if ID_LINE.match(s):
            continue
        return s
    return ""

def redact_id_and_gt_for_api(block: str) -> str:
    """
    Remove: (1) any ID lines (DxBench_***), and (2) the first non-empty header line
    after that (assumed GT), while preserving the rest of the block (Symptoms, etc.).
    """
    lines = [ln.rstrip("\n") for ln in block.splitlines()]

    # find header break
    stop_idx = None
    for i, ln in enumerate(lines):
        if STOP_RE.search(ln):
            stop_idx = i
            break

    if stop_idx is None:
        # No clear header; still try: drop ID line, then first non-empty next line as GT
        new_lines = []
        gt_removed = False
        saw_id = False
        for ln in lines:
            s = ln.strip()
            if not gt_removed:
                if ID_LINE.match(s):
                    saw_id = True
                    continue
                if saw_id and s:
                    gt_removed = True
                    continue
            new_lines.append(ln)
        # trim leading empties
        while new_lines and not new_lines[0].strip():
            new_lines.pop(0)
        return "\n".join(new_lines).strip() + "\n"

    # With header split:
    header = lines[:stop_idx]
    body   = lines[stop_idx:]  # keep intact

    cleaned_header = []
    gt_removed = False
    for ln in header:
        s = ln.strip()
        if ID_LINE.match(s):
            continue  # drop ID
        if not gt_removed and s:
            gt_removed = True  # drop GT (first non-empty)
            continue
        cleaned_header.append(ln)

    final_lines = []
    if cleaned_header:
        final_lines.extend(cleaned_header)
    final_lines.extend(body)

    # trim leading empties
    while final_lines and not final_lines[0].strip():
        final_lines.pop(0)
    return "\n".join(final_lines).strip() + "\n"


# =============================================================================
#                   Build Verification Prompt (Top1/Top3/Top5)
# =============================================================================

def build_verification_prompt(idx: str, gt: str, assistant_output: str) -> str:
    """
    Format EXACTLY as requested:
        ID: dxbench_***
        >> VERIFICATION PROMPT
        GROUND-TRUTH DIAGNOSIS: ...
        Assistant_output:
        <<<
        {assistant_output}
        >>>
        ...instructions...
    """
    return (
        f"ID: {idx}\n"
        f">> VERIFICATION PROMPT\n"
        f"GROUND-TRUTH DIAGNOSIS: {gt}\n\n"
        "Assistant_output:\n"
        "<<<\n"
        f"{assistant_output}\n"
        ">>>\n\n"
        "You are a strict medical judge.\n\n"
        "INPUT:\n"
        "• GT: the ground-truth diagnosis (string)\n"
        "• Assistant_output: free text (may be narrative, lists, or mixed)\n\n"
        "TASK:\n"
        "Decide whether the assistant’s BEST diagnosis (Top-1) matches GT, and also whether GT appears within the assistant’s Top-3 and Top-5 diagnoses.\n\n"
        "HOW TO FIND THE SINGLE BEST DIAGNOSIS (“BEST”) — priority order:\n"
        "1) If the text explicitly names a final/primary/most-likely/definitive diagnosis, use that.\n"
        "2) Else if any diagnoses are ranked, numbered, or given probabilities/likelihoods,\n"
        "   pick the highest-ranked/probability item; ties → pick the one appearing earliest.\n"
        "3) Else if multiple diagnoses are discussed narratively, pick the one the text argues\n"
        "   FOR most strongly (most supportive reasons, least hedging), not merely mentioned.\n"
        "4) Else if only one diagnosis is meaningfully proposed, use it.\n"
        "5) If you cannot extract any diagnosis the assistant endorses → UNSCORABLE.\n\n"
        "MATCHING (normalize before comparing):\n"
        "• Lowercase; remove punctuation and parentheticals.\n"
        "• Accept common clinical synonyms (non-exhaustive): CVA↔stroke; MI/heart attack↔myocardial infarction;\n"
        "  CHD↔coronary artery disease; PID↔pelvic inflammatory disease; PCOS↔polycystic ovary syndrome;\n"
        "  UTI↔urinary tract infection; DVT↔deep vein thrombosis; eczema↔atopic dermatitis.\n"
        "• Do NOT credit broader categories unless clearly meant as the specific disease (no parent/child credit).\n\n"
        "SECONDARY CHECKS:\n"
        "Also note whether GT appears among the assistant’s Top-3 diagnoses, and whether it appears among the Top-5 diagnoses. Mentions only for exclusion do NOT count.\n\n"
        "OUTPUT — single JSON line, no prose:\n"
        '{"TOP1":"YES|NO|UNSCORABLE","TOP3":"YES|NO","TOP5":"YES|NO","BEST":"<assistant_best_dx>"}'
    )


# =============================================================================
#                                Metis API
# =============================================================================

def make_session(api_key: str, bot_id: str) -> str:
    r = requests.post(
        f"{API_BASE}/api/v1/chat/session",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"botId": bot_id, "user": None, "initialMessages": None},
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    )
    r.raise_for_status()
    return r.json()["id"]

def send_message(api_key: str, session_id: str, content: str) -> Dict[str, Any]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(
                f"{API_BASE}/api/v1/chat/session/{session_id}/message",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"message": {"content": content, "type": "USER"}},
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise

def extract_assistant_text(api_response: Dict[str, Any]) -> str:
    """
    Try common shapes:
      - {"messages":[{"role":"ASSISTANT","content": "..."} , ...], ...}
      - {"content":"..."} or {"answer":{"content":"..."}}
    Fallback to str(api_response)
    """
    msgs = api_response.get("messages") or api_response.get("data") or None
    if isinstance(msgs, list):
        for msg in reversed(msgs):
            role = str(msg.get("role", "")).upper()
            if role in ("ASSISTANT", "AI"):
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
                if isinstance(c, dict) and isinstance(c.get("content"), str):
                    return c["content"].strip()
    ans = api_response.get("answer")
    if isinstance(ans, dict) and isinstance(ans.get("content"), str):
        return ans["content"].strip()
    if isinstance(api_response.get("content"), str):
        return api_response["content"].strip()
    return json.dumps(api_response, ensure_ascii=False)


# =============================================================================
#                                 Main
# =============================================================================

def process_file(method: str, prompt_path: Path, out_root: Path):
    # Prepare outputs
    out_dir = out_root / "results"
    verif_dir = out_dir / "verification" / method
    ok_jsonl   = out_dir / f"{method}.jsonl"
    fail_jsonl = out_dir / f"{method}.failures.jsonl"
    bundle_path = out_dir / "verification" / f"{method}.all.txt"

    safe_mkdir(out_dir)
    safe_mkdir(verif_dir)
    safe_mkdir(bundle_path.parent)

    done_ids = load_done_ids(ok_jsonl)

    text = prompt_path.read_text(encoding="utf-8")
    blocks = split_blocks(text)
    print(f"▶ {method}: {len(blocks)} prompts")

    for idx, block in enumerate(blocks):
        cid = extract_id(block) or f"{method}_{idx:04d}"
        if cid in done_ids:
            continue

        gt = extract_gt(block).strip() or "<UNKNOWN_GT>"

        # --- Content for API (REDACTED: no ID, no GT) ---
        content_for_api = redact_id_and_gt_for_api(block)

        # --- Call API or Mock ---
        if DRY_RUN:
            assistant_text = (
                '{"BEST":"Example Dx",'
                '"RANKED":[["Example Dx",0.62],["Alt Dx 1",0.23],["Alt Dx 2",0.15],["Alt Dx 3",0.07],["Alt Dx 4",0.03]]}'
            )
        else:
            if not METIS_API_KEY or not METIS_BOT_ID:
                raise RuntimeError("Missing METIS_API_KEY or METIS_BOT_ID constants.")
            try:
                session_id = make_session(METIS_API_KEY, METIS_BOT_ID)
                api_resp = send_message(METIS_API_KEY, session_id, content_for_api)
                assistant_text = extract_assistant_text(api_resp)
            except Exception as e:
                append_jsonl(fail_jsonl, {"id": cid, "error": repr(e)})
                continue
            time.sleep(SLEEP_BETWEEN_MSGS)

        # --- Save raw assistant output for traceability ---
        append_jsonl(ok_jsonl, {"id": cid, "output": assistant_text})

        # --- Build Verification Prompt file (with ID & GT at top) ---
        verif_text = build_verification_prompt(idx=cid, gt=gt, assistant_output=assistant_text)
        (verif_dir / f"{cid}.txt").write_text(verif_text, encoding="utf-8")
        with bundle_path.open("a", encoding="utf-8") as bf:
            bf.write(verif_text + "\n" + ("-" * 80) + "\n\n")

def main():
    # Optional CLI overrides (kept minimal since you asked for constants at top)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=INPUT_FILE,  help="Path to prompts file")
    parser.add_argument("--out",    default=OUTPUT_ROOT, help="Output root dir")
    parser.add_argument("--method", default=METHOD,      help="Method name")
    parser.add_argument("--dry",    action="store_true", help="Force dry-run (overrides DRY_RUN=True)")
    args = parser.parse_args()

    global DRY_RUN
    if args.dry:
        DRY_RUN = True

    process_file(
        method=args.method,
        prompt_path=Path(args.input).expanduser().resolve(),
        out_root=Path(args.out).expanduser().resolve()
    )

if __name__ == "__main__":
    main()
