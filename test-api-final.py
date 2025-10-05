# -*- coding: utf-8 -*-
import re, json, time, requests, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ----------------------- تنظیمات/مسیرها -----------------------
OUTPUT_DIR = Path("F:\\A-project\\Final\\run\\meerkat - full dataset\\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = ""
BOT_ID  = ""

FILES = {
    "single_step_cot": "F:\\A-project\\Final\\run\\meerkat - full dataset\\verify_single_step_cot.txt",
    "least_to_most": "F:\\A-project\\Final\\run\\meerkat - full dataset\\verify_least_to_most.txt",
    "zero_shot_direct": "F:\\A-project\\Final\\run\\meerkat - full dataset\\verify_zero_shot_direct.txt"
}

# طبق خواسته‌ی شما دقیقاً با همین نام و مقدار:
BASE_URL = "https://api.metisai.ir"

# جداکننده‌ها
DASH_SEP = "=" * 22     # --------------------------------------------------------------------------------
EQUAL_SEP = "-" * 80   # ======================

# تایم‌اوت/ریترا‌ی مثل کد مرجع
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120
MAX_RETRIES = 5
SLEEP_BETWEEN_MSGS = 0.2

# ----------------------- کمکی‌ها -----------------------
def detect_separator(text: str) -> str:
    """
    اگر خطی با 80 '-' پیدا شد، همان را مبنا می‌گیریم؛
    در غیر این صورت به الگوی مرجع (22 '=') برمی‌گردیم.
    """
    for line in text.splitlines():
        s = line.strip()
        if s == DASH_SEP:
            return DASH_SEP
    return EQUAL_SEP

def split_blocks(text: str) -> List[str]:
    """
    اسپلیت دقیق روی خط جداکننده که فقط خودش در یک خط آمده باشد.
    ابتدا 80 '-' را امتحان می‌کنیم؛ اگر نبود، 22 '=' مطابق کد مرجع.
    """
    sep = detect_separator(text)
    patt = rf"(?m)^{re.escape(sep)}\s*$"
    return [b for b in re.split(patt, text) if b.strip()]

def extract_id_and_body(block: str):
    # سطرهای خالی ابتدای بلوک را رد می‌کنیم
    lines = block.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    dataset_id = None
    # بدنهٔ پیش‌فرض: از اولین خط غیرخالی به بعد
    body = "\n".join(lines[i:])

    # اگر اولین خط غیرخالی با ID: شروع شد، جداش کن
    if i < len(lines) and lines[i].strip().lower().startswith("id:"):
        dataset_id = lines[i].split(":", 1)[1].strip()
        body = "\n".join(lines[i+1:]).lstrip("\n")

    return dataset_id, body

def load_done_indices(jsonl_path: Path) -> set:
    done = set()
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    done.add(int(obj.get("idx")))
                except Exception:
                    pass
    return done

def append_jsonl(path: Path, obj: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()

# ----------------------- اندپوینت‌ها (مطابق مرجع) -----------------------
def make_session() -> str:
    """
    ساخت سشن طبق کد مرجع:
    POST {BASE_URL}/api/v1/chat/session
    body: {"botId": ..., "user": None, "initialMessages": None}
    """
    r = requests.post(
        f"{BASE_URL}/api/v1/chat/session",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"botId": BOT_ID, "user": None, "initialMessages": None},
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    )
    r.raise_for_status()
    return r.json()["id"]

def send_message(session_id: str, content: str) -> Dict[str, Any]:
    """
    ارسال پیام طبق کد مرجع با ریترا‌ی/بک‌آف:
    POST {BASE_URL}/api/v1/chat/session/{session_id}/message
    body: {"message": {"content": ..., "type": "USER"}}
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(
                f"{BASE_URL}/api/v1/chat/session/{session_id}/message",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"message": {"content": content, "type": "USER"}},
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise e

# ----------------------- پردازش فایل‌ها -----------------------
def process_method(method: str, prompt_file: Path):
    out_path = OUTPUT_DIR / f"{method}.jsonl"
    fail_path = OUTPUT_DIR / f"{method}.failures.jsonl"

    raw_text = prompt_file.read_text(encoding="utf-8")
    prompts = split_blocks(raw_text)
    print(f"▶ {method}: {len(prompts)} prompts")

    done_idxs = load_done_indices(out_path)
    if done_idxs:
        print(f"↩️  Resuming: {len(done_idxs)} already done, will skip them")

    try:
        for idx, block in enumerate(prompts):
            if idx in done_idxs:
                continue

            dataset_id, body = extract_id_and_body(block)
            if not body.strip():
                append_jsonl(fail_path, {
                    "idx": idx,
                    "dataset_id": dataset_id,
                    "error": "empty_body_after_strip_id"
                })
                continue

            # هر پرامپت = یک سشن مستقل (مثل کد مرجع)
            try:
                session_id = make_session()
            except requests.HTTPError as e:
                append_jsonl(fail_path, {
                    "idx": idx,
                    "dataset_id": dataset_id,
                    "error": "make_session_failed",
                    "status": getattr(e.response, "status_code", None),
                    "body": getattr(e.response, "text", None)
                })
                continue
            except Exception as e:
                append_jsonl(fail_path, {
                    "idx": idx,
                    "dataset_id": dataset_id,
                    "error": f"make_session_exc: {repr(e)}"
                })
                continue

            try:
                ans = send_message(session_id, body)  # ← فقط بدنه ارسال می‌شود
                # خروجی کم‌حجم: بدون prompt
                append_jsonl(out_path, {
                    "idx": idx,
                    "dataset_id": dataset_id,
                    "answer": ans
                })
            except requests.HTTPError as e:
                append_jsonl(fail_path, {
                    "idx": idx,
                    "dataset_id": dataset_id,
                    "error": "send_message_failed",
                    "status": getattr(e.response, "status_code", None),
                    "body": getattr(e.response, "text", None)
                })
            except Exception as e:
                append_jsonl(fail_path, {
                    "idx": idx,
                    "dataset_id": dataset_id,
                    "error": f"send_message_exc: {repr(e)}"
                })

            time.sleep(SLEEP_BETWEEN_MSGS)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Everything up to now is saved.")

# ----------------------- اجرا -----------------------
if __name__ == "__main__":
    if not API_KEY or not BOT_ID:
        raise SystemExit("❌ Fill in API_KEY and BOT_ID at the top of the script.")

    for method, filepath in FILES.items():
        process_method(method, Path(filepath))

    print("✅ Done. Results are in:")
    for method in FILES.keys():
        print(f"  - {method}.jsonl (successes)")
        print(f"  - {method}.failures.jsonl (failures, if any)")
