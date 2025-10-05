# analyze.py  —  fixed + department breakdown
import json, re, csv
from collections import Counter, defaultdict
from pathlib import Path

# =====================[ CONFIG ]=====================
# مسیر فایل نتایج و فولدر خروجی را اینجا تنظیم کن
DATA_PATH  = Path("F:\\A-project\\Final\\run\\meerkat - full dataset\\results\\single_step_cot.jsonl")  # اگر لازم بود مطلق کن
OUTPUT_DIR = Path("F:\\A-project\\Final\\run\\meerkat - full dataset\\results")               # پوشه‌ی خروجی
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ====================================================

# همان منطق پاک‌سازی و پارس JSON در answer.content
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)   # حذف ```/```json
BRACE_BLOCK = re.compile(r"\{.*\}", re.DOTALL)                   # اولین {...}

def parse_content(raw: str):
    """raw = answer.content (ممکن است بلاک مارک‌داون داشته باشد). خروجی: dict JSON داخلی یا None."""
    if raw is None:
        return None
    s = str(raw).strip()
    s = FENCE.sub("", s).strip()
    if not (s.startswith("{") and s.endswith("}")):
        m = BRACE_BLOCK.search(s)
        if m:
            s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # خط خراب را رد کن
                continue

# ---------- تحلیل کلی (مثل نسخه‌ی قبلی شما) ----------
def analyze_overall(path: Path):
    c_top1 = Counter()     # YES/NO/UNSCORABLE
    c_top3 = Counter()     # YES/NO
    c_top5 = Counter()     # YES/NO
    total = 0
    invalid_top1 = 0
    invalid_top3 = 0
    invalid_top5 = 0

    for rec in iter_jsonl(path):
        total += 1
        content = (rec.get("answer") or {}).get("content")
        inner = parse_content(content)
        if not isinstance(inner, dict):
            invalid_top1 += 1
            invalid_top3 += 1
            invalid_top5 += 1
            continue

        top1 = (inner.get("TOP1") or "").strip().upper()
        top3 = (inner.get("TOP3") or "").strip().upper()
        top5 = (inner.get("TOP5") or "").strip().upper()

        if top1 in ("YES","NO","UNSCORABLE"):
            c_top1[top1] += 1
        else:
            invalid_top1 += 1

        if top3 in ("YES","NO"):
            c_top3[top3] += 1
        else:
            invalid_top3 += 1

        if top5 in ("YES","NO"):
            c_top5[top5] += 1
        else:
            invalid_top5 += 1

    return {
        "total_rows": total,
        "TOP1_YES": c_top1.get("YES", 0),
        "TOP1_NO": c_top1.get("NO", 0),
        "TOP1_UNSCORABLE": c_top1.get("UNSCORABLE", 0),
        "TOP3_YES": c_top3.get("YES", 0),
        "TOP3_NO": c_top3.get("NO", 0),
        "TOP5_YES": c_top5.get("YES", 0),
        "TOP5_NO": c_top5.get("NO", 0),
        "invalid_TOP1": invalid_top1,
        "invalid_TOP3": invalid_top3,
        "invalid_TOP5": invalid_top5,
    }

# ---------- نگاشت دپارتمان‌ها بر اساس جدول شما ----------
DEPARTMENTS = [
    ("Surgery", 1, 93),
    ("Obstetrics and Gynecology", 94, 189),
    ("Internal Medicine", 190, 288),
    ("Dentistry", 289, 360),
    ("Neurology", 361, 436),
    ("Oncology", 437, 492),
    ("Orthopedics", 493, 581),
    ("Pediatrics", 582, 640),
    ("Otorhinolaryngology", 641, 724),
    ("Reproductive and Men's Health", 725, 797),
    ("Dermatovenereology", 798, 918),
    ("Other", 919, 989),
    ("Psychology", 990, 1073),
    ("Hematology", 1074, 1121),
    ("Infectious Diseases and Immunology", 1122, 1148),
]

def dept_for_number(n: int):
    for dept, start, end in DEPARTMENTS:
        if start <= n <= end:
            return dept
    return None

def extract_dataset_number(rec: dict):
    """
    فایل فعلی شما فیلد 'dataset_id' دارد مثل 'dxbench_241'.
    این تابع عدد انتهایی را برمی‌گرداند (241).
    """
    # ترتیب جست‌وجو شامل dataset_id هم باشد
    for key in ("dataset_id","id","sample_id","case_id","qid","question_id","dx_id"):
        if key in rec and rec[key] is not None:
            s = str(rec[key])
            m = re.findall(r"(\d+)", s)
            if m:
                return int(m[-1])
    meta = rec.get("meta") or rec.get("metadata") or {}
    for key in ("dataset_id","id","sample_id","case_id","qid","question_id","dx_id"):
        if key in meta and meta[key] is not None:
            s = str(meta[key])
            m = re.findall(r"(\d+)", s)
            if m:
                return int(m[-1])
    return None

# ---------- تحلیل به تفکیک دپارتمان ----------
def analyze_by_department(path: Path):
    # دیکشنری اولیه با همه دپارتمان‌ها
    results = {
        dept: {
            "department": dept,
            "total_rows": 0,
            "TOP1_YES": 0, "TOP1_NO": 0, "TOP1_UNSCORABLE": 0, "invalid_TOP1": 0,
            "TOP3_YES": 0, "TOP3_NO": 0, "invalid_TOP3": 0,
            "TOP5_YES": 0, "TOP5_NO": 0, "invalid_TOP5": 0,
        }
        for dept,_,_ in DEPARTMENTS
    }
    # برای شفافیت: رکوردهایی که نگاشت‌پذیر نیستند
    results["_UNMATCHED"] = {
        "department": "_UNMATCHED",
        "total_rows": 0,
        "TOP1_YES": 0, "TOP1_NO": 0, "TOP1_UNSCORABLE": 0, "invalid_TOP1": 0,
        "TOP3_YES": 0, "TOP3_NO": 0, "invalid_TOP3": 0,
        "TOP5_YES": 0, "TOP5_NO": 0, "invalid_TOP5": 0,
    }

    for rec in iter_jsonl(path):
        num = extract_dataset_number(rec)
        dept = dept_for_number(num) if num is not None else None
        bucket = results[dept] if dept in results else results["_UNMATCHED"]
        bucket["total_rows"] += 1

        content = (rec.get("answer") or {}).get("content")
        inner = parse_content(content)
        if not isinstance(inner, dict):
            bucket["invalid_TOP1"] += 1
            bucket["invalid_TOP3"] += 1
            bucket["invalid_TOP5"] += 1
            continue

        t1 = (inner.get("TOP1") or "").strip().upper()
        t3 = (inner.get("TOP3") or "").strip().upper()
        t5 = (inner.get("TOP5") or "").strip().upper()

        if t1 in ("YES","NO","UNSCORABLE"):
            bucket["TOP1_" + t1] += 1
        else:
            bucket["invalid_TOP1"] += 1

        if t3 in ("YES","NO"):
            bucket["TOP3_" + t3] += 1
        else:
            bucket["invalid_TOP3"] += 1

        if t5 in ("YES","NO"):
            bucket["TOP5_" + t5] += 1
        else:
            bucket["invalid_TOP5"] += 1

    # ترتیب خروجی مطابق سورت جدول شما
    ordered = [results[dept] for dept,_,_ in DEPARTMENTS]
    if results["_UNMATCHED"]["total_rows"] > 0:
        ordered.append(results["_UNMATCHED"])
    return ordered

# ---------- main ----------
def main():
    # 1) خلاصه‌ی کلی
    overall = analyze_overall(DATA_PATH)

    # 2) خلاصه‌ی دپارتمانی
    by_dept = analyze_by_department(DATA_PATH)

    # چاپ خلاصه در ترمینال
    print("\n=== SUMMARY (OVERALL) ===")
    print(f"  total rows: {overall['total_rows']}")
    print(f"  TOP1 → YES: {overall['TOP1_YES']}  NO: {overall['TOP1_NO']}  UNSCORABLE: {overall['TOP1_UNSCORABLE']}  | invalid/missing: {overall['invalid_TOP1']}")
    print(f"  TOP3 → YES: {overall['TOP3_YES']}  NO: {overall['TOP3_NO']}  | invalid/missing: {overall['invalid_TOP3']}")
    print(f"  TOP5 → YES: {overall['TOP5_YES']}  NO: {overall['TOP5_NO']}  | invalid/missing: {overall['invalid_TOP5']}")

    # ذخیره CSV کلی
    out_csv_overall = OUTPUT_DIR / "judgment_counts_overall.csv"
    with out_csv_overall.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "total_rows",
            "TOP1_YES","TOP1_NO","TOP1_UNSCORABLE","invalid_TOP1",
            "TOP3_YES","TOP3_NO","invalid_TOP3",
            "TOP5_YES","TOP5_NO","invalid_TOP5",
        ])
        w.writerow([
            overall["total_rows"],
            overall["TOP1_YES"], overall["TOP1_NO"], overall["TOP1_UNSCORABLE"], overall["invalid_TOP1"],
            overall["TOP3_YES"], overall["TOP3_NO"], overall["invalid_TOP3"],
            overall["TOP5_YES"], overall["TOP5_NO"], overall["invalid_TOP5"],
        ])
    print(f"CSV (overall) saved to: {out_csv_overall.resolve()}")

    # ذخیره CSV به تفکیک دپارتمان
    out_csv_by_dept = OUTPUT_DIR / "judgment_counts_by_department.csv"
    with out_csv_by_dept.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "department","total_rows",
            "TOP1_YES","TOP1_NO","TOP1_UNSCORABLE","invalid_TOP1",
            "TOP3_YES","TOP3_NO","invalid_TOP3",
            "TOP5_YES","TOP5_NO","invalid_TOP5",
        ])
        for row in by_dept:
            w.writerow([
                row["department"], row["total_rows"],
                row["TOP1_YES"], row["TOP1_NO"], row["TOP1_UNSCORABLE"], row["invalid_TOP1"],
                row["TOP3_YES"], row["TOP3_NO"], row["invalid_TOP3"],
                row["TOP5_YES"], row["TOP5_NO"], row["invalid_TOP5"],
            ])
    print(f"CSV (by department) saved to: {out_csv_by_dept.resolve()}")

    # (اختیاری) CSV با نرخ‌ها هم بسازیم
    try:
        import pandas as pd
        def safe_pct(n, d): 
            return round(100.0*n/d, 2) if d else 0.0
        rows = by_dept
        df = pd.DataFrame(rows)
        df["TOP1_ACC_%"] = df.apply(lambda r: safe_pct(r["TOP1_YES"], r["TOP1_YES"]+r["TOP1_NO"]), axis=1)
        df["TOP3_HIT_%"] = df.apply(lambda r: safe_pct(r["TOP3_YES"], r["TOP3_YES"]+r["TOP3_NO"]), axis=1)
        df["TOP5_HIT_%"] = df.apply(lambda r: safe_pct(r["TOP5_YES"], r["TOP5_YES"]+r["TOP5_NO"]), axis=1)
        out_csv_rates = OUTPUT_DIR / "judgment_counts_by_department_with_rates.csv"
        df.to_csv(out_csv_rates, index=False)
        print(f"CSV (by department with rates) saved to: {out_csv_rates.resolve()}")
    except Exception as e:
        print("Skipping rates CSV (pandas not available?):", e)

if __name__ == "__main__":
    main()
