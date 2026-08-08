
"""
compare_errors.py를 확장한 버전 — baseline vs model per_image_errors.csv
(컬럼: image_id, gt_count, pred_cnt, abs_error, crop_triggered)를 받아서
val/test용 Figure에 넣을 후보 이미지(TOP-K 개선 / TOP-K 악화)를 뽑는다.
 
compare_errors.py와의 차이:
  - split을 인자로 직접 받아 출력 CSV에 태그를 남김 (val/test 두 번 돌려서
    합친 리포트를 만들 때 편하도록)
  - "개선"과 "악화"를 분리해서 각각 TOP-K 출력 (compare_errors.py는 악화만 정렬)
  - crop_triggered 여부도 같이 보여줘서, 후보 이미지가 crop 로직 때문인지
    한눈에 확인 가능 (4번 섹션에서 이미 crop 가설은 기각했지만 참고용으로 유지)
 
사용법:
  python compare_errors.py \
      --baseline /workspace/CountSE/inference_baseline_logging/per_image_errors.csv \
      --model    /workspace/CountSE/test_inf_normclip/per_image_errors.csv \
      --split val \
      --out gap_report_val.csv --top_k 5
 
  # test도 동일하게 --split test로 한 번 더 돌리고, 두 out CSV를 합쳐서 보면 됨
"""
 
import csv
import argparse
 
 
def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r["image_id"]] = {
                "gt_count": float(r["gt_count"]),
                "pred_cnt": float(r["pred_cnt"]),
                "abs_error": float(r["abs_error"]),
                "crop_triggered": int(r["crop_triggered"]),
            }
    return d
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="unknown", help="리포트에 태그로 남길 split 이름 (val/test)")
    ap.add_argument("--out", default="gap_report.csv")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()
 
    baseline = load(args.baseline)
    model = load(args.model)
 
    rows = []
    for image_id, m in model.items():
        if image_id not in baseline:
            continue
        b = baseline[image_id]
        gap = m["abs_error"] - b["abs_error"]  # 음수=개선, 양수=악화
        rows.append({
            "image_id": image_id,
            "split": args.split,
            "gt_count": b["gt_count"],
            "baseline_pred": b["pred_cnt"],
            "model_pred": m["pred_cnt"],
            "baseline_err": b["abs_error"],
            "model_err": m["abs_error"],
            "gap": gap,
            "crop_triggered": m["crop_triggered"],
        })
 
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{args.split}] wrote {len(rows)} rows to {args.out}\n")
 
    improved = sorted(rows, key=lambda r: r["gap"])[: args.top_k]
    worsened = sorted(rows, key=lambda r: -r["gap"])[: args.top_k]
 
    print(f"=== [{args.split}] TOP {args.top_k} 개선 (baseline 대비 오차 감소) ===")
    for r in improved:
        print(f"  {r['image_id']:>10} gt={r['gt_count']:6.0f} "
              f"base_err={r['baseline_err']:7.1f} model_err={r['model_err']:7.1f} "
              f"gap={r['gap']:8.1f} crop={r['crop_triggered']}")
 
    print(f"\n=== [{args.split}] TOP {args.top_k} 악화 (baseline 대비 오차 증가) ===")
    for r in worsened:
        print(f"  {r['image_id']:>10} gt={r['gt_count']:6.0f} "
              f"base_err={r['baseline_err']:7.1f} model_err={r['model_err']:7.1f} "
              f"gap={r['gap']:8.1f} crop={r['crop_triggered']}")
 
 
if __name__ == "__main__":
    main()