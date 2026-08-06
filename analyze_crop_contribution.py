# analyze_crop_contribution.py
import sys
import csv
from collections import defaultdict

def analyze(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "image_id": r["image_id"],
                "abs_error": float(r["abs_error"]),
                "crop_triggered": int(r["crop_triggered"]),
            })

    n_total = len(rows)
    crop_rows = [r for r in rows if r["crop_triggered"] == 1]
    non_crop_rows = [r for r in rows if r["crop_triggered"] == 0]

    def rmse(subset):
        if not subset:
            return 0.0
        return (sum(r["abs_error"]**2 for r in subset) / len(subset)) ** 0.5

    def mae(subset):
        if not subset:
            return 0.0
        return sum(r["abs_error"] for r in subset) / len(subset)

    total_sq_error = sum(r["abs_error"]**2 for r in rows)
    crop_sq_error = sum(r["abs_error"]**2 for r in crop_rows)

    print(f"=== {csv_path} ===")
    print(f"전체 이미지 수: {n_total}")
    print(f"crop_triggered=1 이미지 수: {len(crop_rows)} ({100*len(crop_rows)/n_total:.1f}%)")
    print(f"전체 RMSE: {rmse(rows):.2f}")
    print(f"  - crop 이미지만 RMSE: {rmse(crop_rows):.2f}")
    print(f"  - non-crop 이미지만 RMSE: {rmse(non_crop_rows):.2f}")
    print(f"전체 MAE: {mae(rows):.2f}")
    print(f"  - crop 이미지만 MAE: {mae(crop_rows):.2f}")
    print(f"  - non-crop 이미지만 MAE: {mae(non_crop_rows):.2f}")
    print(f"crop 이미지가 전체 '제곱오차 합'에서 차지하는 비중: {100*crop_sq_error/total_sq_error:.1f}%")
    print()

if __name__ == "__main__":
    for path in sys.argv[1:]:
        analyze(path)
        
# python3 analyze_crop_contribution.py /workspace/CountSE/inference_nodistil_logging/per_image_errors.csv