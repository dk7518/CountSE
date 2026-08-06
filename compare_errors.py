# compare_errors.py
import sys, csv

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

baseline_path, model_path = sys.argv[1], sys.argv[2]
baseline = load(baseline_path)
model = load(model_path)

rows = []
for image_id in model:
    if image_id not in baseline:
        continue
    delta = model[image_id]["abs_error"] - baseline[image_id]["abs_error"]
    rows.append((image_id, delta, baseline[image_id]["abs_error"], model[image_id]["abs_error"],
                 baseline[image_id]["gt_count"], model[image_id]["crop_triggered"]))

rows.sort(key=lambda x: -x[1])  # 가장 많이 나빠진 순

print(f"{'image_id':>10} {'delta':>10} {'base_err':>10} {'model_err':>10} {'gt_count':>10} {'crop':>5}")
for r in rows[:30]:
    print(f"{r[0]:>10} {r[1]:>10.1f} {r[2]:>10.1f} {r[3]:>10.1f} {r[4]:>10.1f} {r[5]:>5}")

total_delta_sq = sum((model[k]["abs_error"]**2 - baseline[k]["abs_error"]**2) for k in model if k in baseline)
top30_delta_sq = sum((model[r[0]]["abs_error"]**2 - baseline[r[0]]["abs_error"]**2) for r in rows[:30])
print(f"\n상위 30개 이미지가 전체 제곱오차 증가분에서 차지하는 비중: {100*top30_delta_sq/total_delta_sq:.1f}%")


# python3 compare_errors.py /workspace/CountSE/inference_baseline_logging/per_image_errors.csv /workspace/CountSE/inference_nodistil_logging/per_image_errors.csv