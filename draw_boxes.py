# draw_boxes.py
import json
from PIL import Image, ImageDraw

img = Image.open("/workspace/FSC147_384_V2/images_384_VarV2/7171.jpg").convert("RGB")
W, H = img.size
boxes = json.load(open("/workspace/CountSE/image_chacking_nodistil/debug_boxes_6003.json"))
print(f"box 개수: {len(boxes)}")

draw = ImageDraw.Draw(img)
for box in boxes:
    cx, cy, w, h = box  # GroundingDINO 계열은 보통 정규화 cxcywh
    x0, y0 = (cx - w/2) * W, (cy - h/2) * H
    x1, y1 = (cx + w/2) * W, (cy + h/2) * H
    draw.rectangle([x0, y0, x1, y1], outline="red", width=1)
img.save("debug_boxes_6003_visualized.png")