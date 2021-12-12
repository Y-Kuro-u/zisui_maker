import cv2
from contrast import trans_contrast, calc_histgram
from shapenes import shapeness
from correction import tilt_correction
from trans import outer_trim

# Grayスケールでの読み込み
img = cv2.imread("./sample/zisui.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imwrite("pic/origin_gray.png", gray)
_ = calc_histgram(gray, is_save=True, graph_name="gray_scale_hist.png")

print("contrast ...")
# コントラストを調整
gray = trans_contrast(gray)
print("done ...")
cv2.imwrite("pic/after_contrast.png", gray)
_ = calc_histgram(gray, is_save=True, graph_name="after_contrast_hist.png")

print("shapeness ...")
# 画像の鋭利化
gray = shapeness(gray)
print("done ...")
cv2.imwrite("pic/after_shapenes.png", gray)

print("trim ...")
# 画像のトリミング
gray = outer_trim(gray)
print("done ...")
cv2.imwrite("pic/after_trim.png", gray)

print("rotate ...")
# 回転補正をかける
gray = tilt_correction(gray)
print("done ...")

cv2.imwrite("pic/output.png", gray)