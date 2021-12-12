import cv2
import numpy as np

def make_sharp_kernel(k: int):
  return np.array([
    [-k / 9, -k / 9, -k / 9],
    [-k / 9, 1 + 8 * k / 9, k / 9],
    [-k / 9, -k / 9, -k / 9]
  ], np.float32)

def shapeness(img):
    """
    スキャンした書籍は文字や絵の境界がぶれてしまっている場合があるのでエッジを強調しておく
    """

    # シャープネスを上げるとノイズが強調されてしまうのでノイズを軽く取っておく
    img = cv2.bilateralFilter(img,50,30,20)
    cv2.imwrite("pic/befor_shapenes_denoise.png", img)

    kernel = make_sharp_kernel(1)
    img = cv2.filter2D(img, -1, kernel).astype("uint8")
    return img
