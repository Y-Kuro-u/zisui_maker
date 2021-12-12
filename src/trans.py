import numpy as np
import cv2

def outer_trim(img):
    trim_range = []
    for _ in range(4):
        prev_max = 9e10
        prev_var = 9e10
        for index, i in enumerate(img):

            # 着目しているラインの最大値と分散を取得
            now_max = int(max(i))
            now_var = float(np.var(i))

            # 一つ前のラインの最大値・分散と現在のラインの値を比較して変化がほとんどなければそこで切るとる
            if (0 <= (prev_max - now_max) <= 1.0) and (0 <= (prev_var - now_var) <= 1.0) and now_max == 255:
                trim_range.append(index)
                break

            # ラインの最大値と分散を保持しておく
            prev_max = max(i)
            prev_var = np.var(i)
        
        img = np.rot90(img)

    # 四方向すべてのラインが取れなかった場合は切り取らずに終了
    if len(trim_range) != 4:
        return img

    trimed_img = img.copy()
    for i in range(4):
        # 上を切り取る
        trimed_img = trimed_img[trim_range[i]:]
        trimed_img = np.rot90(trimed_img)

    return trimed_img