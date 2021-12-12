import cv2
import numpy as np
import math

from statistics import pstdev
from skimage import filters, img_as_ubyte

def calc_tilt(x1, y1, x2, y2):
    """
    2点で構成される直線の傾きから角度を算出する(degree形式)
    """
    rad = math.degrees(math.atan2((y2-y1), (x2-x1)))
    return rad

def reverse_hough_trans(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    return x1,y1, x2,y2

def chose_line(degrees):
    """
    hough変換によって求められた直線の傾きから補正角を選ぶために以下アルゴリズムを採用
        1. それぞれの直線の傾きを用意する
        2. 順番に1つずつ省いては標準偏差を求める
        3. 2のうち最も標準偏差の低くなった時に省いた傾きをoutlierと判断し、除去する
        4. 2-3を繰り返し、閾値以下の標準偏差になるか傾きの数が2つになったら打ち切る
        5. 残った傾きのうち最も傾きが小さいものを拾う
    
    ※アルゴリズムは以下URLを参考
    　https://qiita.com/suzuna-honda/items/32920191f775cb2f26bf#%E5%A4%96%E3%82%8C%E5%80%A4%E6%A4%9C%E5%87%BA

    ただし、上記アルゴリズムは線の本数が多い場合計算量が高いので考える必要あり
        計算量[多分]：(n ** 2) / 2
    """

    # そのまま触ると参照渡しの値を操作することになるのでコピーしておく
    degrees = degrees.copy()
    thresh = 0.1

    # 要素が2以下になるまで繰り返す
    while(3 <= len(degrees)):
        min_std = 9e9
        min_index = -1
        for index, line in enumerate(degrees):
            # 角度が入っている配列を直接操作したくないのでコピーを作成
            lines_copy = degrees.copy()

            # コピーされた配列からひとつだけ要素を削除する
            del lines_copy[index]

            # 標準偏差を計算する
            std = pstdev(lines_copy)

            # もし計算した標準偏差の値が他のインデックスを削除したときよりも多かった場合
            # 最小標準偏差とそのインデックス番号を更新する。
            if std <= min_std:
                min_std = std
                min_index = index
        
        # 最も偏差から離れていた角度を元の配列から削除
        del degrees[min_index]

        # 全体の標準偏差が閾値よりも低ければループを終了する
        std = pstdev(degrees)
        if std <= thresh:
            break

    # 残った角度の中から最も値が小さい角度を返却する
    return min(degrees)

def tilt_correction(img):

    # ノイズ低減のためにリサイズ
    height = img.shape[0]
    width = img.shape[1]
    dst = cv2.resize(img, (int(width * 0.8), (int(height * 0.8))))

    # ガウシアンフィルターでノイズを減らす
    edge = cv2.GaussianBlur(dst,(5,5),0)
    cv2.imwrite("pic/rotate/gausian.png", edge)

    # ラプラシアンフィルターでエッジ強調を行う
    edge = cv2.Laplacian(edge,cv2.CV_8U)
    cv2.imwrite("pic/rotate/laplacian.png", edge)

    # hysteresisで2値化を行う
    # このとき、邪魔になりそうな線を消しておきたい
    edge = filters.apply_hysteresis_threshold(edge, 30, 60)
    edge = img_as_ubyte(edge)
    cv2.imwrite("pic/rotate/hysteresis.png", edge)

    # hough変換で直線を求める
    lines = cv2.HoughLines(edge,1,np.pi/360,200, (-1 * math.pi)/4)

    # 直線が存在しない場合はそのまま処理を終了
    if lines is None:
        print("line is none")
        return img

    lines_rad = []
    for line in lines:
        rho, theta = line[0]
        x1, y1, x2, y2 = reverse_hough_trans(rho=rho, theta=theta)
        lines_rad.append(calc_tilt(x1, y1, x2, y2))
        red_line_img = cv2.line(edge, (x1,y1), (x2,y2), (255,255,255), 1)
    cv2.imwrite("pic/rotate/detect_line.png", red_line_img)

    # 補正角を取得
    rad = chose_line(lines_rad)

    # アフィン変換行列を作成する
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rad, scale=1)
    rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height), borderValue=(255, 255, 255))

    return rotated_image
