import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust(img, alpha=1.0, beta=0.0):
    """
    画像のコントラストと明るさを調整する
    input:
        img: 調整する画像データ
        alpha: コントラストの値
        beta: 明るさの値
    """

    # 積和演算を行う。
    dst = alpha * img + beta
    # 画素調整をした際に0 - 255の幅を超えているかもしれないので範囲内に収める
    dst_clip = np.clip(dst, 0, 255).astype(np.uint8)

    return dst_clip

def is_up_contrast(img_hist):
    """
    コントラストをこれ以上上げるかを判定する
    判定基準は以下：
        ヒストグラムの中で最も高い値が255の場合はこれ以上上げない(Falseを返す)
        それ以外の場合は更にコントラストを上げる(Trueを返す)

    input:
        img_hist: 画像のヒストグラムを表した配列
    """
    hist = [x[0] for x in img_hist]
    index = max(enumerate(hist), key = lambda x:x[1])[0]

    if index  == 255:
        return False
    else:
        return True

def is_down_contrast(img_hist):
    """
    コントラストをこれ以上上げるかを判定する
    判定基準は以下：
        [0 ~ 100]の合計値が全体の中で半数以上を占めている場合はこれ以上下げない(Falseを返す)
            - 全体的に暗めの書籍は暗めの色で濃淡を付けていることが多いので少し広い幅で判断する
        それ以外の場合は更にコントラストを下げる(Trueを返す)

    input:
        img_hist: 画像のヒストグラムを表した配列
    """

    hist = [x[0] for x in img_hist]
    hist = [sum(hist[:101])] + hist[101:]
    index = max(enumerate(hist), key = lambda x:x[1])[0]

    if index == 0:
        return False
    else:
        return True

def is_upper(img_hist):
    """
    127を境目にしてヒストグラムの分布をみて以下判断をする。
        1. 127以下が多い場合はコントラストを下げる
        2. 127よりも大きい値が多い場合はコントラストを上げる

    input: numpy.ndarray: 画像のヒストグラム
    """

    hist = [x[0] for x in img_hist]
    down_or_up = [0,0]

    for index, h in enumerate(hist):
        if index <= 127:
            down_or_up[0] += h
        else:
            down_or_up[1] += h

    if down_or_up[0] >= down_or_up[1]:
        return False
    else:
        return True

def calc_histgram(img,is_save=False ,graph_name="histgram.png"):
    """
    入力された画像のヒストグラムを作成する
    input:
        img: ヒストグラムを作成する元になる画像データ
        is_save: Trueならヒストグラムの結果を画像として保存する
        graph_name: ヒストグラムを画像として保存するときの名前
    """

    img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # ヒストグラムを画像として保存する
    if is_save:
        plt.plot(img_hist)
        plt.savefig(graph_name)

    return img_hist

def trans_contrast(img):
    """
    入力された画像のコントラストを調整する

    input: numpy.ndarray : コントラストを調整する画像のデータ
    """

    """
    コントラスト調整の前に以下の判断をする必要がある
        1. 全体的に暗めの画像の場合はコントラストを下げて黒を締める
        2. 全体が明るめの場合はコントラストを上げて白を目立たせたい
    そのために、ヒストグラムから1か2のどちらかを判断する
    """

    hist = calc_histgram(img)
    is_up = is_upper(hist)
    alpha = 1.0

    if is_up:
        while(is_up_contrast(hist)):
            alpha += 0.01
            img = adjust(img, alpha, -8.0)
            hist = calc_histgram(img)
    else:
        while(is_down_contrast(hist)):
            alpha -= 0.01
            img = adjust(img, alpha, -5.0)
            hist = calc_histgram(img)

    return img
