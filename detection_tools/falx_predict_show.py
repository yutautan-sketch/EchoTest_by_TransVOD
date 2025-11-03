# 中心線の長さを求めて描画する

import numpy as np
import cv2 
import torch
import sys
import time
from matplotlib import pyplot as plt
from collections import Counter


# 検知した頭部を余白をつけてくり抜く
def crop_image_with_margin(image, bbox, w_margin_ratio=0.05, h_margin_ratio=0.05):
    """
    余白をつけて画像を切り抜く

    Parameters
    ----------
        image: ndarray
            入力画像
        bbox: tuple
            BBox
        margin_ratio: int
            余白サイズの割合

    Returns
    ----------
        cropped_image: numpy.ndarray 
            切り抜かれた画像
    """
    if image is None:
        raise FileNotFoundError("Image not found.")

    height, width = image.shape[:2]
    xmin, ymin, xmax, ymax = map(int, bbox)
    w_margin = int((xmax - xmin) * w_margin_ratio)
    h_margin = int((ymax - ymin) * h_margin_ratio)

    # Adjust the bounding box by adding margins
    xmin = max(0, xmin - w_margin)
    ymin = max(0, ymin - h_margin)
    xmax = min(width, xmax + w_margin)
    ymax = min(height, ymax + h_margin)

    # Crop the image
    cropped_image = image[ymin:ymax, xmin:xmax]

    return cropped_image


# 画像をコピーしてリサイズ
def resize_and_pad(img, target_width, target_height):
    """
    画像をコピーして、target_width * target_height の範囲にリサイズする。
    ただし、余白はグレーになる(グレーバックグラウンドに画像をリサイズ・貼り付ける)
    """
    
    image = img.copy()
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # グレーバックグラウンド作成
    padded = np.full((target_height, target_width, 3), 128, dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


# 楕円の情報を修正する
def update_ellipse(pts, ellipse, head_slope):
    """
    頭部の輪郭をとった楕円を、中心線から得た短径の情報をもとに修正する

    Parameters
    ----------
        pts: 
            短径の両端座標
        ellipse: 
            元の楕円
        angle: 
            中心線の傾き(タンジェント)
    
    Returns
    -------
        updated_ellipse: 
            修正した楕円
    """
    
    # 0. 頭部の傾き角度を取得
    angle = np.degrees(np.arctan(head_slope)) + 90
    
    # 1. 中心座標を計算
    (x1, y1), (x2, y2) = pts
    center_x = (x1 + x2) // 2  # 2点のx座標の平均
    center_y = (y1 + y2) // 2  # 2点のy座標の平均
        
    # 2. 短径を2点間の距離に変更
    short_diameter = int(np.linalg.norm(np.array([x2 - x1, y2 - y1])))
    if ellipse[1][0] >= ellipse[1][1]: 
        updated_axes = (ellipse[1][0], short_diameter)
    else: 
        updated_axes = (short_diameter, ellipse[1][1])

    # 3. selected_ellipseを更新
    updated_ellipse = ((center_x, center_y), 
                       updated_axes, 
                       angle)
    
    return updated_ellipse


# BBoxが楕円内にあるか判別する
def is_bb_inside_ellipse(ellipse, bb):
    """
    楕円:ellipseの内部にバウンディングボックス bb が完全に含まれるか判定する。
    
    Parameters
    ----------
        ellipse: 
            cv2.fitEllipse() の出力 (中心座標, 軸長, 回転角)
        bb: 
            バウンディングボックス (x_min, y_min, x_max, y_max)
    
    Returns
    ----------
        バウンディングボックスが楕円の内側に完全に含まれる場合 True, そうでなければ False
    """
    (cx, cy), (major_axis, minor_axis), angle = ellipse  # 楕円の情報を取得
    x_min, y_min, x_max, y_max = bb  # バウンディングボックスの座標

    # 楕円の半径
    a = major_axis / 2  # 長軸の半径
    b = minor_axis / 2  # 短軸の半径

    # バウンディングボックスの4つの頂点
    bb_points = np.array([
        [x_min, y_min],  # 左上
        [x_max, y_min],  # 右上
        [x_min, y_max],  # 左下
        [x_max, y_max]   # 右下
    ])

    # 楕円の回転角をラジアンに変換
    theta = np.radians(angle)

    # 回転行列を作成（楕円の軸に沿って座標を変換するため）
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # すべての頂点を楕円の中心基準で回転
    transformed_points = (bb_points - np.array([cx, cy])) @ rotation_matrix.T

    # すべての点が楕円の内部にあるかチェック
    inside = np.all((transformed_points[:, 0]**2 / a**2) + (transformed_points[:, 1]**2 / b**2) <= 1)

    return inside


# フィッティングされた楕円から短径を描画
def vis_short_diam(img, center, diam_length, tan_t):
    """
    楕円中心を通るように楕円の短径を描画する関数
    
    Parameters
    ----------
        img: [height, height, channel]
            入力画像
        center: (int, int)
            中心の座標
        diam_length: float
            短径の長さ
        tan_t: float
            傾きt[rad]に対するtan(t)の値
            = 中心線のBBoxの対角線の傾き

    Returns
    -------
        なし
    """
    
    # 画像の情報を取得
    h, w, _ = img.shape
    center_x, center_y = center
    rad_length = min((diam_length / 2), w, h)
    
    # tan(t)からcos(t)とsin(t)を算出
    t_magnitude = np.sqrt(1 + tan_t**2)
    cos_t = 1 / t_magnitude
    sin_t = tan_t / t_magnitude
    
    # 端点の座標数値を計算
    point_x1 = int(center_x - rad_length * sin_t)
    point_x2 = int(center_x + rad_length * sin_t)
    point_y1 = int(center_y - rad_length * cos_t)
    point_y2 = int(center_y + rad_length * cos_t)
    
    # 線の描画
    cv2.line(img, (point_x1, point_y2), (point_x2, point_y1), (255, 0, 255), thickness=2)


# 中心線を引く
def draw_fitted_line(img, center, x_range=20, max_attempt=3):
    """
    画像内の物体の輪郭を基に直線をフィットし、指定した中心(center)の±x_rangeの範囲で直線を描画する。

    Parameters
    ----------
        img: 
            入力画像
        center: 
            (x, y) 直線を描画する中心
        x_range: 
            描画する横幅(デフォルト ±20ピクセル)
        max_attempt: 
            最大試行回数(デフォルト 3回)
    
    Returns
    -------
        (x_start, y_start), (x_end, y_end), vy/vx
    """
    #img = img.copy()
    proc_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレースケール変換
    proc_img = cv2.GaussianBlur(proc_img, (5, 5), 1)  # ノイズ除去
    
    # Otsuの二値化による閾値決定
    otsu_thresh, binary = cv2.threshold(proc_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    lower = int(otsu_thresh + 20)  # Otsuの値を基に下限を決定
    upper = int(otsu_thresh + 50)  # 上限はOtsuの閾値
    
    kernel = np.ones((3,3), np.uint8)
    
    edges = cv2.Canny(proc_img, lower, upper)  # エッジ検出
    edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)  # クロージング処理
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None  # 輪郭が見つからない場合は終了
    
    # 面積が大きい順にソート
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    img_area = img.shape[0] * img.shape[1]  # 画像の総面積
    largest_width = 0
    attempt = 0
    target_idx = 0
    largest_contour = None
    
    while (attempt < max_attempt * 2 and target_idx < len(sorted_contours)):  # attempt < min(len(sorted_contours), max_attempt):
        # どうしても見つからない時のために、輪郭の再取得を追加
        if all([
            (attempt == max_attempt - 1 or attempt == len(sorted_contours)), 
            largest_width / img.shape[1] < 0.55
            ]): 
            
            # 閾値を緩和
            lower -= 35
            upper -= 30
            
            edges = cv2.Canny(proc_img, lower, upper)  # エッジ検出
            edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)  # クロージング処理
            
            # 輪郭を抽出
            # cv2.imshow('edges', edges)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return None  # 輪郭が見つからない場合は終了
            
            # 面積が大きい順にソート
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # target_idxとattemptを再設定
            target_idx = 0
        
        cnt = sorted_contours[target_idx]
        
        rect = cv2.minAreaRect(cnt)  # memo:250311 輪郭に対して、最もアスペクト比が大きくなるようにするにはどうするべき？考える必要あり
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(img, [box], 0, (255, 150, 55), 2)
        
        # 角度付き矩形の情報を取得
        height, width = rect[1]
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
        if attempt == 0: largest_width = width
        
        # アスペクト比が一定以上かつ横幅が画像の一定割合以上であることを確認
        if all([
            aspect_ratio >= 1.6, 
            #(width * height) / img_area >= 0.5, 
            width / img.shape[1] >= 0.55, 
            ]):
            largest_contour = cnt
            break
        
        attempt += 1
        target_idx += 1
    
    if largest_contour is None:
        return None  # 条件を満たす輪郭が見つからない場合は終了
        
    # 直線フィッティング
    [vx, vy, x0, y0] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # 指定した中心 `center` で x_start ~ x_end の範囲に制限
    x_start = int(center[0] - x_range)
    x_end = int(center[0] + x_range)
    
    # y 座標を計算（y = y0 + (x - x0) * vy / vx）
    y_start = int(y0 + (0 - x0) * vy / vx)
    y_end = int(y0 + (x_range * 2 - x0) * vy / vx)

    # 直線を描画
    cv2.drawContours(img, largest_contour, -1, (0, 255, 255), 2)
    cv2.line(img, (0, y_start), ((int)(x_range*2), y_end), (100, 155, 200), 2)
    # cv2.imshow("Fitted Line", img)
        
    return (x_start, y_start+int(center[1])), (x_end, y_end+int(center[1])), vy/vx


# 輪郭抽出から楕円フィッティングまで
class HeadTiltDetector:
    """
    頭の楕円フィッティングクラス
    
    Parameters
    ----------
        distance_thresh: int 
            より正確な楕円をフィッティングする時、最初の楕円からdistance_threshの範囲内の輪郭のみ用いる
        thresh_addition: int
            Canny法でエッジを得る時、Otsu_threshに加算する値: (lower, upper)
        target_sample_count: int
            より正確な楕円をフィッティングする時、調査する輪郭の数
        neighbor_ratio: float 
            より正確な楕円をフィッティングする時、調査する点の近傍をどれだけ含むか
        max_attempts: int
            楕円検出の最大試行回数
        aspect_ratio_thresh: float
            採用する輪郭のアスペクト比の閾値(これより大きいアスペクト比の輪郭は取り除く)
        fill_ratio_thresh: float
            輪郭矩形をほとんど埋めるような輪郭は、頭蓋骨の断面ではないと判断する
        combine_num: int 
            楕円検出に用いる輪郭数の初期値(上位いくつまでを用いるか)
    """
    def __init__(self, 
                 distance_thresh=0.1, 
                 thresh_addition=38, 
                 target_sample_count=70, 
                 neighbor_ratio=0.1, 
                 max_attempts=10, 
                 aspect_ratio_thresh=1.4, 
                 fill_ratio_thresh=0.35, 
                 combine_num=3, 
                 ):
        
        self.distance_thresh = distance_thresh          # サンプリングの判断とする距離
        self.thresh_addition = thresh_addition          # 二値化の閾値の調整値
        self.target_sample_count = target_sample_count  # 目標とするサンプリング数
        self.neighbor_ratio = neighbor_ratio            # サンプリング後に含む近傍点範囲
        self.max_attempts = max_attempts                # process_ellipse_detectionの最大試行回数
        self.aspect_ratio_thresh = aspect_ratio_thresh  # 縦横比の閾値
        self.fill_ratio_thresh = fill_ratio_thresh      # 塗りつぶし率の閾値
        self.combine_num = combine_num                  # 結合する輪郭数の初期値

    def fit_ellipse(self, contour):
        """輪郭に対して楕円フィッティングを行う"""
        return cv2.fitEllipse(contour) if len(contour) >= 5 else None
    
    def filter_valid_contours(self, contours):
        """入力された輪郭リストから細長い形状の輪郭を抽出し、ノイズを除去する"""
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 塗りつぶし率を計算（ほぼ白いノイズを除外）
            filled_ratio = cv2.contourArea(contour) / (w * h)
            if filled_ratio < self.fill_ratio_thresh:
                valid_contours.append(contour)
                continue  # ほぼ白く塗りつぶされている場合は除外

        return valid_contours

    def sample_contour_points(self, combined_contour, ellipse):
        """楕円:ellipse上の点と輪郭:combined_contourを比較し、近いものとその近傍だけをサンプリングする"""
        # 0. 楕円の情報を取得
        center, axes, angle = ellipse
        
        # 1. 楕円のポイントリストを生成（cv2.ellipse2Polyは中心、半径、角度、開始角、終了角、間隔を指定）
        sampling_interval = max(1, len(combined_contour) // self.target_sample_count)
        ellipse_points = cv2.ellipse2Poly(
            (int(center[0]), int(center[1])),
            (int(axes[0] / 2), int(axes[1] / 2)),
            int(angle), 0, 360, 1
        )
        
        # 2. 楕円上の各点の極角（中心から見た角度）を計算
        ellipse_angles = []
        for pt in ellipse_points:
                dx = pt[0] - center[0]
                dy = pt[1] - center[1]
                angle_pt = np.degrees(np.arctan2(dy, dx))
                ellipse_angles.append(angle_pt)
        
        # 3. 結合後の輪郭から一定間隔でサンプリングし、対応する楕円上の点との距離を計算
        filtered_points = []
        for i in range(0, len(combined_contour), sampling_interval):
            pt = combined_contour[i][0]  # サンプリングされた点
            # 楕円中心からサンプル点までの角度（度単位）を計算
            dx = pt[0] - center[0]
            dy = pt[1] - center[1]
            pt_angle = np.degrees(np.arctan2(dy, dx))
            
            # 4. 楕円上の点の中から、サンプル点の角度に最も近い点を選ぶ
            angle_diffs = np.abs(np.array(ellipse_angles) - pt_angle)
            min_index = int(np.argmin(angle_diffs))
            ellipse_pt = ellipse_points[min_index]
            
            # 5. サンプル点と対応する楕円上の点とのx座標・y座標距離をそれぞれ計算
            x_dist = pt[0] - ellipse_pt[0]
            y_dist = pt[1] - ellipse_pt[1]
            
            # 6. 距離が閾値以下ならフィルタリング対象に追加
            img_h, img_w, _ = self.img.shape
            if x_dist <= img_w * self.distance_thresh and y_dist <= img_h * self.distance_thresh:
                filtered_points.append(pt)
                
                # サンプリングされなかった近傍点も含める: 前後にsampling_interval * neighbor_ratioの範囲
                start = max(i - int(sampling_interval * self.neighbor_ratio), 0)
                end = min(i + int(sampling_interval * self.neighbor_ratio) + 1, len(combined_contour))
                for j in range(start, end):
                    candidate_pt = combined_contour[j][0]
                    if not any(np.array_equal(candidate_pt, fp) for fp in filtered_points):
                        filtered_points.append(candidate_pt)
        
        return np.array(filtered_points).reshape((-1, 1, 2)) if len(filtered_points) > 0 else None

    def process_ellipse_detection(self, debugmode):
        """楕円検出の試行を管理"""
        # 0. 画像情報読み込み
        img_h, img_w, _ = self.img.shape
        
        # 1. ノイズ除去と二値化
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 55, 55)
        
        # ---- 1.1 メディアンフィルタによる明るさムラ補正 ----
        background = cv2.medianBlur(blur, 35)
        subtracted = cv2.subtract(blur, background)
        subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
        
        # ---- 1.2 CLAHEによる局所コントラスト補正 ----
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(blur)
        
        # ---- 1.3 輝度補正 + CLAHE ----
        clahe_subtracted = clahe.apply(subtracted)
        
        # Otsuの二値化
        increased_thresh, binary = cv2.threshold(clahe_subtracted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        increased_thresh += self.thresh_addition  # 二値化の閾値を調整

        attempt = 0  # 試行回数の初期値
        contours_num = 0
        
        while attempt < self.max_attempts:
            # 2. モルフォロジー処理で輪郭補完
            _, binary = cv2.threshold(clahe_subtracted, increased_thresh, 255, cv2.THRESH_BINARY)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if debugmode==2:
                cv2.imwrite("binary.jpg", binary)
                input("Push any key to continue...")
            
            # 輪郭をフィルタリング
            valid_contours = self.filter_valid_contours(contours)
            contours_num = len(valid_contours)
            if contours_num == 0: 
                increased_thresh -= 10
                attempt += 1
                continue

            # 面積が大きい順にソートし、上位を取得して結合
            top_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:min(self.combine_num, contours_num)]
            combined_contour = np.vstack(top_contours) 
            
            if debugmode==2: 
                cv2.drawContours(self.img, [combined_contour], -1, (0, 255, 255), 4)
            
            # 3. 楕円フィッティング
            ellipse = self.fit_ellipse(combined_contour) 
            if ellipse:
                if debugmode==2:
                    print('DEBUG', ellipse) 
                    cv2.ellipse(self.img, ellipse, (0, 255, 0), 2)  # debug
                    cv2.imwrite("Head_Tilt_Detection.jpg", self.img)
                    # cv2.imshow("Head Tilt Detection", self.img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                
                # 楕円の面積と画像内に収まる部分の面積を計算
                center, axes, angle = ellipse
                
                # axes に NaN や inf が含まれていないかチェック
                if np.isnan(axes).any() or np.isinf(axes).any() or axes[0] <= 0 or axes[1] <= 0:
                    print(f"Invalid ellipse detected: center={center}, axes={axes}, angle={angle}")
                    increased_thresh -= 10
                    attempt += 1
                    continue
                
                ellipse_area = np.pi * (axes[0] / 2) * (axes[1] / 2)
                mask = np.zeros_like(gray)
                cv2.ellipse(mask, ellipse, 255, -1)
                inside_area = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=gray))
                
                # 画像外にはみ出すなら再試行
                if ellipse_area > 0 and (ellipse_area - inside_area) / ellipse_area <= 0.3:
                    # さらにいくつか条件づける(ハイパーパラメータ)
                    if any([
                        # アスペクト比が大きすぎる
                        max(axes) / min(axes) > self.aspect_ratio_thresh, 
                        # 楕円の中心が画像の中心から大きくずれている
                        abs(center[0] - img_w / 2) > img_w * 0.2,
                        abs(center[1] - img_h / 2) > img_h * 0.2,
                        # 楕円にフィットする矩形範囲が画像の面積に対して小さすぎる
                        (max(axes) * min(axes)) / (img_w * img_h) < 0.65,
                        ]): 
                        if debugmode==2:
                            print("1:", max(ellipse[1]) / min(ellipse[1]), ">", self.aspect_ratio_thresh)
                            print("2:", abs(center[0] - img_w / 2), ">", img_w * 0.2)
                            print("3:", abs(center[1] - img_h / 2), ">", img_h * 0.2)
                            print("4:", (max(ellipse[1]) * min(ellipse[1])) / (img_w * img_h), "<", 0.65)
                            input("Push any key to continue...")
                        if self.combine_num < contours_num:
                            self.combine_num += 1  # 他の輪郭を取り入れる
                        else: 
                            self.combine_num = 2  # 初期化
                            increased_thresh -= 10 
                        attempt += 1
                        continue
                
                else:
                    increased_thresh += 5
                    attempt += 1
                    continue
                
                # 結合後の輪郭から一定間隔でサンプリングし、対応する楕円上の点との距離を計算
                final_contour = self.sample_contour_points(combined_contour, ellipse)
                
                # サンプリングおよび補完した点があれば、それらをひとまとめにして最終輪郭として利用
                if final_contour is not None:
                    if debugmode==2: 
                        cv2.drawContours(self.img, [final_contour], -1, (255, 0, 255), 2)
                        cv2.imwrite("Head_Tilt_Detection.jpg", self.img)
                        input("Push any key to continue...")
                    hull = cv2.convexHull(final_contour)
                    ellipse = self.fit_ellipse(hull)
                    break
                else:
                    increased_thresh -= 10
                    attempt += 1
                    continue
            else: 
                increased_thresh -= 10
                attempt += 1
                continue
            
        return ellipse if attempt < self.max_attempts else None

    def detect_head_tilt(self, image, debugmode=0):
        """頭部フィッティングの実行関数"""
        # 表示用の画像
        self.img = image.copy()
        
        # 楕円の取得
        ellipse = self.process_ellipse_detection(debugmode=debugmode)
        if debugmode>0: 
            cv2.imwrite("Head_Tilt_Detection.jpg", self.img)
            # cv2.imshow("Head Tilt Detection", self.img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        if ellipse is None:
            print("適切な楕円が見つかりませんでした。")
            return None

        # 傾き判定
        center, axes, angle = ellipse
        if angle > 90:
            angle -= 180
        tilt_direction = "left" if angle > 0 else "right"

        cv2.ellipse(self.img, ellipse, (200, 100, 200), 2)
        
        # 傾きを短径で表示
        center_point = (int(center[0]), int(center[1]))
        length = int(axes[1] / 2)
        rad = np.deg2rad(angle)
        x_offset = int(length * np.cos(rad))
        y_offset = int(length * np.sin(rad))
        
        # 結果表示
        pt1 = (center_point[0] - x_offset, center_point[1] - y_offset)
        pt2 = (center_point[0] + x_offset, center_point[1] + y_offset)
        cv2.line(self.img, pt1, pt2, (255, 0, 0), 2)
        cv2.putText(self.img, tilt_direction, (center_point[0] - 50, center_point[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if debugmode>0: 
            cv2.imwrite("Head_Tilt_Detection.jpg", self.img)
            # cv2.imshow("Head Tilt Detection", self.img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        return ellipse, tilt_direction, self.img


# 短径を引くクラス
class ShortDimFinder():
    """
    短径の探索と描画
    
    Parameters
    ----------
        image: 
            入力画像
        steps: 
            閾値以上のピクセルを探索するときの間隔(ほぼstepsピクセルごとに閾値以上か確かめる)
        threshold: 
            輪郭と判断するピクセルの明るさの閾値(デフォルトではOtsuの閾値から自動で算出)
        decrease_factor: 
            短径探索の再試行時に、閾値を緩和する割合
    """
    def __init__(self, image, steps=10, threshold=None, decrease_factor=0.9):
        self.img = image
        self.height, self.width, _ = image.shape
        self.steps = steps
        self.dec_factor = decrease_factor
        
        # Otsuの閾値を使用
        if threshold is None:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            otsu_thresh, otsu_binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.threshold = otsu_thresh + 80
            # cv2.imshow("ShortDimFinder", otsu_binary_img)
            
        else:
            self.threshold = threshold
        
    def find_endpoint(self, cx, cy, dx, dy):
        """中心から直線上を進み、閾値以上のピクセルを探索"""
        
        x, y = cx, cy
        
        while 0 <= int(x) < self.width and 0 <= int(y) < self.height:
            pixel_value = self.img[int(y), int(x)]
            if np.mean(pixel_value) >= self.threshold:
                return int(x), int(y)
            x += dx
            y += dy
        
        return None
    
    def find_line(self, center, tan_t, approx_length):
        # tan_tに垂直な角度を取得（短径は楕円の向きに垂直）
        perp_angle = np.arctan(-1 / tan_t)

        # 角度から傾きを計算
        dx = np.cos(perp_angle)
        dy = np.sin(perp_angle)
        
        # 初期値をセット
        best_pair = None
        min_length_diff = float('inf')
        endpoint_pairs = []
        endpoint1, endpoint2 = None, None

        # 輪郭と判断する閾値を調整しながら端点を探索
        while self.threshold > 20:
            count = 0
            
            # 中心から探したとき、最初に見つかる端点を取得
            if endpoint1 is None:
                endpoint1 = self.find_endpoint(center[0], center[1], dx, dy)
            if endpoint2 is None:
                endpoint2 = self.find_endpoint(center[0], center[1], -dx, -dy)

            prev_ep1, prev_ep2 = endpoint1, endpoint2
            
            # 最適な端点を見つける
            while endpoint1 or endpoint2:
                # 最初から端点のどちらかが見つからなければbreak（閾値を下げる）
                if prev_ep1 is None or prev_ep2 is None:
                    break
                
                count += 1
                
                # 新たに端点を見つけられなかった場合、前回の情報を入れる
                if endpoint1 is None:
                    endpoint1 = prev_ep1
                if endpoint2 is None:
                    endpoint2 = prev_ep2
                    
                endpoint_pairs.append((endpoint1, endpoint2))
                length = np.linalg.norm(np.array(endpoint1) - np.array(endpoint2))  # 端点間距離=短径長を取得
                
                # 画像サイズから求めたおおよその短径長と比較し、最も誤差が小さいものを探す
                if all([
                    abs(length - approx_length) < abs(min_length_diff), 
                    abs(length - approx_length) < approx_length * 0.1, 
                    ]):
                    min_length_diff = length - approx_length
                    best_pair = (endpoint1, endpoint2)
                    count += 1
                    
                # 新たに、より外側の端点を探す
                prev_ep1, prev_ep2 = endpoint1, endpoint2
                endpoint1 = self.find_endpoint(endpoint1[0] + self.steps * dx, 
                                               endpoint1[1] + self.steps * dy, 
                                               dx, dy) if endpoint1 else None
                endpoint2 = self.find_endpoint(endpoint2[0] - self.steps * dx, 
                                               endpoint2[1] - self.steps * dy,
                                               -dx, -dy) if endpoint2 else None
            
            if best_pair: return best_pair, min_length_diff     # 単点のペアが存在すれば終了する
            self.threshold *= self.dec_factor                   # 閾値を緩和
            
        return best_pair, min_length_diff 

    def vis_line(self, center, tan_t, approx_length):
        result, diff = self.find_line(center, tan_t, approx_length)
        if result is None:
            print("線を描画できませんでした")
            return None
        
        (point1, point2) = result
        if all([
            point1 is not None, 
            point2 is not None, 
            ]):
            cv2.line(self.img, point1, point2, (0, 255, 255), thickness=2)
            return result
        else:
            print("線を描画できませんでした")
            return None


if __name__ == "__main__":
    # テスト画像のパスを指定
    image_path = "/home/kodaira/modeltest/TransVOD_Lite/detection_tools/20250625_162048_0670_00001.jpg"
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    # HeadTiltDetectorを初期化
    detector = HeadTiltDetector(
        distance_thresh=20,
        thresh_addition=70, # 38
        target_sample_count=70,
        neighbor_ratio=0.1,
        max_attempts=10,
        aspect_ratio_thresh=3.5,
        fill_ratio_thresh=0.8,
        combine_num=2
    )

    # 楕円フィッティングを実行（debugmode=1で途中結果も表示）
    result = detector.detect_head_tilt(image, debugmode=0)

    if result is not None:
        ellipse, tilt_direction, result_img = result
        print(f"楕円情報: {ellipse}")
        print(f"傾き方向: {tilt_direction}")

        # 結果を表示
        cv2.imwrite("head_tilt_detection_result.jpg", result_img)
    else:
        print("楕円フィッティングに失敗しました。")
