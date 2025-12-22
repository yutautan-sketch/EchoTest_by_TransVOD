import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# 楕円の近似周囲長（Ramanujan 第2式） from https://arxiv.org/pdf/math/0506384
# radii = (rad_x, rad_y)
def ellipse_perimeter(radii):
    print(radii)
    a, b = radii
    h = ((a - b) ** 2) / ((a + b) ** 2)
    # Ramanujan approximation
    perimeter = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
    return perimeter

def sample_contour_points(
    img, combined_contour, ellipse, 
    target_sample_count=70,
    distance_thresh=0.08,
    neighbor_ratio=0.05
    ):
    """楕円:ellipse上の点と輪郭:combined_contourを比較し、近いものとその近傍だけをサンプリングする"""
    # 0. 楕円の情報を取得
    center, axes, angle = ellipse
    
    # 1. 楕円のポイントリストを生成（cv2.ellipse2Polyは中心、半径、角度、開始角、終了角、間隔を指定）
    sampling_interval = max(1, len(combined_contour) // target_sample_count)
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
        img_h, img_w, _ = img.shape
        if x_dist <= img_w * distance_thresh and y_dist <= img_h * distance_thresh:
            filtered_points.append(pt)
            
            # サンプリングされなかった近傍点も含める: 前後にsampling_interval * neighbor_ratioの範囲
            start = max(i - int(sampling_interval * neighbor_ratio), 0)
            end = min(i + int(sampling_interval * neighbor_ratio) + 1, len(combined_contour))
            for j in range(start, end):
                candidate_pt = combined_contour[j][0]
                if not any(np.array_equal(candidate_pt, fp) for fp in filtered_points):
                    filtered_points.append(candidate_pt)
    
    return np.array(filtered_points).reshape((-1, 1, 2)) if len(filtered_points) > 0 else None

def body_detect(img,
                max_attempt,
                combine_num,
                mask_size, 
                mask_mode,
                debugmode):
    img_cp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    best_ellipse = None
    best_circle = None
    
    # ---- 1. ノイズ除去（平滑化）----
    blur = cv2.bilateralFilter(img, 9, 5, 5)  # 輪郭を保持した平滑化
    
    # ---- 2.1. 上下輝度補正（照明ムラ対策）----
    # 上部が暗く下部が明るい場合、背景補正を実施
    background = cv2.medianBlur(blur, 35)
    subtracted = cv2.subtract(blur, background)
    subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)

    # ---- 2.2 CLAHEによる局所コントラスト補正 ----
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(blur)
    _, clahe_mask = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---- 2.3 輝度補正 + CLAHE ----
    clahe_subtracted = clahe.apply(subtracted)
    _, clsb_mask = cv2.threshold(clahe_subtracted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ---- 2.4 CLAHE + 輝度補正二値化による補助マスク ----
    if mask_mode == 'ellipse':
        # 楕円マスク作成(各辺長の一定割合)
        center = (w // 2, h // 2)
        axes = (int(w * mask_size / 2), int(h * mask_size / 2))
        border = np.ones_like(img, np.uint8) * 255
        cv2.ellipse(border, center, axes, 0, 0, 360, 0, -1)  # 中心黒・外側白
    elif mask_mode == 'rectangle':
        # ---- 長方形マスク作成（各軸の9割）----
        margin_x = int(w * ((1-mask_size)/2))
        margin_y = int(h * ((1-mask_size)/2))
        border = np.ones_like(img, np.uint8) * 255
        cv2.rectangle(border, (margin_x, margin_y), (w - margin_x, h - margin_y), 0, -1)
    
    # 外周限定フィルタリング
    filtered_mask = clahe_mask.copy()
    filtered_mask[border == 255] = cv2.bitwise_and(clahe_mask, clsb_mask)[border == 255]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel=kernel)
    mask_img = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(mask_img, (center, (axes[0]*2, axes[1]*2), 0), (255, 0, 0), thickness=1)
    
    # ---- 3.1 楕円フィッティング ----
    attempt = 0
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_num = len(contours)
    while attempt < max_attempt and combine_num < contours_num+1:
        top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:min(combine_num, contours_num)]
        combined_contour = np.vstack(top_contours)
        ellipse = cv2.fitEllipse(combined_contour)
        
        if ellipse is not None:
            final_contour = sample_contour_points(img_cp, combined_contour, ellipse)
            if final_contour is not None:
                hull = cv2.convexHull(final_contour)
                if hull is not None and len(hull) >= 5:
                    elli_filtered = cv2.fitEllipse(hull)
                    if elli_filtered is not None:
                        center, axes, angle = elli_filtered
                        if all([
                            # 楕円の中心が画像の中心から大きくずれていない
                            abs(center[0] - w / 2) < w * 0.05,
                            abs(center[1] - h / 2) < h * 0.05,
                            # 楕円にフィットする矩形範囲が画像の面積に対して小さすぎない
                            (math.pi * max(axes) * min(axes)) / (w * h) >= 0.65,
                        ]):
                            if debugmode > 0:
                                cv2.drawContours(img_cp, top_contours, -1, (0, 0, 255), 2)
                                cv2.ellipse(img_cp, ellipse, (0, 255, 0), 2)
                                cv2.drawContours(img_cp, [final_contour], -1, (255, 0, 255), 2)
                            
                            # NOTE 楕円の長半径を半径に持つ円に変換 (精度が上昇するか一時的に試験中)
                            elli_filtered = (center, (max(axes), max(axes)), angle)
                            
                            cv2.ellipse(img_cp, elli_filtered, (0, 255, 255), 2)
                            best_ellipse = (center, (axes[0]/2, axes[1]/2), angle)
                            break
        
        combine_num += 1
        attempt += 1
        if debugmode > 0:
            print(f"combine_num -> {combine_num}")
            print(f"1: {abs(center[0] - w / 2)} < {w * 0.2}")
            print(f"2: {abs(center[1] - h / 2)} < {h * 0.2}")
            print(f"3: {(math.pi * max(axes) * min(axes)) / (w * h)} >= 0.63")
    
    # ---- 3.2 円フィッティング(Hough) ----
    circles = cv2.HoughCircles(
        filtered_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=50,
        param1=100,
        param2=20,
        minRadius=0,
        maxRadius=max(int(h/2), int(w/2))
    )
    if debugmode > 0:
        cv2.imwrite('clahe_mask.jpg', clahe_mask)
        cv2.imwrite('clsb_mask.jpg', clsb_mask)
        cv2.imwrite('filtered_mask.jpg', filtered_mask)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            if all([
                # 楕円の中心が画像の中心から大きくずれていない
                abs(x - w / 2) < w * 0.15,
                abs(y - h / 2) < h * 0.15,
                # 楕円にフィットする矩形範囲が画像の面積に対して小さすぎない
                (math.pi * r * r) / (w * h) >= 0.6,
            ]):
                cv2.circle(img_cp, (x, y), r, (255, 255, 0), 2)
                cv2.circle(img_cp, (x, y), 2, (0, 0, 255), 3)
                # print(f"腹部測定結果: ({x}, {y}), 半径={r}")
                best_circle = ((x, y), (r, r), 0.0)  # 楕円と型式を統一
                break
        return img_cp, best_ellipse, best_circle
    else:
        print("腹部の測定に失敗しました。")
        return None
