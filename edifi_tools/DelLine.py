import cv2
import os
import numpy as np

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}  # ".jpg", ".jpeg", ".png", ".bmp", ".tiff"

# モノクロかどうかの基準値
diff_bg = 15
diff_gr = 15
diff_br = 20

# 完全に黒か判断(仮)
threshold_black = 4

# 画像の読み込み場所
dir_path = 'root'
# 書き込み場所
write_path = 'output'
os.makedirs(write_path, exist_ok=True)

def DeleteLine(img_name):
    image = cv2.imread(os.path.join(dir_path, img_name))
    h, w = image.shape[:2]
    n = 0

    for i in range(h):
        for j in range(w):
            b, g, r = image[i, j]
            # モノクロ(r = g = b)なら通過
            if r == g == b:
                n = b
                continue
            
            # 影ではなく、後からつけられたと考えられる黒色箇所の消去(必要に応じて消去してください)
            # if r < threshold_black or g < threshold_black or b < threshold_black: 
            #     image[i, j] = (n, n, n)
            #     continue
            
            # ほぼモノクロと言えるか判定
            if max(b, g) - min(b, g) < diff_bg:  # |b - g|
                if max(g, r) - min(g, r) < diff_gr:  # |g - r|
                    if max(b, r) - min(b, r) < diff_br:  # |b - r|
                        
                        # 黄色っぽい部分の修正(必要に応じて消去してください)
                        # if g >= b:   
                        #     b = g + 5
                        #     image[i, j] = (b, g, r)
                            
                        n = np.mean([b, g, r])  # b, g, rの差が閾値以下ならモノクロとする
                        continue

            image[i, j] = (n, n, n)  # モノクロでないなら色を変更
                
    return image

for img_name in os.listdir(dir_path):
    if not any(img_name.lower().endswith(ext) for ext in valid_extensions):
        print(f"Skipping unsupported file: {img_name}")
        continue
    
    print(f"Processing: {img_name}")
    image = cv2.imread(os.path.join(dir_path, img_name))
    image = DeleteLine(img_name)
    new_img_name = os.path.splitext(img_name)[0] + ".jpg"
    cv2.imwrite(os.path.join(write_path, new_img_name), image)
    
print('Done.')