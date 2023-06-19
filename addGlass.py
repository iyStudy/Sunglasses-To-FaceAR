import cv2
import mediapipe as mp

try:

    # MediaPipe Face Meshの初期化
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # サングラス画像を読み込む
    sunglasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)  # PNGを透過チャンネルとして読み込む

    # カメラのキャプチャを開始
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # イメージをMediaPipeで処理
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # イメージをBGRに戻す
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 顔のランドマークを描画
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # サングラスを配置するための座標を取得
                left_eye_outer = (int(face_landmarks.landmark[33].x * image.shape[1]), int(face_landmarks.landmark[33].y * image.shape[0]))
                right_eye_outer = (int(face_landmarks.landmark[263].x * image.shape[1]), int(face_landmarks.landmark[263].y * image.shape[0]))

                # サングラスの幅と高さを計算
                sunglasses_width = int((right_eye_outer[0] - left_eye_outer[0]) * 1.6)  # 幅を60%大きくする
                aspect_ratio = sunglasses.shape[1] / sunglasses.shape[0]
                sunglasses_height = int(sunglasses_width / aspect_ratio)

                # サングラスをリサイズ
                resized_sunglasses = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height))

                # 目の中心を計算
                eye_center = ((left_eye_outer[0] + right_eye_outer[0]) // 2,
                            (left_eye_outer[1] + right_eye_outer[1]) // 2)

                # サングラスを配置する領域の左上の座標を計算
                roi_top_left = (eye_center[0] - resized_sunglasses.shape[1] // 2,  # 中心に配置
                                eye_center[1] - resized_sunglasses.shape[0] // 2)  # 中心より少し上に配置

                # ROIの座標を調整して画像の境界を超えないようにする
                roi_top_left = (max(roi_top_left[0], 0), max(roi_top_left[1], 0))
                roi_bottom_right = (min(roi_top_left[0] + sunglasses_width, image.shape[1]),
                                    min(roi_top_left[1] + sunglasses_height, image.shape[0]))

                # サングラス画像とROIのサイズを調整
                width = roi_bottom_right[0] - roi_top_left[0]
                height = roi_bottom_right[1] - roi_top_left[1]
                alpha = resized_sunglasses[:, :, 3] / 255.0
                alpha = alpha[:height, :width]
                resized_sunglasses = resized_sunglasses[:height, :width]

                roi = image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

                image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = \
                    (1.0 - alpha).reshape(alpha.shape[0], alpha.shape[1], 1) * roi + \
                    alpha.reshape(alpha.shape[0], alpha.shape[1], 1) * resized_sunglasses[:, :, :3]

                # 合成領域を抽出
                roi = image[roi_top_left[1]:roi_top_left[1] + sunglasses_height, roi_top_left[0]:roi_top_left[0] + sunglasses_width]



            # サングラス画像のアルファチャンネルを使用して画像を合成
            alpha = resized_sunglasses[:, :, 3] / 255.0
            
            # ROIの座標を調整して画像の境界を超えないようにする
            roi_top_left = (max(roi_top_left[0], 0), max(roi_top_left[1], 0))
            roi_bottom_right = (min(roi_top_left[0] + sunglasses_width, image.shape[1]),
                                min(roi_top_left[1] + sunglasses_height, image.shape[0]))
            
            # サングラス画像とROIのサイズを調整
            width = roi_bottom_right[0] - roi_top_left[0]
            height = roi_bottom_right[1] - roi_top_left[1]
            alpha = alpha[:height, :width]
            resized_sunglasses = resized_sunglasses[:height, :width]

            roi = image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

            image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = \
                (1.0 - alpha).reshape(alpha.shape[0], alpha.shape[1], 1) * roi + \
                alpha.reshape(alpha.shape[0], alpha.shape[1], 1) * resized_sunglasses[:, :, :3]



        # ウィンドウに表示
        cv2.imshow('MediaPipe FaceMesh with Sunglasses', image)

        # 'q' キーで終了
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"エラーが発生しました: {e}")
finally:
    # リソースを解放
    # キャプチャをリリースし、ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()
