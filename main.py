import cv2
import mediapipe as mp

try:

    # MediaPipe Face Meshの初期化
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # MediaPipe描画ユーティリティの初期化
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    landmark_drawing_spec=drawing_spec
                )

        # ウィンドウに表示
        cv2.imshow('MediaPipe FaceMesh', image)

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
