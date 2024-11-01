import cv2
import os

label = "500000"

cap = cv2.VideoCapture(0)

# Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
i = 0
while True:
    # Capture frame-by-frame
    i += 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)

    # Hiển thị
    cv2.imshow('frame', frame)

    # Lưu dữ liệu
    if 60 <= i <= 2060:
        print("Số ảnh capture = ", i - 60)
        # Tạo thư mục nếu chưa có
        directory = f'data/VNCurrency/currency_train/{label}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        cv2.imwrite(f'{directory}/{i - 60}.png', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
