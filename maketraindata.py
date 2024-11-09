"""
    @author: Nguyen Hai Trieu <22110082@student.hcmute.edu.vn> 
"""
import cv2
import os

label = "500000"
cap = cv2.VideoCapture(0)
i = 0
while True:
    i += 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
    cv2.imshow('frame', frame)
    if 60 <= i <= 2060:
        print("Number of images that is captured = ", i - 60)
        directory = f'data/VNCurrency/currency_train/{label}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(f'{directory}/{i - 60}.png', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
