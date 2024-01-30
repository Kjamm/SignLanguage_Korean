import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time

max_num_hands = 1

gesture = {0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
           10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ',
           19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ', 24: 'ㅐ', 25: 'ㅔ', 26: 'ㅖ', 27: 'ㅆ'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5)

f = open('recognized_gestures.txt', 'w')
file = np.genfromtxt('dataSet.txt', delimiter=',')
angle_file = file[:, :-1]
label_file = file[:, -1]
angle = angle_file.astype(np.float32)
label = label_file.astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)
cap = cv2.VideoCapture(0)

start_time = time.time()
prev_index = 0
sentence = ''
recognize_delay = 1

while True:
    ret, img = cap.read()
    if not ret:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            compare_v1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
            compare_v2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            angle = np.arccos(np.einsum('nt,nt->n', compare_v1, compare_v2))
            angle = np.degrees(angle)
            if keyboard.is_pressed('a'):
                for num in angle:
                    num = round(num, 6)
                    f.write(str(num))
                    f.write(',')
                    f.write("27.000000")
                f.write('\n')

    cv2.imshow('HandTracking', img)

    if keyboard.is_pressed('b'):
        break

    data = np.array([angle], dtype=np.float32)
    ret, results, neighbours, dist = knn.findNearest(data, 3)
    index = int(results[0][0])
    if index in gesture.keys():
        if index != prev_index:
            start_time = time.time()
            prev_index = index
        else:
            if time.time() - start_time > recognize_delay:
                if index == 26:
                    sentence += ' '
                elif index == 27:
                    sentence = ''
                else:
                    sentence += gesture[index]
                start_time = time.time()
                cv2.putText(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] - 10),
                                                           int(res.landmark[0].y * img.shape[0] + 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=3)
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                cv2.putText(img, sentence, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=3)

f.close()
