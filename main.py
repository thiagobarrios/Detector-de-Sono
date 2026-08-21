import numpy as np
import dlib
import cv2
from math import hypot

cap = cv2.VideoCapture(0) # Inicializa a webcam

detector = dlib.get_frontal_face_detector() # Inicializa o detector facial do Dlib e o preditor de marcadores

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Arquivo externo...

def mid(p1, p2): # Funções auxiliares descritas no projeto, mas que faltavam no main.py original
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def eye_aspect_ratio(eye_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(eye_landmark[0]).x, face_roi_landmark.part(eye_landmark[0]).y)
    right_point = (face_roi_landmark.part(eye_landmark[3]).x, face_roi_landmark.part(eye_landmark[3]).y)
    center_top = mid(face_roi_landmark.part(eye_landmark[1]), face_roi_landmark.part(eye_landmark[2]))
    center_bottom = mid(face_roi_landmark.part(eye_landmark[5]), face_roi_landmark.part(eye_landmark[4]))
    
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    
    
    if ver_line_length == 0: # evita a divisão por zero se o olho fechar perfeitamente
        return 10.0 # Um valor alto para garantir que o alarme dispare
        
    ratio = hor_line_length / ver_line_length
    return ratio

def mouth_aspect_ratio(lips_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(lips_landmark[0]).x, face_roi_landmark.part(lips_landmark[0]).y)
    right_point = (face_roi_landmark.part(lips_landmark[2]).x, face_roi_landmark.part(lips_landmark[2]).y)
    center_top = (face_roi_landmark.part(lips_landmark[1]).x, face_roi_landmark.part(lips_landmark[1]).y)
    center_bottom = (face_roi_landmark.part(lips_landmark[3]).x, face_roi_landmark.part(lips_landmark[3]).y)
    
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    
    if hor_line_length == 0:
        return ver_line_length
    ratio = ver_line_length / hor_line_length
    return ratio

count = 0
font = cv2.FONT_HERSHEY_TRIPLEX

while True: # Loop de processamento de vídeo
    ret, img = cap.read()
    if not ret:
        break
        
    img = cv2.flip(img, 1) # Inverte a imagem (efeito espelho)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face_roi in faces:
        landmark_list = predictor(gray, face_roi)
        
        left_eye_ratio = eye_aspect_ratio([36, 37, 38, 39, 40, 41], landmark_list) # Análise dos olhos
        right_eye_ratio = eye_aspect_ratio([42, 43, 44, 45, 46, 47], landmark_list)
        eye_open_ratio = (left_eye_ratio + right_eye_ratio) / 2
        cv2.putText(img, f"Olhos: {eye_open_ratio:.2f}", (10, 30), font, 0.5, (235, 9, 9)) #BGR
        
        inner_lip_ratio = mouth_aspect_ratio([60,62,64,66], landmark_list) # Análise da boca (bocejo)
        outter_lip_ratio = mouth_aspect_ratio([48,51,54,57], landmark_list)
        mouth_open_ratio = (inner_lip_ratio + outter_lip_ratio) / 2
        cv2.putText(img, f"Boca: {mouth_open_ratio:.2f}", (448, 30), font, 0.5, (235, 9, 9))
        
        x, y = face_roi.left(), face_roi.top() # Coordenadas do rosto
        x1, y1 = face_roi.right(), face_roi.bottom()
        
######### Lógica de detecção (os valores podem precisar de ajuste dependendo da iluminação)#############
        if (mouth_open_ratio > 0.380 and eye_open_ratio > 3.5) or (eye_open_ratio > 3.9): # Se bocejar E fechar levemente o olho (>3.5), OU fechar o olho completamente (>3.8)
            count += 1
        else:
            count = 0
            
        if count > 10:
            cv2.rectangle(img, (x,y), (x1,y1), (0, 0, 255), 2)
            cv2.putText(img, "Com Sono", (x, y-5), font, 0.5, (0, 0, 255))

        else:
            cv2.rectangle(img, (x,y), (x1,y1), (0, 255, 0), 2)

            
   
    cv2.imshow("Detector de sono", img)  # Exibe a janela de vídeo
    
    
    key = cv2.waitKey(1) # Aguarda a tecla ESC ser pressionada para fechar
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()