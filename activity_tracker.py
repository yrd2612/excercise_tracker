import colorsys
from turtle import color
import cv2
import numpy as np
import mediapipe as mp
counter = 0
stage = 0
mpPose = mp.solutions.pose
Pose = mpPose.Pose()
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
with mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        sucess, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        # drawing connections
        draw.draw_landmarks(image2, results.pose_landmarks,
                            mpPose.POSE_CONNECTIONS)

        # coordinate for left shoulder
        l_shldr_x = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER].x
        l_shldr_y = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER].y

        l_elbw_x = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW].x
        l_elbw_y = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW].y

        l_wrst_x = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST].x
        l_wrst_y = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST].y

        l_hip_x = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP].x
        l_hip_y = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP].y

        elbow = (l_elbw_x, l_elbw_y)
        body = (l_shldr_x,l_shldr_y)

        #print('left elbow : ',l_elbw_x,'\nleft shoulder',l_shldr_x,'\nleft wrist',l_wrst_x)

        a = np.array([l_shldr_x, l_shldr_y])
        b = np.array([l_elbw_x, l_elbw_y])
        c = np.array([l_wrst_x, l_wrst_y])
        d = np.array([l_hip_x, l_hip_y])

        ba = a - b
        bc = c - b
        ad = a - d
        ac = a - c

        # angle b/w wrist, elbow, shoulder
        cosine_angle_1 = np.dot(
            ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle_1 = np.arccos(cosine_angle_1)
        

        # angle b/w wrist,shoulder , hip
        cosine_angle_2 = np.dot(
            ad, ac) / (np.linalg.norm(ad) * np.linalg.norm(ac))
        angle_2 = np.arccos(cosine_angle_2)
        colors = [255, 255, 255]
        thickness = 2
        #print('Angle of upper body is : ', np.degrees(angle_1))
        #print('Angle of lower body is : ', np.degrees(angle_2))
        hand_angle = np.degrees(angle_1)
        body_angle = np.degrees(angle_2)
        cv2.putText(image2,str(hand_angle),
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
        cv2.putText(image2,str(body_angle),
                           tuple(np.multiply(body, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                                
        # Making counter
        
        if np.degrees(angle_2) > 160:
            stage = 1
        if np.degrees(angle_2) < 30 and stage == 1:
            if hand_angle > 160:
                stage = 0
                counter += 1
                # print(counter)
            
        org = (0,200) 
        cv2.putText(image2,str(counter),org,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)   
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        cv2.imshow('pose capture', image2)
        
print('The number of times you did excercise is : ',counter)

cap.release()
cv2.destroyAllWindows()

