import cv2
import mediapipe as mp
import math
import numpy as np
def detect_pose():
    mp_pose = mp.solutions.pose
    mp_hand = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # Eye landmark indices from MediaPipe Face Mesh
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144, 145, 159, 23]
    RIGHT_EYE_LANDMARKS = [263, 387, 385, 362, 373, 380, 374, 386, 253]
    
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Store previous eye positions
    prev_left_eye = None
    prev_right_eye = None
    movement_threshold = 5  # Minimum pixel movement to trigger warning

    cap = cv2.VideoCapture(1)

    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def point_to_line_distance(px, py, x1, y1, x2, y2):
        num = abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1))
        den = distance(x1, y1, x2, y2)
        return num / den if den != 0 else float('int')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        hand_result = hands.process(image)
        face_result=face_mesh.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            h, w, _ = image.shape

            # Draw pose landmarks
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                   mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

            # Get landmark positions
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            left_eye_inner = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER]
            right_eye_inner = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER]
            right_elbow= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]

            # Convert normalized coordinates to pixel values
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            left_ear_x, left_ear_y = int(left_ear.x * w), int(left_ear.y * h)
            right_ear_x, right_ear_y = int(right_ear.x * w), int(right_ear.y * h)
            left_eye_x, left_eye_y = int(left_eye_inner.x * w), int(left_eye_inner.y * h)
            right_eye_x, right_eye_y = int(right_eye_inner.x * w), int(right_eye_inner.y * h)
            right_elbow_x, right_elbow_y = int(right_elbow.x * w), int (right_elbow.y * h)
            left_elbow_x, left_elbow_y = int(left_elbow.x * w), int (left_elbow.y * h)


            # Compute distances from ears to the line between eyes and nose
            left_ear_dist = point_to_line_distance(left_ear_x, left_ear_y, left_eye_x, left_eye_y, nose_x, nose_y)
            right_ear_dist = point_to_line_distance(right_ear_x, right_ear_y, right_eye_x, right_eye_y, nose_x, nose_y)

            # Draw landmarks
            cv2.circle(image, (nose_x, nose_y), 5, (0, 0, 255), -1)
            cv2.circle(image, (left_ear_x, left_ear_y), 5, (255, 0, 0), -1)
            cv2.circle(image, (right_ear_x, right_ear_y), 5, (255, 0, 0), -1)
            cv2.line(image, (left_eye_x, left_eye_y), (nose_x, nose_y), (0, 255, 0), 2)
            cv2.line(image, (right_eye_x, right_eye_y), (nose_x, nose_y), (0, 255, 0), 2)

            # Condition to check if the ear touches the eye-nose line
            threshold = 10  # Adjust threshold as needed
            if left_ear_dist < threshold:
                cv2.putText(image, "Flag: Left ear touching eye-nose line", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if right_ear_dist < threshold:
                cv2.putText(image, "Flag: Right ear touching eye-nose line", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if right_elbow.visibility > 0.5 or left_elbow.visibility > 0.5:
                cv2.putText(image, "Flag: Elbow detected", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            

        # Draw hand landmarks if detected
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hand.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                
                if hand_landmarks:
                    cv2.putText(image, "Flag: Hand detected", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if face_result.multi_face_landmarks:
            
            for face_landmarks in face_result.multi_face_landmarks:
                mp_draw.draw_landmarks(
                     image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                     mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                     mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))
               # Get left and right eye landmark positions
                left_eye_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE_LANDMARKS]
                right_eye_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE_LANDMARKS]

                # Calculate the center of the left and right eye
                left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
                right_eye_center = np.mean(right_eye_points, axis=0).astype(int)

                # Draw left and right eye centers
                cv2.circle(image, tuple(left_eye_center), 3, (0, 255, 0), -1)
                cv2.circle(image, tuple(right_eye_center), 3, (0, 255, 0), -1)

                # Detect eye movement
                if prev_left_eye is not None and prev_right_eye is not None:
                    left_movement = np.linalg.norm(left_eye_center - prev_left_eye)
                    right_movement = np.linalg.norm(right_eye_center - prev_right_eye)

                    if left_movement > movement_threshold or right_movement > movement_threshold:
                        cv2.putText(image, "WARNING: Eye Movement Detected!", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Update previous eye positions
                prev_left_eye = left_eye_center
                prev_right_eye = right_eye_center
        else:
            cv2.putText(image, "No face detected", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)



        cv2.imshow('Pose Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_pose()


