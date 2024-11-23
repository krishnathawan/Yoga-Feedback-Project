import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./final_yoga_model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
cap.set(3,2000)
cap.set(4,2000)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)

ideal_angles_data = pickle.load(open('final_yoga_data.pickle', 'rb'))
ideal_angles = ideal_angles_data['avg_ideal_angles']

# converting angles to ranges to avoid negligible error of angles
ideal_ranges = {}
for key, angles in ideal_angles.items():
    ideal_ranges[key] = [(angle-18 , angle+18 ) for angle in angles]


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

labels_dict = {0: 'Tree', 1: 'Goddess', 2: 'Mountain',3: 'Chair Pose',4: 'Cobra Pose',5: 'Diamond Pose'}
pose_images = {
    'Tree': r"C:\Users\krish\tree_pose.jpeg",
    'Goddess':r"C:\Users\krish\Goddess.jpg",
    'Mountain': r"C:\Users\krish\mountain_pose.jpeg",
    'Chair Pose': r"C:\Users\krish\chair_pose.jpg",
    'Cobra Pose': r"C:\Users\krish\cobra_pose.jpeg",
    'Diamond Pose': r"C:\Users\krish\diamond_pose.jpg"
}

# ask user to select a pose
print("Select a yoga pose by entering the corresponding number:")
for key, label in labels_dict.items():
    print(f"{key}: {label}")

selected_pose_index = int(input("Enter the number of the yoga pose: "))
selected_pose_label = labels_dict[selected_pose_index]
reference_image = cv2.imread(pose_images[selected_pose_label])

while True:
    data_aux = []
    x_ = []
    y_ = []
    feedback_background = np.zeros((500, 500, 3), dtype=np.uint8)
    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,  
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS, 
            mp_drawing_styles.get_default_pose_landmarks_style())
        

        for i in range(len(results.pose_landmarks.landmark)):
            x = results.pose_landmarks.landmark[i].x
            y = results.pose_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = []

            
            
            # Shoulder Angles
            left_shoulder_angle = calculate_angle(
                (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            )
            angles.append(left_shoulder_angle)
            
            right_shoulder_angle = calculate_angle(
                (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
            )
            angles.append(right_shoulder_angle)

            # Knee Angles
            left_knee_angle = calculate_angle(
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
            )
            angles.append(left_knee_angle)

            right_knee_angle = calculate_angle(
                (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
            )
            angles.append(right_knee_angle)


            # Hip Angles
            left_hip_angle = calculate_angle(
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
            )
            angles.append(left_hip_angle)

            right_hip_angle = calculate_angle(
                (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
            )
            angles.append(right_hip_angle)

            # Elbow Angles
            left_elbow_angle = calculate_angle(
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
                (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)
            )
            angles.append(left_elbow_angle)

            right_elbow_angle = calculate_angle(
                (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
                (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
            )
            angles.append(right_elbow_angle)

            
            
        arr = ['left shoulder','right shoulder','left knee','right knee','left hip','right hip','left elbow','right elbow']  

        data_aux = angles 

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_pose = labels_dict[int(prediction[0])]

        probabilities = model.predict_proba([np.asarray(data_aux)])  
        confidence = np.max(probabilities) 
        

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        visible_landmarks = sum([1 for lm in landmarks if lm.visibility > 0.5])  # Visibility threshold = 0.5
        total_landmarks = len(landmarks)

        if visible_landmarks < total_landmarks * 0.9:  # Less than 90% of landmarks visible
            feedback = "Can't see complete body"
            cv2.putText(feedback_background, feedback, (11, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
        # we took threshold confidence as 65% 
        elif confidence > 0.65:
            cv2.putText(frame, f'{predicted_pose} ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            predicted_ideal_angles = ideal_ranges.get(str(int(prediction[0])), None)

            feedback = []
            if predicted_ideal_angles:
                for i, angle in enumerate(data_aux):
                    min_angle, max_angle = predicted_ideal_angles[i]
                    if angle < min_angle:
                        feedback.append(f'Increase angle of {arr[i]}')
                    elif angle > max_angle:
                        feedback.append(f'Decrease angle of {arr[i]}')
           

            
            if feedback:
                for i, comment in enumerate(feedback):
                    cv2.putText(frame, comment, (x1, y1 + 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Perfect Pose!", (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, "No pose", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    live_frame_resized = cv2.resize(frame, (500, 700))
    reference_image_resized = cv2.resize(reference_image, (500, 700))
    feedback_resized = cv2.resize(feedback_background, (500, 700))

    # concatenation of frames 
    composite_frame = cv2.hconcat([live_frame_resized,feedback_resized,reference_image_resized])

    # showing the combined frame 
    cv2.imshow('Yoga Pose Feedback', composite_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
