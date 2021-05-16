import cv2
import mediapipe as mp

drawing = mp.solutions.drawing_utils
hand = mp.solutions.hands

cap = cv2.VideoCapture(0)

with hand.Hands(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as hands:


    while True:
        success, image = cap.read()

        image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
        cv2.imshow("image",image)

        image.flags.writeable = False
        results = hands.process(image)

        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bottom = [5,9,13,17]
        if results.multi_hand_landmarks:
            # print(results.multi_hand_landmarks)
            for hand_marks in results.multi_hand_landmarks:
                drawing.draw_landmarks(
                    image, hand_marks, hand.HAND_CONNECTIONS
                )

                for id, landmark in enumerate(hand_marks.landmark):
                    height, width, channel = image.shape

                    x,y = int(landmark.x * width), int(landmark.y * height)
                    # print(id,(x,y))

                    # if id in bottom:
                    cv2.putText(image, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,0,0), 1)


        
        cv2.imshow("hand_land_marks",image)



        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()    