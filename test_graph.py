import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('model_file3.h5')

video = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

# Initialize emotion counts
emotion_counts = {emotion: 0 for emotion in labels_dict.values()}

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img,(48,48))
        normalize = resized/255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        # Update the count of the predicted emotion
        emotion_counts[labels_dict[label]] += 1

        # Get the percentage of the predicted emotion
        emotion_probability = np.max(result)
        emotion_percentage = round(emotion_probability * 100, 2)

        print("Emotion: {}, Confidence: {}%".format(labels_dict[label], emotion_percentage))
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, "{}: {}%".format(labels_dict[label], emotion_percentage), (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow("Frame",frame)
    # k = cv2.waitKey(1)
    # if k == ord('q'):
    #     break
    k = cv2.waitKey(0) 
    if k == ord('q'):
        break


video.release()
cv2.destroyAllWindows()

# Plot the emotion counts
plt.figure(figsize=(10, 5))
plt.bar(emotion_counts.keys(), emotion_counts.values())
plt.title('Emotion Counts')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('emotion_counts.png')  # Save the plot to a file
plt.show()
