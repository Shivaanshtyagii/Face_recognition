import cv2
import numpy as np
import pickle
import os

# --- Initialization ---
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

name = input("Enter Your Name: ")

# --- Main Loop ---
while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Display feedback on the frame
    if len(faces) == 0:
        cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            # Crop and process the face image
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            
            # Only add data if we have less than 100 samples
            if len(faces_data) < 100:
                faces_data.append(resized_img)
                i += 1
                
                # Display the counter on the frame
                cv2.putText(frame, f"Collected: {i}/100", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)

            # Draw the rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    cv2.imshow("Frame", frame)
    
    # Check for exit key ('q') or if 100 faces are collected
    if cv2.waitKey(1) & 0xFF == ord('q') or i >= 100:
        break

video.release()
cv2.destroyAllWindows()

# --- Save the collected data ---
if len(faces_data) == 100:
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(100, -1)

    # Create the data directory if it doesn't exist
    if not os.path.isdir('data'):
        os.makedirs('data')

    # Load existing data, or initialize empty lists/arrays
    if 'names.pkl' in os.listdir('data/'):
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        with open('data/faces.pkl', 'rb') as f:
            faces = pickle.load(f)
    else:
        names = []
        # The shape should match the flattened image data: 50*50*3 = 7500
        faces = np.empty((0, 7500))

    # Check if the name already exists. If so, remove the old data before adding new.
    if name in names:
        print(f"Name '{name}' already exists. Overwriting previous data.")
        # Create a new list/array excluding the old data for this name
        new_names = []
        new_faces_indices = []
        for i, n in enumerate(names):
            if n != name:
                new_names.append(n)
                new_faces_indices.append(i)
        
        # Update the main variables with the filtered data
        names = new_names
        faces = faces[new_faces_indices]

    # Add the new data
    names.extend([name] * 100)
    faces = np.append(faces, faces_data, axis=0)

    # Save the final, corrected data back to the files
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('data/faces.pkl', 'wb') as f:
        pickle.dump(faces, f)
            
    print("Data saved successfully!")
else:
    print("Could not collect 100 face samples. Data not saved.")

