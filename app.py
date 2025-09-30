import streamlit as st
import pandas as pd
import cv2
import numpy as np
import pickle
from datetime import datetime
import os

# --- Page Configuration ---
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("Face Recognition Attendance System")

# --- File and Folder Setup ---
# Create Attendance folder if it doesn't exist
if not os.path.isdir("Attendance"):
    os.makedirs("Attendance")

# Get today's date
ts = datetime.now().timestamp()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

# Define the attendance file for today
attendance_file = f"Attendance/Attendance_{date}.csv"

# Create the CSV file for today if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Time\n')

# --- Load Pre-trained Models ---
try:
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('data/faces.pkl', 'rb') as f:
        FACES = pickle.load(f)

    # Use a modern KNN model
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}. Please make sure 'names.pkl', 'faces.pkl', and 'haarcascade_frontalface_default.xml' are in the 'data' folder.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()


# --- Streamlit UI ---
col1, col2 = st.columns(2)

with col1:
    st.header("Live Camera Feed")
    FRAME_WINDOW = st.image([])
    run_camera = st.toggle('Start Camera', key="start_camera")


with col2:
    st.header("Attendance Record")
    
    # Function to update the attendance display
    def show_attendance():
        try:
            df = pd.read_csv(attendance_file)
            st.dataframe(df.style.highlight_max(axis=0))
        except pd.errors.EmptyDataError:
            st.write("No attendance recorded yet for today.")
    
    attendance_placeholder = st.empty()
    show_attendance()

    # --- New Feature: Remove Attendance ---
    st.subheader("Remove Attendance Record")
    with st.form(key='remove_form'):
        name_to_remove = st.text_input("Enter name to remove")
        submit_button = st.form_submit_button(label='Remove')

        if submit_button:
            if name_to_remove:
                try:
                    df = pd.read_csv(attendance_file)
                    if name_to_remove in df['Name'].values:
                        df = df[df['Name'] != name_to_remove] # Filter out the name
                        df.to_csv(attendance_file, index=False) # Save back to file
                        st.success(f"Attendance for {name_to_remove} removed successfully.")
                        st.rerun() # Rerun the app to show the updated table
                    else:
                        st.warning(f"'{name_to_remove}' not found in today's attendance record.")
                except pd.errors.EmptyDataError:
                    st.warning("Attendance file is empty. Nothing to remove.")
            else:
                st.warning("Please enter a name to remove.")


# --- Main Application Logic ---
if run_camera:
    video = cv2.VideoCapture(0)
    
    if not video.isOpened():
        st.error("Could not open webcam. Please check permissions and ensure it's not in use by another app.")
    else:
        while run_camera:
            ret, frame = video.read()
            if not ret or frame is None:
                st.error("Failed to capture image from camera. Stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                
                output = knn.predict(resized_img)
                
                ts = datetime.now().timestamp()
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                
                df = pd.read_csv(attendance_file)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if str(output[0]) in df['Name'].values:
                    cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
                    cv2.putText(frame, "Attendance Marked", (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                    cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    
                    with open(attendance_file, 'a') as f:
                        f.write(f'{str(output[0])},{timestamp}\n')

            FRAME_WINDOW.image(frame, channels="BGR")
            
            with attendance_placeholder.container():
                show_attendance()
        
        video.release()
else:
    st.write("Camera is off. Toggle the switch to start.")

