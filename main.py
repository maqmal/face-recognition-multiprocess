import face_recognition
import cv2
import numpy as np
import os,glob
import time
from datetime import datetime
from multiprocessing import Process, Queue, Value, Lock, Array
import sharedmem

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
image_dir = '/'
image_array = []
registered_face = []
known_face_encodings = []
if os.path.exists(image_dir) and os.path.isdir(image_dir):  
    if not os.listdir(image_dir):
        print("IMAGE DIRECTORY NOT FOUND")
    else:
        for file in glob.glob("*.jpg"):
            new = file
            image_array.append(new)

for i in range(len(image_array)):
    registered_face.append(face_recognition.load_image_file(image_array[i]))

for x in range(len(image_array)):
    known_face_encodings.append(face_recognition.face_encodings(registered_face[x])[0])

known_face_names = [
    "Aqmal",
    "Jokowi",
    "Megawati"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
name = []

def grab_display(run_flag, send_frame_queue, receive_contour_queue,receive_name_queue,receive_face_queue, p_start_turn):
    last_contour_receive_time = 0
    startTime_ms = 0
    start_time = 0
    start_datetime = datetime.now()
    face_locations_shared = sharedmem.empty(face_locations)
    #receive_face_shared = sharedmem.empty(face_names)
    while (run_flag.value):
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        mask = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        current_time = datetime.now()
        delta_time = current_time-start_datetime
        delta_time_ms = delta_time.total_seconds()*1000

        # Only put frame in queue if it has past 30ms and exceeds blue threshold and there are fewer than 4 frames in queue
        if ((delta_time_ms > 30) and (send_frame_queue.qsize() < 4)):
            start_datetime = current_time # Update last send to queue time
            send_frame_queue.put(mask) # Put mask in queue
        #Check if receive_contour_queue is not empty contour = face_locations
        if ((not receive_contour_queue.empty())):
            last_contour_receive_time = time.time()
            face_locations_shared = receive_contour_queue.get()
            name = receive_name_queue.get()
            #print("INI SHARED =>",receive_face_shared)
            if ((time.time()-last_contour_receive_time) < 0.5):
                # Display the results
                for (top, right, bottom, left) in (face_locations_shared):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        k = cv2.waitKey(5) & 0xFF
        if k == ord('q'): # Press q to exit program safely
            run_flag.value = 0
            print("set run_flag --- 0")
        
    print("Quiting Main Processor 0")


def process_frame_1(run_flag, send_frame_queue, receive_contour_queue,receive_name_queue,receive_face_queue, p_start_turn):
    while (run_flag.value):
        startTime = datetime.now()
        startTime_ms = startTime.second *1000 + startTime.microsecond/1000
        # If frame queue not empty and it is Worker Process 1's turn
        if ((not send_frame_queue.empty()) and (p_start_turn.value == 1)):
            mask = send_frame_queue.get() # Grab a frame
            p_start_turn.value = 2 

            face_locations = face_recognition.face_locations(mask)
            face_encodings = face_recognition.face_encodings(mask, face_locations)
            name = "Unknown"
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
            receive_contour_queue.put(face_locations)
            receive_name_queue.put(name)
        else:
            time.sleep(0.03)
            currentTime = datetime.now()
            currentTime_ms = currentTime.second *1000 + currentTime.microsecond/1000
    print("Quiting Processor 1")

def process_frame_2(run_flag, send_frame_queue, receive_contour_queue,receive_name_queue,receive_face_queue, p_start_turn):
    while (run_flag.value):
        startTime = datetime.now()
        startTime_ms = startTime.second *1000 + startTime.microsecond/1000
        # If frame queue not empty and it is Worker Process 1's turn
        if ((not send_frame_queue.empty()) and (p_start_turn.value == 2)):
            mask = send_frame_queue.get() # Grab a frame
            p_start_turn.value = 3 

            face_locations = face_recognition.face_locations(mask)
            face_encodings = face_recognition.face_encodings(mask, face_locations)
            name = "Unknown"
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
            receive_contour_queue.put(face_locations)
            receive_name_queue.put(name)
        else:
            time.sleep(0.03)
            currentTime = datetime.now()
            currentTime_ms = currentTime.second *1000 + currentTime.microsecond/1000
    print("Quiting Processor 2")

def process_frame_3(run_flag, send_frame_queue, receive_contour_queue,receive_name_queue,receive_face_queue, p_start_turn):
    while (run_flag.value):
        startTime = datetime.now()
        startTime_ms = startTime.second *1000 + startTime.microsecond/1000
        # If frame queue not empty and it is Worker Process 1's turn
        if ((not send_frame_queue.empty()) and (p_start_turn.value == 3)):
            mask = send_frame_queue.get() # Grab a frame
            p_start_turn.value = 1 

            face_locations = face_recognition.face_locations(mask)
            face_encodings = face_recognition.face_encodings(mask, face_locations)
            name = "Unknown"
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
            receive_contour_queue.put(face_locations)
            receive_name_queue.put(name)
        else:
            time.sleep(0.03)
            currentTime = datetime.now()
            currentTime_ms = currentTime.second *1000 + currentTime.microsecond/1000
    print("Quiting Processor 3")


if __name__ == '__main__':
    # run_flag is used to safely exit all processes
    run_flag = Value('i', 1) 
    # p_start_turn is used to keep worker processes process in order
    p_start_turn = Value('i', 1)  
    send_frame_queue = Queue()
    receive_contour_queue = Queue()
    receive_name_queue = Queue()
    receive_face_queue = Queue()
    p0 = Process(target=grab_display, args=(run_flag, send_frame_queue, receive_contour_queue,receive_name_queue,receive_face_queue, p_start_turn))

    p1 = Process(target=process_frame_1, args=(run_flag, send_frame_queue, receive_contour_queue,receive_name_queue,receive_face_queue, p_start_turn))
    p2 = Process(target=process_frame_2, args=(run_flag, send_frame_queue, receive_contour_queue,receive_name_queue,receive_face_queue, p_start_turn))
    p3 = Process(target=process_frame_3, args=(run_flag, send_frame_queue, receive_contour_queue,receive_name_queue,receive_face_queue, p_start_turn))

    p0.start()
    p1.start()
    p2.start()
    p3.start()
    # Wait for four processes to safely exit
    p0.join()
    p1.join()
    p2.join()
    p3.join()

    cv2.destroyAllWindows()    
    video_capture.release()