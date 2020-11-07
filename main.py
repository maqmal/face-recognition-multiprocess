import os,glob,time,sharedmem,cv2,face_recognition
import numpy as np
from datetime import datetime
from multiprocessing import Process, Queue, Value, Lock, Array

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

face_locations = []
face_encodings = []
face_names = []
def grab_display(run_flag, send_frame_queue, receive_location_queue,receive_name_queue, p_start_turn):
    last_location_receive_time = 0
    startTime_ms = 0
    start_time = 0
    start_datetime = datetime.now()
    face_locations_shared = sharedmem.empty(face_locations)
    font = cv2.FONT_HERSHEY_DUPLEX
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for_draw=""
    name="Unknown"
    while (run_flag.value):
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame_clone = frame.copy()
        
        gray = cv2.cvtColor(small_frame_clone, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for_draw = name
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, for_draw, (x,y-10), font, 0.7, (255, 255, 255), 1)

        mask = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        current_time = datetime.now()
        delta_time = current_time-start_datetime
        delta_time_ms = delta_time.total_seconds()*1000

        # Only put frame in queue if it has past 30ms and there are fewer than 5 frames in queue
        if ((delta_time_ms > 30) and (send_frame_queue.qsize() < 5)):
            start_datetime = current_time # Update last send to queue time
            send_frame_queue.put(mask) # Put mask in queue
        #Check if receive_location_queue is not empty
        if ((not receive_location_queue.empty())):
            last_location_receive_time = time.time()
            face_locations_shared = receive_location_queue.get()
            name = receive_name_queue.get()
            
            if ((time.time()-last_location_receive_time) < 0.5):
                # Display the results
                for_draw = name                    

        # Display the resulting image
        cv2.imshow('Video', frame)

        k = cv2.waitKey(5) & 0xFF
        if k == ord('q'): # Press q to exit program safely
            run_flag.value = 0
            print("set run_flag --- 0")
        
    print("Quiting Main Processor 0")
    

def process_frame_1(run_flag, send_frame_queue, receive_location_queue,receive_name_queue, p_start_turn):
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
                
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
            receive_location_queue.put(face_locations)
            receive_name_queue.put(name)
        else:
            time.sleep(0.03)
            currentTime = datetime.now()
            currentTime_ms = currentTime.second *1000 + currentTime.microsecond/1000
    print("Quiting Processor 1")

def process_frame_2(run_flag, send_frame_queue, receive_location_queue,receive_name_queue, p_start_turn):
    while (run_flag.value):
        startTime = datetime.now()
        startTime_ms = startTime.second *1000 + startTime.microsecond/1000
        
        if ((not send_frame_queue.empty()) and (p_start_turn.value == 2)):
            mask = send_frame_queue.get() # Grab a frame
            p_start_turn.value = 3 

            face_locations = face_recognition.face_locations(mask)
            face_encodings = face_recognition.face_encodings(mask, face_locations)
            name = "Unknown"
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
            receive_location_queue.put(face_locations)
            receive_name_queue.put(name)
        else:
            time.sleep(0.03)
            currentTime = datetime.now()
            currentTime_ms = currentTime.second *1000 + currentTime.microsecond/1000
    print("Quiting Processor 2")

def process_frame_3(run_flag, send_frame_queue, receive_location_queue,receive_name_queue, p_start_turn):
    while (run_flag.value):
        startTime = datetime.now()
        startTime_ms = startTime.second *1000 + startTime.microsecond/1000
        
        if ((not send_frame_queue.empty()) and (p_start_turn.value == 3)):
            mask = send_frame_queue.get() # Grab a frame
            p_start_turn.value = 4 

            face_locations = face_recognition.face_locations(mask)
            face_encodings = face_recognition.face_encodings(mask, face_locations)
            name = "Unknown"
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
            receive_location_queue.put(face_locations)
            receive_name_queue.put(name)
        else:
            time.sleep(0.03)
            currentTime = datetime.now()
            currentTime_ms = currentTime.second *1000 + currentTime.microsecond/1000
    print("Quiting Processor 3")

def process_frame_4(run_flag, send_frame_queue, receive_location_queue,receive_name_queue, p_start_turn):
    while (run_flag.value):
        startTime = datetime.now()
        startTime_ms = startTime.second *1000 + startTime.microsecond/1000
        
        if ((not send_frame_queue.empty()) and (p_start_turn.value == 4)):
            mask = send_frame_queue.get() # Grab a frame
            p_start_turn.value = 1 

            face_locations = face_recognition.face_locations(mask)
            face_encodings = face_recognition.face_encodings(mask, face_locations)
            name = "Unknown"
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
            receive_location_queue.put(face_locations)
            receive_name_queue.put(name)
        else:
            time.sleep(0.03)
            currentTime = datetime.now()
            currentTime_ms = currentTime.second *1000 + currentTime.microsecond/1000
    print("Quiting Processor 4")


if __name__ == '__main__':
    # run_flag is used to safely exit all processes
    run_flag = Value('i', 1) 
    # p_start_turn is used to keep worker processes process in order
    p_start_turn = Value('i', 1)  
    send_frame_queue = Queue()
    receive_location_queue = Queue()
    receive_name_queue = Queue()

    p0 = Process(target=grab_display, args=(run_flag, send_frame_queue, receive_location_queue,receive_name_queue, p_start_turn))

    p1 = Process(target=process_frame_1, args=(run_flag, send_frame_queue, receive_location_queue,receive_name_queue, p_start_turn))
    p2 = Process(target=process_frame_2, args=(run_flag, send_frame_queue, receive_location_queue,receive_name_queue, p_start_turn))
    p3 = Process(target=process_frame_3, args=(run_flag, send_frame_queue, receive_location_queue,receive_name_queue, p_start_turn))
    p4 = Process(target=process_frame_4, args=(run_flag, send_frame_queue, receive_location_queue,receive_name_queue, p_start_turn))

    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    
    p0.join()
    p1.join()
    p2.join()
    p3.join()
    p4.join()

    cv2.destroyAllWindows()    
    video_capture.release()

    