import cv2,time, pandas as pd
from datetime import datetime

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
first_frame = None
status_lst = [None,None]
time_lst_start = []
time_lst_end = []
df = pd.DataFrame(columns=["Start_Time","End_Time"])


while True:
    check, frame = video.read()
    status = 0

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    if first_frame is None:     #assign first frame to video
        first_frame = gray      #first frame should be background
        continue
    
    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame,30, 255, cv2.THRESH_BINARY)[1] #30 is threshold value and 255 is for color assigned to pixels >30

    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #detecting object greater than a particular size and updating status
    for contour in cnts:
        if cv2.contourArea(contour) < 10000: #change 10000 according to pixel size of object to be detected
            continue
        status = 1
        
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    
    #timestamping
    status_lst.append(status)
    if status_lst[-1] == 1 and status_lst[-2] == 0:
        time_lst_start.append(datetime.now())
    if status_lst[-1] == 0 and status_lst[-2] == 1:
        time_lst_end.append(datetime.now())

    
    cv2.imshow("Gray Frame",gray)                       #blurred frame
    cv2.imshow("Delta frame", delta_frame)              #frame showing difference between first frame and current frame
    cv2.imshow("Threshold frame", thresh_frame)         #frame depicting moving object as white and background as black    
    cv2.imshow("Color Frame", frame)                    #colored frame depicting contours obtained from threshold frame

    key = cv2.waitKey(1)
    
    if key==ord('q'):   # pressing 'q' button will end the process of capturing frames
        if status==1:
            time_lst_end.append(datetime.now())
        break

    
print(time_lst_start)
print(time_lst_end)

for i in range(0,len(time_lst_start)):
    #appending data of starting time and ending time of motion in dataframe
    df = df.append({"Start_Time":time_lst_start[i], "End_Time":time_lst_end[i]}, ignore_index=True)

#exporting dataframe to csv file  
a = time.strftime("%Y%m%d-%H%M%S")    #creating a var to store current time as string for filename
df.to_csv("{}.csv".format(a))

video.release()
cv2.destroyAllWindows()