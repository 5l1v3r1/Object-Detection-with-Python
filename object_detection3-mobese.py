#openCV Background Subtraction Model mobese_cam

import numpy as np
import cv2
import imutils

video = cv2.VideoCapture("test_videos/mobese.mp4")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
background_subtraction = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

#Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;

# Filter by Area.
params.filterByArea = True
params.minArea = 400

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else :
    detector = cv2.SimpleBlobDetector_create(params)

right=0
left=0

if video:

  while True:

    ret, frame = video.read()
    #new_video=imutils.rotate_bound(frame,0)
    #cv2.resizeWindow(window_name, (1800, 900)

    if ret:
        # Videoya Bacground Subtraction Uygula
        fgmask = background_subtraction.apply(frame)
        #threshold = cv2.adaptiveThreshold(fgmask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        #Red Line
        cv2.line(frame, (400, 180), (550, 180), (0, 0, 255), 5)
        #Blue Line
        cv2.line(frame, (130, 250), (360,250), (255, 0, 0), 5)

        im,contours, hierarchy = cv2.findContours(fgmask.copy(), mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
        try: hierarchy = hierarchy[0]
        except: hierarchy = []

        for contour, hier in zip(contours, hierarchy):
            (x,y,w,h) = cv2.boundingRect(contour)
            if w > 35 and h > 35:
                # Araçların Etrafına Çizgi Çek
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                #cv2.drawMarker(new_video, (x, y), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,line_type=cv2.LINE_AA)

                # To find centroid of the Car
                x1 = w / 2
                y1 = h / 2
                cx = x + x1
                cy = y + y1
                # Kütle Merkezi
                centroid = (cx, cy)
                # Kütle Merkezini Göster
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

                if centroid >= (150, 290) and centroid <= (350, 290):
                    cv2.line(frame, (130, 250), (360, 250), (0, 255, 0), 5)
                    #left+=1
                    # print(count)

                elif centroid >= (548,180) and centroid <= (550,180):
                    # Change Line Color
                    cv2.line(frame, (400, 180), (550, 180), (0, 255, 0), 5)
                    #count++
                    right+=1
                    #print(count)



    # Araç Sayma Ekranı
    cv2.putText(frame, "Number of Cars: " + str(right), (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Araç Sayma Ekranı
    cv2.putText(frame, "Number of Cars: " + str(left), (140, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Window Test", frame)
    cv2.imshow("Background", fgmask)

    key = cv2.waitKey(10)
    if key == ord('q'):
            break

frame.release()
cv2.destroyAllWindows()


