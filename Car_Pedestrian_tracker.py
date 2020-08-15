import cv2

# Our Image
img_file = "Car_Image.jpg"

video = cv2.VideoCapture('NYC.mp4')

# Our pre-trained car classifier
vehichle_tracker_file = "vehichle_detector.xml"
pedestrian_tracker_file = "pedestrian_detector.xml"


#create car and pedestrains lassifier
car_tracker = cv2.CascadeClassifier(vehichle_tracker_file)

pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)



#Run continously
while True:

    #read the current frame
    (read_successful), frame = video.read()

    #safe coding
    if read_successful:
        #must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255,0,0), 2)

    #Draw rectangles around the cars
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,225,255), 2)

    #Display the image with the faces spotted
    cv2.imshow('Car Detector', frame)

    #Don 't autoclose(Wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    #Stop if Q pressed
    if key==81 or key==113:
        break 

#release video capture
video.release()
    


"""

#create opencv image
img = cv2.imread(img_file)

#convert the image to black and white
graydout_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect cars
cars = car_tracker.detectMultiScale(graydout_img)

#Draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)


#Display the image with the faces spotted
cv2.imshow('Car Detector', img)

#Don 't autoclose(Wait here in the code and listen for a key press)
cv2.waitKey()

"""
print ("code completed")