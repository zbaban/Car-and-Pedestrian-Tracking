import cv2

# Our Image
img_file = "Car_Image.jpg"

# Our pre-trained car classifier
classifier_file = "vehichle_detector.xml"

#create opencv image
img = cv2.imread(img_file)

#convert the image to black and white
graydout_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect cars
cars = car_tracker.detectMultiScale(graydout_img)

#Draw rectangles around the cars
#for (x, y, w, h) in cars:
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)


#Display the image with the faces spotted
cv2.imshow('Car Detector', img)

#Don 't autoclose(Wait here in the code and listen for a key press)
cv2.waitKey()


print ("code completed")