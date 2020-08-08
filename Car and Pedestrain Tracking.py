import cv2

'''
# Our image
img_file = "car.png"

# Our pre-trained car classifier
classifier_file = 'cars_detector.xml'


# create opencv image
img = cv2.imread(img_file)

# create car classifier
car_traker = cv2.CascadeClassifier(classifier_file)


# Convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect cars
cars = car_traker.detectMultiScale(black_n_white)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# Display the image with the faces spotted
cv2.imshow('Car Detector', img)

# Don't autoclose
cv2.waitKey()

print("Code completed :)")
'''

car_traker = cv2.CascadeClassifier('cars_detector.xml')
pedestrain_tracker = cv2.CascadeClassifier('pedestrain.xml')

# video = cv2.VideoCapture('Pedestrians Compilation.mp4')
video = cv2.VideoCapture(
    'Tesla Autopilot Dashcam Compilation 2018 Version.mp4')

while True:

    # Read the current frame
    read_successful, frame = video.read()

    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars
    # , scaleFactor = 1.1, minNeighbors=2
    cars = car_traker.detectMultiScale(grayscaled_frame)

    # Detect pedestrains
    pedestrains = pedestrain_tracker.detectMultiScale(
        grayscaled_frame)  # , scaleFactor = 1.1, minNeighbors=2

    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw rectangles around the pedestrains
    for (x, y, w, h) in pedestrains:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Display the image with the cars & pedestrains spotted
    cv2.imshow('Self Driving Cars', frame)

    # Listen for a key press for 1 milliseconds, then move on
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
video.release()

print("Code Completed")
