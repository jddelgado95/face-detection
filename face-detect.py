import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade using the cv2 class and CascadeClassifier method
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image and convert it to grayscale.
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
# detectMultiScale detects objects. Using on the face cascade, that’s what it detects
# function returns a list of rectangles in which it believes it found a face
faces = faceCascade.detectMultiScale(
    gray, # grayscale image. 
    scaleFactor=1.1, # Since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this
    minNeighbors=5, #detection algorithm uses a moving window to detect objects. This defines how many objects are detected near the current one before it declares the face found
    minSize=(30, 30), # size of each window
    #flags = cv2.CV_HAAR_SCALE_IMAGE
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)