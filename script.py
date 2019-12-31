import cv2


imagePath = "img/example.jpg" # Insert the name of your picture 
cascadeClassifierPath = "haarcascade_frontalface_alt.xml"

cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)

image = cv2.imread(imagePath) 

grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

detectedFaces = cascadeClassifier.detectMultiScale(grayImage) 

for(x,y,width,height) in detectedFaces:
    cv2.rectangle(image, (x,y), (x+width, y+height), (0,255 ,0),5)

cv2.imwrite('result_cv2.jpg', image) #Result 
