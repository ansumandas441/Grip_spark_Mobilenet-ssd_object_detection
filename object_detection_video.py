# import the necessary packages
import numpy as np
import argparse
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=False,
	help="path to input Video")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
if args["video"]:
    cap = cv2.VideoCapture(args["video"])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)


#prediction
print("[INFO] computing object detections...")


while True:
    ret, image = cap.read()
    if image is None:
        continue
    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
    	(300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
    	confidence = detections[0, 0, i, 2]
    	# filter out weak detections by ensuring the `confidence` is
    	# greater than the minimum confidence
    	if confidence > args["confidence"]:
    		# extract the index of the class label from the `detections`,
    		# then compute the (x, y)-coordinates of the bounding box for
    		# the object
    		idx = int(detections[0, 0, i, 1])
    		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    		(startX, startY, endX, endY) = box.astype("int")
    		# display the prediction
    		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
    		print("[INFO] {}".format(label))
    		cv2.rectangle(image, (startX, startY), (endX, endY),
    			COLORS[idx], 2)
    		y = startY - 15 if startY - 15 > 15 else startY + 15
    		cv2.putText(image, label, (startX, y),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)



    cv2.imshow('annotated', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

#usage for live
#python3 object_detection_video.py -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -c 0.9

#Usage for a video
#python3 object_detection_video.py -v Horse.mp4 -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -c 0.9
