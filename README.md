

This is an implementation of Mobilenet-ssd model for object detection. In this project a pretrained caffe model is used. This project can be used for resource constraint devices such as mobiles, robots. 

Dependencies: Opencv3

Usage:

For image:
	
	python3 deep_learning_object_detection.py -i IMAGE_PATH -p CAFFE_MODEL_PROTOTXT_PATH -m CAFFE_MODEL_PATH -C CONFIDENCE_CUTOFF
	
Example:

	python3 deep_learning_object_detection.py -i 000275.jpg -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -c 0.9

For Video:
	
	python3 object_detection_video.py -v VIDEO_PATH -p CAFFE_MODEL_PROTOTXT_PATH -m CAFFE_MODEL_PATH -C CONFIDENCE_CUTOFF
	
Example:

	python3 object_detection_video.py -v Horse.mp4 -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -c 0.9

For Live:

	python3 object_detection_video.py -p CAFFE_MODEL_PROTOTXT_PATH -m CAFFE_MODEL_PATH -C CONFIDENCE_CUTOFF

Result:

![Output](https://user-images.githubusercontent.com/42487965/118390143-2c7b1b80-b64b-11eb-9846-1fae5f2a0134.jpg)
