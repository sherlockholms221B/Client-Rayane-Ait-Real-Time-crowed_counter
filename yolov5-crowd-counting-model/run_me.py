from detect import *

# pre-trained models
# modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'

modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz'
# classes
classFile = 'model_data/coco.names'
imgPath = 'test/FB_IMG_15898708394383294.jpg'
threshold = 0.5
videoPath = 0

detector = Detector()

detector.readClassList(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imgPath, threshold)
# detector.predictVideo(videoPath, threshold)
