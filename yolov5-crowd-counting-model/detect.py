import time
import os
import cv2
import tensorflow as tf

import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(20)


class Detector:
    def __init__(self):
        pass

    def readClassList(self, classFilePath):
        with open(classFilePath, 'r') as f:
            self.classList = f.read().splitlines()

            # Colors from np
            self.colorList = np.random.uniform(low=0, high=255, size=(
                len(self.classList), 3))

            # test print
            # print(len(self.classList), len(self.colorList))

    # define the __download traind model__ function
    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]

      #   print(fileName)
      #   print(self.modelName)
        self.cacheDir = './pretrained_models'

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName, origin=modelURL,
                 cache_dir=self.cacheDir, cache_subdir='checkpoints', extract=True)

    def loadModel(self):
        print('Loading model...'+self.modelName)

        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(
            self.cacheDir, 'checkpoints', self.modelName, 'saved_model'))

        print('model loaded successfully')

    def createBoundingBox(self, image, threshold=0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)

        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexs = detections['detection_classes'][0].numpy().astype(
            np.int32)
        classScores = detections['detection_scores'][0].numpy()

        # image height and width
        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(
            bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)
      #   print(bboxIdx)

        # location
        if len(bboxIdx) != 0:
            total_count_ppl = 0
            for i in bboxIdx:
                bbox = tuple(bboxs[i])
                classsConfidence = round(100*classScores[i])
                classIndex = classIndexs[i]

                classLabelText = self.classList[classIndex].upper()
                classColor = self.colorList[classIndex]

                displayText = '{}:{}%'.format(classLabelText, classsConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (
                    xmin*imW, xmax*imW, ymin*imH, ymax*imH)
                xmin, xmax, ymin, ymax = int(xmin), int(
                    xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymax), color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin-10),
                            cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                lineWidth = min(int((xmax-xmin)*0.2), int((ymax-ymin)*0.2))
                # Top border
                ###################################
                cv2.line(image, (xmin, ymin), (xmin+lineWidth, ymin),
                         classColor, thickness=5)
                cv2.line(image, (xmin, ymin), (xmin, ymin+lineWidth),
                         classColor, thickness=5)
                ###################################
                cv2.line(image, (xmax, ymin), (xmax-lineWidth, ymin),
                         classColor, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin+lineWidth),
                         classColor, thickness=5)

                # Bottom border
                ###################################
                cv2.line(image, (xmin, ymax), (xmin+lineWidth, ymax),
                         classColor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax-lineWidth),
                         classColor, thickness=5)
                ###################################
                cv2.line(image, (xmax, ymax), (xmax-lineWidth, ymax),
                         classColor, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax-lineWidth),
                         classColor, thickness=5)
                total_count += 1
        print(total_count_ppl)
        return image

    def predictImage(self, imgPath, threshold=0.5):
        image = cv2.imread(imgPath)

        bboxImage = self.createBoundingBox(image, threshold)

        cv2.imwrite(self.modelName + '.jpg', bboxImage)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    def predictVideo(self, videoPath, threshold=0.5):
        capture = cv2.VideoCapture(videoPath)

        if (capture.isOpened() == False):
            print("Error opening camera")

            return
        (success, image) = capture.read()

        startTime = 0
        while success:
            currentTime = time.time()

            fps = 1/(currentTime-startTime)
            startTimes = currentTime

            bboxImage = self.createBoundingBox(image, threshold)

            cv2.putText(bboxImage, "FPS:"+str(int(fps)), (20, 70),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            cv2.imshow('Result', bboxImage)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            (success, image) = capture.read()

        cv2.destroyAllWindows()
