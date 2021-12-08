import random

import cv2

padding = 20
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']


class GenderDetection():
    def __init__(self):
        faceProto = 'data/opencv_face_detector.pbtxt'
        faceModel = 'data/opencv_face_detector_uint8.pb'
        genderProto = 'data/gender_deploy.prototxt'
        genderModel = 'data/gender_net.caffemodel'

        self.ans = [True, False]

        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)

    def highlightFace(self, net, frame, conf_threshold=0.9):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                     [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0),
                              int(round(frameHeight / 150)), 8)
        return frameOpencvDnn, faceBoxes

    # opencv
    def detect(self, img):
        try:
            resultImg, faceBoxes = self.highlightFace(self.faceNet, img)
            if not faceBoxes:
                return self.ans[random.randint(0, 1)]
            for faceBox in faceBoxes:
                if (max(faceBox) > 1024):
                    continue
                face = img[max(0, faceBox[1] -
                               padding):min(faceBox[3] +
                                            padding, img.shape[0] - 1),
                           max(0, faceBox[0] -
                               padding):min(faceBox[2] +
                                            padding, img.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face,
                                             1.0, (227, 227),
                                             MODEL_MEAN_VALUES,
                                             swapRB=False)
                self.genderNet.setInput(blob)
                genderPreds = self.genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                if (gender == 'Female'):
                    return True
                else:
                    return False
        except:  # isort:skip  # noqa
            return self.ans[random.randint(0, 1)]
