import dlib
import numpy as np


def generate_face_correspondences(theImage1, theImage2):
    # Detect the landmarks of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    corresp = np.zeros((68, 2))

    imgList = [theImage1, theImage2]
    list1 = []
    list2 = []
    j = 1

    for img in imgList:

        size = (img.shape[0], img.shape[1])

        if j == 1:
            currList = list1
        else:
            currList = list2

        dets = detector(img, 1)

        j = j + 1

        for k, rect in enumerate(dets):

            # Get the landmarks of the face using the pretrained model
            shape = predictor(img, rect)

            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                currList.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y

    # Add back the background
    narray = corresp / 2

    return [size, imgList[0], imgList[1], list1, list2, narray]
