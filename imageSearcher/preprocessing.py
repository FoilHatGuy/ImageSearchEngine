import cv2 as cv
import numpy as np
import random as rng
from . import exceptions as ex


def imageSearchAndRestore(src, config):
    # src = cv.imread(name)
    if src is None:
        raise ex.PPDataMissing
    # Convert image to gray and blur it
    # src = cv.resize(src, )
    # cv.imshow('Source', src)

    # set variables
    resultSize = [240, 240]
    centerX = src.shape[1] / 2
    centerY = src.shape[0] / 2
    threshold = 110
    contrs_img = src.copy()
    hull_list = []
    max_hull = None
    centered = 1000000

    # further process of image
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.equalizeHist(src_gray)
    src_gray = cv.blur(src_gray, (3, 3))

    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 3)
    canny_output = cv.blur(canny_output, (3, 3))
    # cv.imshow('canny', canny_output)

    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find the convex hull object for each contour
    for i in range(len(contours)):
        if 3 < len(contours[i]) < 1500:
            hull = cv.convexHull(contours[i])
            epsilon = 0.1 * cv.arcLength(hull, True)
            hull = cv.approxPolyDP(hull, epsilon, True)
            hull_list.append(hull)
            bound = cv.boundingRect(hull)

            x, y, w, h = bound
            # how far is quadrangle from centre
            cnt = max(0, max((centerX - (x + w)), (x - centerX))) ** 2 + \
                  max(0, max((centerY - (y + h)), (y - centerY))) ** 2

            # check if it is large enough and closer than previous guess
            if w > centerX * 0.2 and h > centerY * 0.2 and cnt < centered and len(hull) == 4:
                # printing proportions
                prop_y = np.hypot(hull[1, 0, 1] - hull[0, 0, 1], hull[1, 0, 0] - hull[0, 0, 0]) / \
                         np.hypot(hull[3, 0, 1] - hull[2, 0, 1], hull[3, 0, 0] - hull[2, 0, 0])

                prop_x = np.hypot(hull[2, 0, 1] - hull[1, 0, 1], hull[2, 0, 0] - hull[1, 0, 0]) / \
                         np.hypot(hull[0, 0, 1] - hull[3, 0, 1], hull[0, 0, 0] - hull[3, 0, 0])
                # print(prop_x, prop_y, '\n')

                # draw painting nominee
                cv.drawContours(contrs_img, [hull], 0, (0, 255, 255))

                # check proportions of opposite sides
                if 0.6 < prop_x < 1.5 and 0.6 < prop_y < 1.5:
                    centered = cnt
                    max_hull = hull

    # print("centered:", centered)
    # Draw contours + hull results
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(hull_list)):
        color = (rng.randint(50, 256), rng.randint(50, 256), 0)
        cv.drawContours(drawing, hull_list, i, color)
    color = (0, 0, 255)
    cv.circle(drawing, (int(centerX), int(centerY)), 3, color, 2)
    if max_hull is not None:
        cv.drawContours(contrs_img, [max_hull], 0, (0, 0, 255))
        cv.drawContours(drawing, [max_hull], 0, (0, 0, 255))
        # Show in a window

        x, y, w, h = cv.boundingRect(max_hull)
        # print(max_hull)
        max_hull = np.roll(max_hull, -np.argmax(list(map(lambda el: el[0][0] * el[0][1], max_hull))), axis=0)
        # print([resultSize, [0, resultSize[1]], [0, 0], [resultSize[0], 0]])
        matrix = cv.getPerspectiveTransform(np.array(max_hull, np.float32),
                                            np.array([resultSize, [0, resultSize[1]],
                                                     [0, 0], [resultSize[0], 0]], np.float32))
        # print(matrix)
        new_img = cv.warpPerspective(src, matrix, (int(centerX) * 2, int(centerY) * 2))[0:resultSize[0], 0:resultSize[1]]
        # cv.imwrite(f"result-3.jpg", new_img)
        # cv.imshow('ContourMAX', contrs_img)
        # cv.imshow('Contours', drawing)
        return new_img
    else:
        raise ex.PPError
        # cv.imshow('ContourMAX', contrs_img)
        # cv.imshow('Contours', drawing)
