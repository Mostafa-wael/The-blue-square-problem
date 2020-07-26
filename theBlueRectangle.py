import cv2
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
import numpy as np

def midPoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def drawRedCorners(sourceImg):
    #  recognizing the square by drawing red circles on its corners
    #  I  estimated those numbers from the image
    sqrCoord = np.array([[282, 216], [348, 188], [429, 222], [366, 263]], dtype=np.float32)  # square coordinates
    for pt in sqrCoord:
      cv2.circle(sourceImg, tuple(pt.astype(np.int)), 1, (0, 0, 255), -1)  # drawing red circles with radius 1 on the square corners
    cv2.imshow('source', sourceImg)
    return sourceImg, sqrCoord

def compTransMtrx (sourceImg,sqrCoord,startX,startY,step):  # step is the width of the figure
    # compute transformation matrix and apply it
    newCoord = np.array([[startX, startY], [startX + step, startX], [startX + step, startY + step], [startX, startY + step]], dtype=np.float32)  # new square coordinates
    transfMatrix = cv2.getPerspectiveTransform(sqrCoord, newCoord) # getting the transformation matrix
    modifiedImg = cv2.warpPerspective(sourceImg, transfMatrix, sourceImg.shape[:2][::-1])  # appying the transformation matrix  on the source image
    cv2.imshow('modified', modifiedImg)
    return modifiedImg

def detectCnt_DrawDist (img,refWidth,minCntSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting image to the grayScale to be able to get the Cnt
    gray = cv2.GaussianBlur(gray, (7, 7), 0)  # blurring the image slightly to avoid false lines

    edged = cv2.Canny(gray, 50, 100)  # edge detecting
    edged = cv2.dilate(edged, None,iterations=1)  # it is followed by the erosion process to increase the size of the erroted objects
    edged = cv2.erode(edged, None, iterations=1)  # erasing the noise like the small white and black points

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # finding the contours
    cnts = imutils.grab_contours(cnts)


    (cnts, _) = contours.sort_contours(cnts)  # sorting the contours from left-to-right

    refObj = None # setting reference object to be null

    
    for c in cnts: # loopping over each contour
        if cv2.contourArea(c) < minCntSize:  # ignoring insufficiently large contours
            continue
            
        # computing the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)  # since we use version 3
        box = np.int0(box)
        
        # ordering the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left order
        # then, drawing the outline of the rotated bounding box
        box = perspective.order_points(box)
        
        # computing the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        if refObj is None: # setting the first contour (on the top left) as the reference contour
            (topLeft, topRight, bottomRight, bottomLeft) = box
            # calculating the midPoints of the corners
            (tl_bl_x, tl_bl_y) = midPoint(topLeft, bottomLeft)
            (tr_br_x, tr_br_y) = midPoint(topRight, bottomRight)

            # computing the Euclidean distance between the midPoints
            D = dist.euclidean((tl_bl_x, tl_bl_y), (tr_br_x, tr_br_y))

            # constructing the reference object
            refObj = (box, (cX, cY), D / refWidth) # corner points, midpoint, distance between the midPoints i.e width
            continue

        # drawing the contours on the image
        orig = img.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 0, 0), 5)  # first contour
        cv2.drawContours(orig, [refObj[0].astype("int")], -1, (255, 255, 255), 5)  # second contour

        # drawing circles corresponding to the current points
        cv2.circle(orig, (int(refObj[1][0]),int(refObj[1][1])), 5, (100, 50, 100), -1)
        cv2.circle(orig, (int(cX), int(cY)), 5, (100, 50, 100), -1)

        # connecting them with a line
        cv2.line(orig, (int(refObj[1][0]),int(refObj[1][1])), (int(cX), int(cY)), (100, 50, 100), 2)

        # computing the Euclidean distance between the coordinates then, convert the distance in pixels to distance in units
        D = dist.euclidean(refObj[1], (cX, cY)) / refObj[2]

        (mX, mY) = midPoint(refObj[1], (cX, cY))  # writting the text in the mid distance between the two contours
        cv2.putText(orig, "{:.1f}cm".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 50, 100),2)

        # showing the output image
        cv2.imshow("Dimensions", orig)


source = cv2.imread('road_final.jpg')  # reading the image
x = drawRedCorners(source)  # the modified image and the coordinates of the square
size = 75 # square's width
detectCnt_DrawDist(compTransMtrx(x[0], x[1], 250, 250, size), size, 40)

cv2.waitKey()

