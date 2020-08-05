import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np


def midPoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def preparedImgForCntr(img):
    # preparing image for contour detction
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)  # converting image to the grayScale to be able to get the Cnt
    gray = cv2.GaussianBlur(gray, (7, 7), 0)  # blurring the image slightly to avoid false lines
    edged = cv2.Canny(gray, 50, 100)  # edge detecting
    edged = cv2.dilate(edged, None, iterations=1)  # it is followed by the erosion process to increase the size of the erroted objects
    edged = cv2.erode(edged, None, iterations=1)  # erasing the noise like the small white and black points
    return edged


def getSqrCoord (img):
    # getting the coordinates of the blue square
    image = preparedImgForCntr(img)
    cnts = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    copy = img.copy()
    square = None
    for c in cnts:
        epsilon = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * epsilon, True)
        if len(approx) == 4: # square has 4-points
            square = approx
            cv2.drawContours(copy, [square], -1, (0, 0, 0), 3)
            cv2.imshow("square", copy)
            square = np.array(square, dtype=np.float32)
            break  # if you found the square, break
    return square


def calcTransMtrx (sourceImg,sqrCoord,startX,startY,step):  # step is the width of the figure
    # compute transformation matrix and apply it
    # newCoord = topLeft, bottomLeft, bottomRight, topRight
    newCoord = np.array([[startX, startY], [startX, startY + step], [startX + step, startY + step], [startX + step, startX]], dtype=np.float32)  # new square coordinates
    transfMatrix = cv2.getPerspectiveTransform(sqrCoord, newCoord) # getting the transformation matrix
    modifiedImg = cv2.warpPerspective(sourceImg, transfMatrix, sourceImg.shape[:2][::-1])  # appying the transformation matrix on the source image

    newCoord = np.array(newCoord, dtype=int)
    return modifiedImg, newCoord


def regionOfInterest(img, vertices):
    mask = np.zeros_like(img)  # a zero matrix with the same size as the original image
    match_mask_color = (255, 255, 255)  # color to be put as a mask
    # cropping the region of interest by drawing a polygon with the passed vertices
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def cropRegionOfInterest(img, heightFactor, widthFactor):
    height, width, _ = img.shape
    # region_of_interest_vertices = [topLeft,bottomLeft ,topRight, bottomRight]
    RegionOfInterestList = [(width * widthFactor, 0), (width * widthFactor, height), (height * heightFactor, 0),
                            (height * heightFactor, height)]
    imagePrepared = preparedImgForCntr(img)
    croppedImage = regionOfInterest(imagePrepared, np.array([RegionOfInterestList], np.int32))
    return croppedImage


def drawLines(img, lines, thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), [255, 0, 0], thickness)
    # Merge the image with the lines onto the original, we could both functions but, addWeighted is better
    #img = cv2.add(img, line_img)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


def drawHoughLines (img):
    # detecting the lines
    lines = cv2.HoughLinesP(  # this function will return the lines holding line's points instead of rho and theta
        cropRegionOfInterest(img, 4/3, 1/3),
        rho=10,  # the length of the perpendicular line between the origin and the line
        theta=0.01,  # angle with the x-axis clockwise where, the x-axis is at the top of the image
        threshold=200,
        lines=np.array([]),
        minLineLength=170,  # minimum line length
        maxLineGap=10  # the max distance between two lines to treat them as two different lines
    )
    listOfLines = []
    if lines is None:
        raise Exception("No lines found")
    if len(lines) < 2:
        raise Exception("you can't calculate distance with less than two lines")

    print("number of lines:\n", len(lines))
    i = 0
    for line in lines:
        listOfPoints = []
        for x1, y1, x2, y2 in line:
            print("line number:", i, "[ x1:", x1, "x2:", x2, "y1:", y1, "y2:", y2, "]")
            listOfPoints.append(x1)
            listOfPoints.append(y1)
            listOfPoints.append(x2)
            listOfPoints.append(y2)
        listOfLines.append(listOfPoints)
        i += 1
    # drawing the lines
    line_image = drawLines(image, lines)
    print()
    return listOfLines, line_image


def calcDistancebetTwoLines (line1, line2 , Y, scale=1):
    # calc the distance between the 2-lines in the x direction at point Y
    slope1 = (line1[3] - line1[1]) / (line1[2] - line1[0])
    slope2 = (line2[3] - line2[1]) / (line2[2] - line2[0])
    # the value of X at height y
    XOfline1AtY = line1[0] + (Y - line1[1]) / slope1
    XOfline2AtY = line2[0] + (Y - line2[1]) / slope2

    distanceBetMidPoints = abs(XOfline1AtY - XOfline2AtY)
    # actual length is: length / scale
    distanceBetMidPoints /= scale
    return distanceBetMidPoints


def drawDistanceAtY (img, line1, line2, D, Y):
    # drawing the distance in meters at height = Y
    # drawing the contours on the image
    orig = img.copy()
    slope1 = (line1[3] - line1[1]) / (line1[2] - line1[0])
    slope2 = (line2[3] - line2[1]) / (line2[2] - line2[0])
    XOfline1AtY = int(line1[0] + (Y - line1[1]) / slope1)
    XOfline2AtY = int(line2[0] + (Y - line2[1]) / slope2)
    # drawing circles corresponding to the current points
    cv2.circle(orig, (XOfline1AtY, Y), 5, (100, 50, 100), -1)
    cv2.circle(orig, (XOfline2AtY, Y), 5, (100, 50, 100), -1)

    # connecting them with a line
    cv2.line(orig, (XOfline1AtY, Y), (XOfline2AtY, Y), (100, 50, 100), 2)

    (mX, mY) = midPoint((XOfline1AtY, Y), (XOfline2AtY, Y))  # writting the text in the mid distance between the two contours
    writeDistAboveLineBy = 10
    cv2.putText(orig, "{:.2f}m".format(D), (int(mX), int(mY - writeDistAboveLineBy)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (100, 50, 100), 2)

    # returning the output image
    return orig

#############################################################################################
if __name__ == '__main__':
    if not cv2.useOptimized():
        cv2.setUseOptimized(True)  # setting the optimization to true to decrease the execution time of the code

    # Reading the image
    source = cv2.imread('road_final.jpg')
    cv2.imshow("Source Image", source)

    #  detecting the blue square and saving its coordinates
    sqrCoord = getSqrCoord(source)
    sqWidth = 75  # the real width of the square
    scale = 1  # the scale of the image after modifying the perspective
    scaledWidth = sqWidth * scale  # 1 : 1 scale
    # transforming the image and getting square's new coordinates
    modifiedImage, sqrCoord = calcTransMtrx(source, sqrCoord, 250, 250, scaledWidth)
    cv2.imshow("birdEyeView", modifiedImage)
#############################################################################################
    image = modifiedImage
    # Extracting the lines from the image and creating an image with the lines drawn on it
    lines, line_image = drawHoughLines(image)  # the Hough lines and the new image with lines drawn on it
    line1 = lines[0]
    line2 = lines[1]

    H = 350  # the height to calc and draw the distance at
    distanceBetTwoLines = calcDistancebetTwoLines(line1, line2,H,scale)   # the distance between the 2 lines at height H
    distanceBetTwoLines /= 100.00  # so that the distance is in meters
    dimImage = drawDistanceAtY(image, line1, line2, distanceBetTwoLines, H)  # drawing the distance between the 2 lines at height H
    cv2.imshow("Dimensions", dimImage)
    print('width of the road at height', H, ':', distanceBetTwoLines, "m")
    # printing the Hough lines
    plt.imshow(line_image)
    plt.show()

    cv2.waitKey()
