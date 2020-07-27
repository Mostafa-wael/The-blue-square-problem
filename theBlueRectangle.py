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
    edged = cv2.dilate(edged, None,iterations=1)  # it is followed by the erosion process to increase the size of the erroted objects
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
    modifiedImg = cv2.warpPerspective(sourceImg, transfMatrix, sourceImg.shape[:2][::-1])  # appying the transformation matrix  on the source image

    newCoord = np.array(newCoord, dtype= int)
    return modifiedImg,newCoord

def RegionOfInterest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def showCoordOfImg(img):
    # printing out some stats and plotting the image
    # print('This image is:', type(img), 'with dimensions:', img.shape)
    # by moving the cursor over the image we can get the coordinates of any point
    plt.imshow(img)
    plt.show()
    height, width, colorNum = img.shape
    return height, width  # return a tuple of height and width

def cropRegionOfInterest (img):
    # region_of_interest_vertices = [topLeft,bottomLeft ,topRight, bottomRight]
    RegionOfInterestList = [(image.shape[1] / 3, 0), (image.shape[1] / 3, image.shape[0]),
                                   (4 * image.shape[0] / 3, 0), (4 * image.shape[0] / 3, image.shape[0])]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyedImage = cv2.Canny(gray, 200, 300)
    croppedImage = RegionOfInterest(cannyedImage, np.array([RegionOfInterestList], np.int32), )
    return croppedImage

def draw_lines(img, lines,thickness=3):
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
            cv2.line(line_img, (x1-15, y1), (x2-15, y2), [255,0,0], thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img

def drawHoughLines (img):
    # detecting the lines
    lines = cv2.HoughLinesP(
        cropRegionOfInterest(img),
        rho=30,
        theta=np.pi / 20,
        threshold=190,
        lines=np.array([]),
        minLineLength=150,
        maxLineGap=20
    )
    listOfLines = []
    for line in lines:
        listOfPoints = []
        for x1, y1, x2, y2 in line:
            listOfPoints.append(x1)
            listOfPoints.append(y1)
            listOfPoints.append(x2)
            listOfPoints.append(y2)
        listOfLines.append(listOfPoints)
    # drawing the lines
    line_image = draw_lines(image, lines)

    return listOfLines, line_image

def calcDistance (lines, scale, Y):
    # calc the distance between the 2-lines in the x direction at point Y
    line1, line2 = lines
    slope1 = (line1[3] - line1[1]) / (line1[2] - line1[0])
    slope2 = (line2[3] - line2[1]) / (line2[2] - line2[0])
    Xline1 = line1[0] + (line1[3] - line1[1]) / slope1
    Xline2 = line2[0] + (line2[3] - line2[1]) / slope2

    distanceBetMidPoints = abs(Xline1 - Xline2)
    # actual length is: length / scale
    distanceBetMidPoints /= scale
    return distanceBetMidPoints

def drawDistanceMidPoints (img, line1, line2, D):
    # drawing the distcance at the midponts of the 2-lines
    # drawing the contours on the image
    orig = img.copy()
    M1X = int((line1[0]+line1[2])/2)
    M1Y = int((line1[1]+line1[3])/2)
    M2X = int((line2[0] + line2[2]) / 2)
    M2Y = int((line2[1] + line2[3]) / 2)
    # drawing circles corresponding to the current points
    cv2.circle(orig, (M1X,M1Y), 5, (100, 50, 100), -1)
    cv2.circle(orig, (M2X, M2Y), 5, (100, 50, 100), -1)

    # connecting them with a line
    cv2.line(orig, (M1X, M1Y), (M2X, M2Y), (100, 50, 100), 2)

    (mX, mY) = midPoint((M1X,M1Y), (M2X, M2Y))  # writting the text in the mid distance between the two contours
    cv2.putText(orig, "{:.1f}cm".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 50, 100), 2)

    # showing the output image
    cv2.imshow("Dimensions", orig)

def drawDistanceAtY (img, line1, line2, D, Y):
    # drawing the distance at height = Y
    # drawing the contours on the image
    orig = img.copy()
    line1, line2 = lines
    slope1 = (line1[3] - line1[1]) / (line1[2] - line1[0])
    slope2 = (line2[3] - line2[1]) / (line2[2] - line2[0])
    Xline1 = int(line1[0] + (line1[3] - line1[1]) / slope1) -15
    Xline2 = int(line2[0] + (line2[3] - line2[1]) / slope2) -15
    # drawing circles corresponding to the current points
    cv2.circle(orig, (Xline1, Y), 5, (100, 50, 100), -1)
    cv2.circle(orig, (Xline2, Y), 5, (100, 50, 100), -1)

    # connecting them with a line
    cv2.line(orig, (Xline1, Y), (Xline2, Y), (100, 50, 100), 2)

    (mX, mY) = midPoint((Xline1, Y), (Xline2, Y))  # writting the text in the mid distance between the two contours
    cv2.putText(orig, "{:.1f}cm".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 50, 100), 2)

    # showing the output image
    cv2.imshow("Dimensions", orig)

if __name__ == '__main__':
    #############################################################################################
    # Reading the image
    source = cv2.imread('road_final.jpg')
    cv2.imshow("Source Image", source)

    #  detecting the blue sqaure and saving its coordinates
    sqrCoord = getSqrCoord(source)
    sqWidth = 75  # the real width of the square
    scale = 1  # the scale of the image after modifying the perspective
    size = sqWidth * scale  # 1 : 1 scale

    modifiedImage, sqrCoord = calcTransMtrx(source, sqrCoord, 250, 250, size)
    cv2.imshow("birdEyeView", modifiedImage)
#############################################################################################
    image = modifiedImage

    lines, line_image = drawHoughLines(image)  # the Hough lines and the new image with lines drawn on it

    H = 350
    distanceBetMidLines = calcDistance(lines, scale, H)   # the distance between the 2 lines at height H
    print('width of the road', distanceBetMidLines, "cm")

    drawDistanceAtY(image, lines[0], lines[1], distanceBetMidLines, H) # drawing the distance between the 2 lines at height H

    # printing the Hough lines
    plt.figure()
    plt.imshow(line_image)
    plt.show()




    cv2.waitKey()

