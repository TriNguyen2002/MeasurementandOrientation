from blend_modes import divide
import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import numpy as np


def de_shadow(image):
    # splitting the image into channels
    bA = image[:, :, 0]
    gA = image[:, :, 1]
    rA = image[:, :, 2]

    # dialting the image channels individually to spead the text to the background
    dilated_image_bB = cv2.dilate(bA, np.ones((7, 7), np.uint8))
    dilated_image_gB = cv2.dilate(gA, np.ones((7, 7), np.uint8))
    dilated_image_rB = cv2.dilate(rA, np.ones((7, 7), np.uint8))

    # blurring the image to get the backround image
    bB = cv2.medianBlur(dilated_image_bB, 21)
    gB = cv2.medianBlur(dilated_image_gB, 21)
    rB = cv2.medianBlur(dilated_image_rB, 21)

    # blend_modes library works with 4 channels, the last channel being the alpha channel
    # so we add one alpha channel to our image and the background image each
    image = np.dstack((image, np.ones((image.shape[0], image.shape[1], 1)) * 255))
    image = image.astype(float)
    dilate = [bB, gB, rB]
    dilate = cv2.merge(dilate)
    dilate = np.dstack((dilate, np.ones((image.shape[0], image.shape[1], 1)) * 255))
    dilate = dilate.astype(float)

    # now we divide the image with the background image
    # without rescaling i.e scaling factor = 1.0
    blend = divide(image, dilate, 1.0)
    blendb = blend[:, :, 0]
    blendg = blend[:, :, 1]
    blendr = blend[:, :, 2]
    blend_planes = [blendb, blendg, blendr]
    blend = cv2.merge(blend_planes)
    # blend = blend*0.85
    blend = np.uint8(blend)

    # returning the shadow-free image
    return blend


def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
    return biggest, max_area

# quan trong nhat
def get_contours(
    bgr_image,
    canny_threshold=(100, 200),
    min_area=500,
    shadow_free=False,
    show_canny=False,
    draw_cont=False,
):
    # preprocess img
    if shadow_free:
        no_shadow_img = de_shadow(bgr_image)
        gray_img = cv2.cvtColor(no_shadow_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
    # find edges and more preprocess
    canny_img = cv2.Canny(blur_img, canny_threshold[0], canny_threshold[1])
    kernel = np.ones((5, 5))
    dilate_img = cv2.dilate(canny_img, kernel, iterations=3)
    erode_img = cv2.erode(dilate_img, kernel, iterations=2)
    if show_canny:
        cv2.imshow(
            "Cany edge dectection",
            cv2.resize(erode_img, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA),
        )
    # this function returns 3 output: modified img, contours, hierarchy
    contours, _ = cv2.findContours(
        erode_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    final_contours = []
    for cont in contours:
        # area is a float
        area = cv2.contourArea(cont)
        if area > min_area:
            peri = cv2.arcLength(cont, closed=True)
            # approx_corners has 4 elements for 4 corners of the rect
            approx_corners = cv2.approxPolyDP(cont, 0.02 * peri, closed=True)            
            rotated_rect = cv2.minAreaRect(cont)
            # bbox has 4 elements for 4 corners
            bbox = np.int0(cv2.boxPoints(rotated_rect))
            # when found 4 corners, write down data in the list

            final_contours.append((area, approx_corners, bbox, cont))
    # sorting the highest val first            
    final_contours = sorted(final_contours, key=lambda x: x[0], reverse=True)
    if draw_cont:
        for cont in final_contours:
            cv2.drawContours(bgr_image, cont[1], -1, (0, 0, 255), 7)
    return final_contours


def reorder_corner_points(points):
    new_points = np.zeros_like(points)
    # change the size when you apply it
    reshape_points = points.reshape((4, 2))  # reshape into 2-D array, gay ra loi khi co vat the lon
    # Find the upper left corner and the lower right corner
    add = np.sum(reshape_points, axis=1)
    new_points[0] = reshape_points[np.argmin(add)]
    new_points[3] = reshape_points[np.argmax(add)]
    # Find the upper right corner and the lower left corner
    diff = np.diff(reshape_points, axis=1)
    new_points[1] = reshape_points[np.argmin(diff)]
    new_points[2] = reshape_points[np.argmax(diff)]
    return new_points


def warp_image(img, corner_points, width, height, pad=10):
    points = reorder_corner_points(corner_points)
    # Find perspective transform matrix
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # warping perspective
    warp_img = cv2.warpPerspective(img, matrix, (width, height))
    warp_img = warp_img[pad : warp_img.shape[0] - pad, pad : warp_img.shape[1] - pad]
    return warp_img

# fix it for the tube
def calculate_distance(pts1, pst2):
    return ((pst2[0] - pts1[0]) ** 2 + (pst2[1] - pts1[1]) ** 2) ** 0.5

def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)
  ## [visualization1]
 

def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv2.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 2)
  drawAxis(img, cntr, p2, (0, 0, 255), 4)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  label = str(int(np.rad2deg(angle)) + 90) + ' degree'
#   textbox = cv2.rectangle(img, (cntr[0]-20, cntr[1]+30), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  cv2.putText(img, 
              label, 
              (cntr[0]-15, cntr[1]+25), 
              cv2.FONT_HERSHEY_SIMPLEX, 
              0.55, 
              (0,0,255), 
              1, 
              )
 
  return int(np.rad2deg(angle)) + 90