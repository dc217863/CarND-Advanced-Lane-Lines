import cv2
import glob
import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

from utils import *


## Camera calibration and warping of image

cam_mtx, cam_dist = camera_setup()

img_test = mpimage.imread('test_images/test3.jpg')
img_st = mpimage.imread('test_images/straight_lines2.jpg')

img_undist = undistort_image_examples(img_test, cam_mtx, cam_dist, test_example=True)
img_st_undist = undistort_image_examples(img_st, cam_mtx, cam_dist, test_example=True)

sx_thresh_temp = (40, 100)
s_thresh_temp = (150, 255)

img_binary = binary_transform_v2(img_undist, sx_thresh=sx_thresh_temp, s_thresh=s_thresh_temp)
img_st_binary = binary_transform_v2(img_st_undist, sx_thresh=sx_thresh_temp, s_thresh=s_thresh_temp)

img_shape = img_binary.shape
img_st_shape = img_st_binary.shape

vertices = np.array([[(0, img_shape[0]), (550, 470), (700, 470), (img_shape[1], img_shape[0])]], dtype=np.int32)
img_roi, _ = region_of_interest(img_binary, vertices)
img_st_roi, _ = region_of_interest(img_st_binary, vertices)

plot_roi(img_undist, img_binary, img_roi)

# Previous region-of-interest mask's function is absorbed by the warp
img_warped, _, persp_trans_inv = warp(img_binary)
# img_warped, _, persp_trans_inv = warp(img_st_roi)
plot_warped(img_undist, img_warped)

## Fit polynomial to lane lines

# Histogram and get pixels in window
horizontal_offset = 40
leftx, lefty, rightx, righty = histogram_pixels(img_warped, horizontal_offset=horizontal_offset)

# Fit a second order polynomial to each fake lane line
left_fit, left_coeffs = fit_second_order_poly(lefty, leftx, return_coeffs=True)
right_fit, right_coeffs = fit_second_order_poly(righty, rightx, return_coeffs=True)

print("Left coeffs: ", left_coeffs)
print("Right coeffs: ", right_coeffs)
blank_canvas = np.zeros_like(img_binary)
polyfit_left = draw_poly(blank_canvas, lane_poly, left_coeffs, 30)
polyfit_drawn = draw_poly(polyfit_left, lane_poly, right_coeffs, 30)

colour_canvas = cv2.cvtColor(blank_canvas.astype(np.uint8), cv2.COLOR_GRAY2RGB)
trace = colour_canvas
trace[polyfit_drawn > 1] = [0, 0, 255]
area = highlight_lane_line_area(blank_canvas, left_coeffs, right_coeffs)
trace[area == 1] = [0, 255, 0]

plot_polynomial(img_warped, left_fit, lefty, right_fit, righty,
                polyfit_drawn,
                trace)


## Determine curvature of the lane and vehicle position with respect to center

# Determine curvature of the lane
# Define y-value where we want radius of curvature
# maximum y-value chosen, corresponding to the bottom of the image
y_eval = 500
left_curve_rad = np.absolute(((1 + (2 * left_coeffs[0] * y_eval + left_coeffs[1])**2) ** 1.5) \
                /(2 * left_coeffs[0]))
right_curve_rad = np.absolute(((1 + (2 * right_coeffs[0] * y_eval + right_coeffs[1]) ** 2) ** 1.5) \
                 /(2 * right_coeffs[0]))
print("Left lane curve radius: ", left_curve_rad, "pixels")
print("Right lane curve radius: ", right_curve_rad, "pixels")
curve_rad = (left_curve_rad + right_curve_rad) / 2
centre = center(719, left_coeffs, right_coeffs)
min_curvature = min(left_curve_rad, right_curve_rad)


## Warp the detected lane boundaries back onto the original image

# Warp lane boundaries back onto original image
lane_lines = cv2.warpPerspective(trace, persp_trans_inv, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR)
combined_img = plot_poly_on_image(lane_lines, img_undist, curve_rad, centre, min_curvature, left_coeffs, right_coeffs)

plt.show()
# ----------------------------------------------------------------




