import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
# %matplotlib inline

from utils import *

cam_mtx, cam_dist = camera_setup()

prev_left_coeffs = None
prev_right_coeffs = None

def image_pipeline(file, filepath=False):
    global prev_left_coeffs
    global prev_right_coeffs

    if filepath == True:
        # Read in image
        raw = cv2.imread(file)
    else:
        raw = file

    # Parameters
    img_shape = raw.shape

    sx_thresh_temp = (40, 100)
    s_thresh_temp = (150, 255)

    # Apply distortion correction to raw image
    img_undist = undistort_image_examples(raw, cam_mtx, cam_dist, test_example=False)

    have_fit = False
    while not have_fit:
        # img_binary, _ = binary_transform_v1(img_undist)
        img_binary = binary_transform_v2(img_undist, sx_thresh=sx_thresh_temp, s_thresh=s_thresh_temp)
        # plt.imshow(img_binary)

        # Previous region-of-interest mask's function is absorbed by the warp
        img_warped, _, persp_trans_inv = warp(img_binary)
        # plt.imshow(img_warped)
        # plt.show()

        # Histogram and get pixels in window
        leftx, lefty, rightx, righty = histogram_pixels(img_warped, horizontal_offset=40)

        # print('leftx: ', leftx, 'rightx: ', rightx)
        if len(leftx) > 1 and len(rightx) > 1:
            have_fit = True
        xgrad_thresh_temp = (xgrad_thresh_temp[0] - 2, xgrad_thresh_temp[1] + 2)
        s_thresh_temp = (s_thresh_temp[0] - 2, s_thresh_temp[1] + 2)

    # Fit a second order polynomial to each fake lane line
    left_fit, left_coeffs = fit_second_order_poly(lefty, leftx, return_coeffs=True)
    right_fit, right_coeffs = fit_second_order_poly(righty, rightx, return_coeffs=True)

    # print("Left coeffs: ", left_coeffs)
    # print("Right coeffs: ", right_coeffs)

    # Plot data

    # plt.plot(left_fit, lefty, color='green', linewidth=3)
    # plt.plot(right_fit, righty, color='green', linewidth=3)
    # plt.imshow(img_warped, cmap="gray")
    # plt.show()

    # Determine curvature of the lane
    # Define y-value where we want radius of curvature
    # maximum y-value chosen, corresponding to the bottom of the image
    y_eval = 500
    left_curve_rad = np.absolute(((1 + (2 * left_coeffs[0] * y_eval + left_coeffs[1]) ** 2) ** 1.5) \
                                / (2 * left_coeffs[0]))
    right_curve_rad = np.absolute(((1 + (2 * right_coeffs[0] * y_eval + right_coeffs[1]) ** 2) ** 1.5) \
                                 / (2 * right_coeffs[0]))
    # print("Left lane curve radius: ", left_curve_rad, "pixels")
    # print("Right lane curve radius: ", right_curve_rad, "pixels")
    curve_rad = (left_curve_rad + right_curve_rad) / 2
    min_curverad = min(left_curve_rad, right_curve_rad)

    # TODO: if plausible, continue.. else don't
    if not plausible_curvature(left_curve_rad, right_curve_rad) or \
            not plausible_continuation_of_traces(left_coeffs, right_coeffs, prev_left_coeffs, prev_right_coeffs):
        if prev_left_coeffs is not None and prev_right_coeffs is not None:
            left_coeffs = prev_left_coeffs
            right_coeffs = prev_right_coeffs

    prev_left_coeffs = left_coeffs
    prev_right_coeffs = right_coeffs

    # find vehicle position wrt center
    centre = center(719, left_coeffs, right_coeffs)

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

    # Warp lane boundaries back onto original image
    lane_lines = cv2.warpPerspective(trace, persp_trans_inv, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR)
    combined_img = plot_poly_on_image(lane_lines, img_undist, curve_rad, centre, min_curverad, left_coeffs, right_coeffs)

    return combined_img

# combined_img = image_pipeline("test_images/test3.jpg", filepath=True)


VIDEOS = ["videos/project_video.mp4", "videos/challenge_video.mp4", "videos/harder_challenge_video.mp4"]
SELECTED_VIDEO = 0

# Import moviepy needed to edit/save video clips
from moviepy.editor import VideoFileClip
# from IPython.display import HTML

clip1 = VideoFileClip(VIDEOS[SELECTED_VIDEO])
project_clip = clip1.fl_image(image_pipeline)

project_output = VIDEOS[SELECTED_VIDEO][:-4] + '_out_v1.mp4'
project_clip.write_videofile(project_output, audio=False)


# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format(output))
