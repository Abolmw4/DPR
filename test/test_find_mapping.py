import cv2
import numpy as np

from utils.find_mapping import create_lighting_filter, apply_lighting_filter
import unittest


class MyTestCase(unittest.TestCase):
    def test_create_lighting_filter(self):
        refrence_image: np.ndarray = cv2.imread("/home/abolfazl/Documents/DPR/obama.jpg")
        target_image: np.ndarray = cv2.imread("/home/abolfazl/Documents/DPR/candidiate_config/candidate1/result/rotate_light_04.jpg")
        smooth_gain, b = create_lighting_filter(img_x=refrence_image, img_y=target_image)

    def test_apply_lighting_filter(self):
        refrence_image: np.ndarray = cv2.imread("/home/abolfazl/Documents/DPR/obama.jpg")
        target_image: np.ndarray = cv2.imread("candidiate_config/candidate1/result/rotate_light_04.jpg")
        smooth_gain, _ = create_lighting_filter(img_x=refrence_image, img_y=target_image)
        result = apply_lighting_filter(refrence_image, smooth_gain)
        filter_vis = cv2.normalize(smooth_gain, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow('Lighting Filter Map', filter_vis)
        cv2.waitKey(0)

        cv2.imwrite("/home/abolfazl/Documents/DPR/res.jpg", result)
        cv2.imshow("result_img", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
