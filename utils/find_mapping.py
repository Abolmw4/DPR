import cv2
import numpy as np


def create_lighting_filter(img_x: np.ndarray, img_y: np.ndarray, epsilon: float=1e-5):
    lab_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2LAB).astype(np.float32)

    lx = lab_x[:, :, 0]
    ly = lab_y[:, :, 0]


    gain_map = ly / (lx + epsilon)

    # 3. کلیپ کردن برای جلوگیری از مقادیر غیرعادی
    gain_map = np.clip(gain_map, 0.1, 10.0)

    # 4. صاف‌سازی فیلتر (Smoothing)
    # استفاده از GaussianBlur یا بهتر از آن GuidedFilter
    # GuidedFilter باعث می‌شود فیلتر نوری دقیقاً با فرم صورت منطبق بماند

    # پارامترها را بسته به رزولوشن تصویر تنظیم کنید
    # برای یک تصویر 1080p، شعاع 50 تا 100 مناسب است
    radius = 100
    eps = 0.1 * (255 ** 2)  # پارامتر نرمی برای Guided Filter

    smooth_gain = cv2.ximgproc.guidedFilter(guide=lx.astype(np.uint8),
                                            src=gain_map,
                                            radius=radius,
                                            eps=eps)

    return smooth_gain, lab_x


def apply_lighting_filter(target_img: np.ndarray, gain_filter: np.ndarray):
    # 1. تبدیل تصویر مقصد به LAB
    lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 2. اعمال فیلتر روی کانال L
    lab[:, :, 0] = lab[:, :, 0] * gain_filter

    # 3. محدود کردن مقادیر روشنایی به بازه استاندارد LAB (0 تا 100)
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)

    # 4. تبدیل مجدد به BGR
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result
