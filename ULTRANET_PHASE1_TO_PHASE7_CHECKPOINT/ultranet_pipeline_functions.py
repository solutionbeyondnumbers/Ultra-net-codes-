

# ============================================================
# ULTRA-Net Core Utility Functions
# ============================================================

import cv2
import pywt
import numpy as np

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_ultrasound_image(image):

    denoised = cv2.medianBlur(
        image,
        3
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        denoised
    )

    gamma = 1.2

    normalized = enhanced / 255.0

    gamma_corrected = np.power(
        normalized,
        gamma
    )

    gamma_corrected = np.uint8(
        gamma_corrected * 255
    )

    final_image = cv2.normalize(
        gamma_corrected,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return final_image

# ============================================================
# ROI EXTRACTION
# ============================================================

def extract_roi(
    image,
    mask,
    padding=20,
    output_size=(256, 256)
):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        resized_image = cv2.resize(
            image,
            output_size
        )

        resized_mask = cv2.resize(
            mask,
            output_size,
            interpolation=cv2.INTER_NEAREST
        )

        return resized_image, resized_mask

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)

    x2 = min(
        x + w + padding,
        image.shape[1]
    )

    y2 = min(
        y + h + padding,
        image.shape[0]
    )

    roi_image = image[
        y1:y2,
        x1:x2
    ]

    roi_mask = mask[
        y1:y2,
        x1:x2
    ]

    roi_image = cv2.resize(
        roi_image,
        output_size
    )

    roi_mask = cv2.resize(
        roi_mask,
        output_size,
        interpolation=cv2.INTER_NEAREST
    )

    return roi_image, roi_mask

# ============================================================
# ENERGY
# ============================================================

def compute_energy(arr):

    return np.sum(
        np.square(arr.astype(np.float64))
    )

# ============================================================
# GLCM FEATURES
# ============================================================

def extract_glcm_features(image):

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = {

        "glcm_contrast":
        graycoprops(glcm, "contrast")[0, 0],

        "glcm_correlation":
        graycoprops(glcm, "correlation")[0, 0],

        "glcm_energy":
        graycoprops(glcm, "energy")[0, 0],

        "glcm_homogeneity":
        graycoprops(glcm, "homogeneity")[0, 0]

    }

    return features

# ============================================================
# LBP FEATURES
# ============================================================

def extract_lbp_features(
    image,
    radius=3,
    n_points=24
):

    lbp = local_binary_pattern(
        image,
        n_points,
        radius,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype("float32")

    hist /= (
        hist.sum() + 1e-8
    )

    return hist

# ============================================================
# WAVELET FEATURES
# ============================================================

def extract_wavelet_features(image):

    coeffs = pywt.wavedec2(
        image,
        wavelet="haar",
        level=2
    )

    features = []

    for coeff in coeffs:

        if isinstance(coeff, tuple):

            for arr in coeff:

                features.extend([

                    np.mean(arr),
                    np.std(arr),
                    compute_energy(arr)

                ])

        else:

            features.extend([

                np.mean(coeff),
                np.std(coeff),
                compute_energy(coeff)

            ])

    return np.array(
        features,
        dtype=np.float32
    )

# ============================================================
# COMPLETE TEXTURE FEATURES
# ============================================================

def extract_texture_features(image):

    glcm_features = extract_glcm_features(
        image
    )

    lbp_features = extract_lbp_features(
        image
    )

    wavelet_features = extract_wavelet_features(
        image
    )

    feature_vector = []

    feature_vector.extend(
        list(glcm_features.values())
    )

    feature_vector.extend(
        lbp_features.tolist()
    )

    feature_vector.extend(
        wavelet_features.tolist()
    )

    return np.array(
        feature_vector,
        dtype=np.float32
    )

