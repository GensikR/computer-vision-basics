import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.feature import SIFT, match_descriptors
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import ProjectiveTransform, SimilarityTransform, warp

def compute_affine_transform(src_points, dst_points):
    src_homogeneous = np.hstack((src_points, np.ones((len(src_points), 1))))
    dst_homogeneous = np.hstack((dst_points, np.ones((len(dst_points), 1))))

    A = np.zeros((2 * len(src_points), 6))
    b = np.zeros((2 * len(src_points), 1))

    for i in range(len(src_points)):
        A[2*i, :3] = src_homogeneous[i]
        A[2*i+1, 3:] = src_homogeneous[i]
        
        b[2*i:2*i+2] = dst_homogeneous[i][:2, np.newaxis]

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    affine_matrix = np.vstack((x.reshape(2, 3), [0, 0, 1]))

    return affine_matrix

def compute_projective_transform(src_points, dst_points):
    src_homogeneous = np.hstack((src_points, np.ones((len(src_points), 1))))
    dst_homogeneous = np.hstack((dst_points, np.ones((len(dst_points), 1))))

    A = np.zeros((2 * len(src_points), 8))
    b = np.zeros((2 * len(src_points), 1))

    for i in range(len(src_points)):
        src_h = src_homogeneous[i]
        dst_h = dst_homogeneous[i]

        A[2 * i, 0:3] = src_h
        A[2 * i, 6:8] = -dst_h[0] * src_h[:2]
        A[2 * i + 1, 3:6] = src_h
        A[2 * i + 1, 6:8] = -dst_h[1] * src_h[:2]
        
        b[2 * i] = dst_h[0]
        b[2 * i + 1] = dst_h[1]

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    x = x.flatten()

    projective_matrix = np.zeros((3, 3))
    projective_matrix[0, :] = [x[0], x[1], x[2]]
    projective_matrix[1, :] = [x[3], x[4], x[5]]
    projective_matrix[2, :] = [x[6], x[7], 1]

    return projective_matrix

def compute_stitched_shape(dst_img, dst_img_rgb, src_img_rgb, best_model_affine, best_model_projective):
    # Transform the corners of img1 by the inverse of the best fit models
    rows, cols = dst_img.shape
    corners = np.array([
        [0, 0],
        [cols, 0],
        [0, rows],
        [cols, rows]
    ])

    # Apply the transformations to the corners
    corners_homogeneous = np.hstack((corners, np.ones((len(corners), 1))))
    corners_affine_proj = np.dot(best_model_affine, corners_homogeneous.T).T
    corners_affine_proj /= corners_affine_proj[:, 2, None]  # Normalize homogeneous coordinates

    corners_homogeneous_proj = np.dot(best_model_projective, corners_homogeneous.T).T
    corners_homogeneous_proj /= corners_homogeneous_proj[:, 2, None]  # Normalize homogeneous coordinates

    # Find the bounding boxes of the transformed corners
    corner_min_affine_proj = np.min(corners_affine_proj[:, :2], axis=0)
    corner_max_affine_proj = np.max(corners_affine_proj[:, :2], axis=0)
    output_shape_affine_proj = np.ceil(corner_max_affine_proj - corner_min_affine_proj).astype(int)

    corner_min_proj = np.min(corners_homogeneous_proj[:, :2], axis=0)
    corner_max_proj = np.max(corners_homogeneous_proj[:, :2], axis=0)
    output_shape_proj = np.ceil(corner_max_proj - corner_min_proj).astype(int)

    # Compute the offsets for warping
    offset = SimilarityTransform(translation=-corner_min_affine_proj)

    if best_model_affine.shape == (3, 3):
        transform_affine = SimilarityTransform(matrix=best_model_affine)
    else:
        transform_affine = SimilarityTransform(matrix=best_model_affine)

    if best_model_projective.shape == (3, 3):
        transform_projective = ProjectiveTransform(matrix=best_model_projective)
    else:
        transform_projective = ProjectiveTransform(matrix=best_model_projective)

    # Warp the destination image for affine transformation
    dst_warped_affine = warp(dst_img_rgb, offset.inverse, output_shape=output_shape_affine_proj)

    # Warp the source image for affine transformation
    tf_img_affine = warp(src_img_rgb, (transform_affine + offset).inverse, output_shape=output_shape_affine_proj)

    # Warp the destination image for projective transformation
    dst_warped_proj = warp(dst_img_rgb, offset.inverse, output_shape=output_shape_proj)

    # Warp the source image for projective transformation
    tf_img_proj = warp(src_img_rgb, (transform_projective + offset).inverse, output_shape=output_shape_proj)

    # Combine the images
    dst_warped_affine[tf_img_affine > 0] = tf_img_affine[tf_img_affine > 0]
    dst_warped_proj[tf_img_proj > 0] = tf_img_proj[tf_img_proj > 0]

    # Plot the results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))


    ax1.imshow(dst_img_rgb)
    ax1.set_title('Destination Image', color='red')

    ax2.imshow(src_img_rgb)
    ax2.set_title('Source Image', color='red')

    ax3.imshow(dst_warped_affine)
    ax3.set_title('Affine Transformation', color='red')

    ax4.imshow(dst_warped_proj)
    ax4.set_title('Projective Transformation', color='red')

    plt.tight_layout()
    plt.show()

def ransac(src_keypoints, dst_keypoints, model_func, n_iterations, n_samples, threshold):
    best_model = None
    best_inlier_count = 0
    best_inliers_mask = None

    for _ in range(n_iterations):
        sample_indices = np.random.choice(len(src_keypoints), n_samples, replace=False)
        maybe_inliers_src = src_keypoints[sample_indices]
        maybe_inliers_dst = dst_keypoints[sample_indices]

        maybe_model = model_func(maybe_inliers_src, maybe_inliers_dst)

        inliers_mask = []

        for j in range(len(src_keypoints)):
            src_point = src_keypoints[j]
            dst_point = dst_keypoints[j]

            # Project src_point using maybe_model
            projected_point = np.dot(maybe_model, np.append(src_point, 1))
            projected_point /= projected_point[2]

            # Calculate error as distance between projected point and dst_point
            error = np.linalg.norm(dst_point - projected_point[:2])

            # Determine if it's an inlier
            if error < threshold:
                inliers_mask.append(True)
            else:
                inliers_mask.append(False)

        inlier_count = np.count_nonzero(inliers_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = maybe_model
            best_inliers_mask = np.array(inliers_mask)

    # Extract inlier keypoints
    src_best = src_keypoints[best_inliers_mask]
    dst_best = dst_keypoints[best_inliers_mask]

    return best_model, src_best, dst_best

def main():
    dst_img_rgb = np.asarray(Image.open('yosemite1.jpg'))
    src_img_rgb = np.asarray(Image.open('yosemite2.jpg'))

    if dst_img_rgb.shape[2] == 4:
        dst_img_rgb = rgba2rgb(dst_img_rgb)
    if src_img_rgb.shape[2] == 4:
        src_img_rgb = rgba2rgb(src_img_rgb)

    dst_img = rgb2gray(dst_img_rgb)
    src_img = rgb2gray(src_img_rgb)

    detector1 = SIFT()
    detector2 = SIFT()
    detector1.detect_and_extract(dst_img)
    detector2.detect_and_extract(src_img)
    keypoints1 = detector1.keypoints
    descriptors1 = detector1.descriptors
    keypoints2 = detector2.keypoints
    descriptors2 = detector2.descriptors

    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    dst = keypoints1[matches[:, 0]]
    src = keypoints2[matches[:, 1]]

    aff_model, aff_src, aff_dst = ransac(src, dst, compute_affine_transform, n_iterations=300, n_samples=4, threshold=1)
    proj_model, proj_src, proj_dst = ransac(src, dst, compute_projective_transform, n_iterations=300, n_samples=4, threshold=1)

    compute_stitched_shape(dst_img, dst_img_rgb, src_img_rgb, aff_model, proj_model)

if __name__ == '__main__':
    main()
