import cv2
import numpy as np

# Step 1: Undistort Images
def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted_image

# Step 2: Compute Homography
def compute_homography(src_points, dst_points):
    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return homography

# Step 3: Warp Images
def warp_image(image, homography, target_size):
    warped_image = cv2.warpPerspective(image, homography, target_size)
    return warped_image

# Step 4: Blend Images (Optional)
def blend_images(image1, image2):
    blended_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
    return blended_image

# Load camera intrinsics from cameras.txt
def load_camera_intrinsics(filename):
    intrinsics = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            data = line.split()
            camera_id = int(data[0])
            fx, fy, cx, cy, k1, k2 = map(float, data[4:])
            intrinsics[camera_id] = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'dist_coeffs': (k1, k2, 0, 0, 0)  # Distortion coefficients k3, p1, p2 are assumed to be 0
            }
    return intrinsics

# Load camera extrinsics from images.txt
def load_camera_extrinsics(filename):
    extrinsics = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            data = lines[i].split()
            image_id = int(data[0])
            qw, qx, qy, qz, tx, ty, tz, camera_id, name = map(float, data[1:])
            extrinsics[image_id] = {
                'rotation': np.array([qw, qx, qy, qz]),
                'translation': np.array([tx, ty, tz]),
                'camera_id': int(camera_id),
                'name': name
            }
    return extrinsics

# Example usage
if __name__ == "__main__":
    # Load camera intrinsics
    camera_intrinsics = load_camera_intrinsics('cameras.txt')

    # Load camera extrinsics
    camera_extrinsics = load_camera_extrinsics('images.txt')

    # Load images
    image1 = cv2.imread('image1.jpg')
    image2 = cv2.imread('image2.jpg')

    # Example image IDs
    image1_id = 1
    image2_id = 2

    # Step 1: Undistort Images
    undistorted_image1 = undistort_image(image1, camera_intrinsics[camera_extrinsics[image1_id]['camera_id']]['camera_matrix'], camera_intrinsics[camera_extrinsics[image1_id]['camera_id']]['dist_coeffs'])
    undistorted_image2 = undistort_image(image2, camera_intrinsics[camera_extrinsics[image2_id]['camera_id']]['camera_matrix'], camera_intrinsics[camera_extrinsics[image2_id]['camera_id']]['dist_coeffs'])

    # Step 2: Compute Homography
    # You'll need to provide corresponding points (src_points and dst_points) for each camera pair
    # Implement manual point selection / use cv2 mcc to find ColorChecker board

    # Step 3: Warp Images
    # You'll need to compute the homography for each camera pair and warp the images accordingly

    # Step 4: Blend Images (Optional)
    # You'll need to blend the warped images together if they overlap

    # Display results
    cv2.imshow('Undistorted Image 1', undistorted_image1)
    cv2.imshow('Undistorted Image 2', undistorted_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
