import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_corners(image: np.ndarray, coordinates: np.ndarray, margin=10, color='k', marker_size=200, marker='+', style='both') -> None:
    """
    Visualize corners on the given image using matplotlib, plotting only dots.
    
    Args:
    - image: NumPy array of shape (H, W, C) representing the image
    - coordinates: NumPy array of shape (N, 2) where N is the number of corners,
                   each row contains [x, y] coordinates
    - color: Color of the corner markers (default: 'r' for red)
    - marker_size: Size of the corner markers
    - marker: Marker style (default: 'o' for circle)
    - style: Style of the figures of position locator (default: 'both' for complet view and four corners, 'complete' for complete view, 'corners' for four corners)
    
    Returns:
    - matplotlib Figure object
    """
    if style == 'both' or style == 'complete':
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c=color, s=marker_size, marker=marker)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()
    if style == 'both' or style == 'corners':
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(4):
            c = coordinates[i]
            axes[i].imshow(image[c[1]-margin:c[1]+margin, c[0]-margin:c[0]+margin], cmap='viridis')
            axes[i].scatter(margin, margin, color='k', marker='+', s=marker_size)
        plt.tight_layout()
        plt.show()



def transform_image(image, frame_view, frame_view_ref):
    transformation_matrix = cv2.getPerspectiveTransform(frame_view.astype(np.float32), frame_view_ref.astype(np.float32))
    transformed_image = cv2.warpPerspective(image, transformation_matrix, (image.shape[1], image.shape[0]))
    return transformed_image

def align_plumes(plumes, frame_view, frame_view_ref):
    align_plumes = np.zeros(plumes.shape, dtype=plumes.dtype)
    n_plume, n_frame, h, w = plumes.shape
    for n1 in range(n_plume):
        for n2 in range(n_frame):
            align_plumes[n1, n2] = transform_image(plumes[n1, n2], frame_view, frame_view_ref)
    return align_plumes