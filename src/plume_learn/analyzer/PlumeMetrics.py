import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from viz import create_axes_grid
from analyzer.HorizontalLineProfileAnalyzer import HorizontalLineProfileAnalyzer

class PlumeMetrics:

    def __init__(self, time_interval, start_position, position_range, threshold=200, progress_bar=True):
        '''
        This is a class used to calculate the velocity of the plume based on its positions in consecutive frames.

        :param time_interval: time interval
        :type time_interval: int

        :param position_range: position range for x axis/horizontal axis
        :type position_range: tuple

        :param start_position: start position
        :type start_position: tuple
        '''
        self.time_interval = time_interval
        self.start_position = start_position
        self.position_range = position_range
        self.threshold = threshold
        self.progress_bar = progress_bar

    def calculate_area(self, frame, viz=False):

        # determine the threshold based on the line profile along the horizontal axis
        y = self.start_position[1]
        x_start = self.position_range[0]

        # method 1: use the threshold provided by the user
        if self.threshold == 'flexible':
            # method 2: use the horizontal line profile to detect the threshold
            analyzer = HorizontalLineProfileAnalyzer(frame, row=y, line_width=5)
            profile = analyzer.extract_profile()
            position, threshold = analyzer.detect(target_x=x_start, show_image=False, show_profile=False, show_difference=False)
            if threshold == None:
                threshold = 200 # use the default threshold(200)

        elif isinstance(self.threshold, int):
            threshold = self.threshold

        else:
            raise ValueError('Please provide the threshold value or use the flexible method to detect the threshold.')


        _, frame_binary = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)

        # calculate the area
        num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(frame_binary)

        if num_labels < 2:
            return 0, (0,0), np.zeros(frame.shape, dtype=np.uint8)
        
        index_areas = np.stack([np.arange(num_labels), stats[:,-1]])
        
        # Sort components based on area, ignoring the first component (background)
        sorted_indices = np.argsort(stats[:, -1])[::-1]  # Sort in descending order

        # Exclude the background component
        sum_area_values = [np.mean(frame[labeled_image==sorted_indices[index]]) for index in range(2)]
        background_index = np.argmin(sum_area_values)
        largest_blob_index = np.argmax(sum_area_values)

        if viz:
            self.label_blob(frame, frame_binary, labeled_image, stats, centroids, sorted_indices, background_index, n_show=5)
        
        area = stats[sorted_indices[largest_blob_index], -1]
        coord = (centroids[sorted_indices[largest_blob_index]]).astype(np.uint8)
        return area, coord, labeled_image


    def calculate_area_for_plume(self, plume):
        areas = np.zeros(plume.shape[0])
        coords = np.zeros((plume.shape[0],2), dtype=np.uint8)
        labeled_images = np.zeros(plume.shape)

        for j in range(plume.shape[0]):
            areas[j], coords[j], labeled_images[j] = self.calculate_area(plume[j])
        return areas, coords, labeled_images


    def calculate_area_for_plumes(self, plumes, return_format='numpy'):
        areas = np.zeros(plumes.shape[:2])
        coords = np.zeros((*plumes.shape[:2],2))
        labeled_images = np.zeros(plumes.shape)

        if self.progress_bar:
            for i in tqdm(range(plumes.shape[0])):
                areas[i], coords[i], labeled_images[i] = self.calculate_area_for_plume(plumes[i])
        else:
            for i in range(plumes.shape[0]):
                areas[i], coords[i], labeled_images[i] = self.calculate_area_for_plume(plumes[i])
                
        return areas, coords, labeled_images
    
    def to_df(self, areas):
        num_plumes, num_times = areas.shape[:2]
        plume_indices = np.repeat(np.arange(num_plumes), num_times)
        time_indices = np.tile(np.arange(num_times), num_plumes)
        multi_index = pd.MultiIndex.from_arrays([plume_indices, time_indices], names=['plume_index', 'time_index'])

        df = pd.DataFrame({'Area': areas.flatten() 
                            }, index=multi_index)
        return df
        

    @staticmethod
    def viz_area(df, x_range):
        if not isinstance(df, pd.DataFrame):
            raise ValueError('Input should be a pandas DataFrame.')

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(x="time_index", y="area", data=df)
        plt.xlim(*x_range)
        plt.show()


    @staticmethod
    def viz_blob_plume(plume, areas, coords, labeled_images, title=None):

        fig, axes = create_axes_grid(len(plume), n_per_row=8, plot_height=1.1)
        axes = axes.flatten()
        for i in range(len(plume)):
            axes[i].imshow(labeled_images[i], cmap='viridis')
            if np.sum(coords[i]) != 0:
                axes[i].text(coords[i][0], coords[i][1], f'Area:{areas[i]}', color='white', fontsize=8, ha='center', va='center')
            axes[i].axis('off')

        if title:
            plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def label_blob(image, image_binary, labels, stats, centroids, sorted_indices, background_index, n_show=5):

        # Create a color map for visualization
        colors = plt.cm.jet(np.linspace(0, 1, n_show + 1))

        # Create an RGB image for visualization
        image_colored = np.zeros((*image_binary.shape, 3), dtype=np.uint8)
        
        for i, label in enumerate(sorted_indices[:n_show]):
            if label == background_index:
                continue
            mask = labels == label
            image_colored[mask] = colors[i + 1][:3] * 255  # Convert color from [0,1] to [0,255]

        fig, axes = plt.subplots(1, 3, figsize=(6, 2))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'Original image')

        axes[1].imshow(image_binary, cmap='gray')
        axes[1].set_title('Binary image')

        axes[2].imshow(image_colored)
        axes[2].set_title('Colored blobs')
        for i, label in enumerate(sorted_indices[:n_show]):
            if label == background_index:
                continue
            plt.text(centroids[label][0], centroids[label][1], f'{i}th area:{stats[label, -1]}',
                    color='white', fontsize=8, ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.show()