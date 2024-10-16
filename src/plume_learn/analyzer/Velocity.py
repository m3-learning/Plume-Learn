import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage import measure
from skimage.measure import regionprops, regionprops_table

from viz import create_axes_grid
from analyzer.HorizontalLineProfileAnalyzer import HorizontalLineProfileAnalyzer

class VelocityCalculator:

    def __init__(self, time_interval, start_position, position_range, threshold=200, progress_bar=True):
        '''
        This is a class used to calculate the velocity of the plume based on its positions in consecutive frames.

        :param time_interval: time interval
        :type time_interval: int

        :param position_range: position range 
        :type position_range: tuple

        :param start_position: start position
        :type start_position: tuple

        :param threshold: threshold
        :type threshold: int

        '''

        self.time_interval = time_interval
        self.position_range = position_range
        self.start_position = start_position
        self.threshold = threshold
        self.progress_bar = progress_bar
    
    def to_df(self, plume_positions, plume_distances, plume_velocities):
        num_plumes, num_times = plume_positions.shape[:2]
        plume_indices = np.repeat(np.arange(num_plumes), num_times)
        time_indices = np.tile(np.arange(num_times), num_plumes)
        multi_index = pd.MultiIndex.from_arrays([plume_indices, time_indices], names=['plume_index', 'time_index'])

        df = pd.DataFrame({'Distance': plume_distances.flatten(),
                           'Velocity': plume_velocities.flatten()
                           }, index=multi_index)
        return df

    def calculate_distance_area_for_plumes(self, plumes, return_format='numpy'):
        '''
        This is a function used to calculate the velocity of the plume based on its positions in consecutive frames.

        :param plumes: plume images
        :type plumes: numpy.ndarray
        '''

        # Calculate the velocities and distances for each video
        if self.progress_bar:
            results = [self.calculate_velocity_and_distance_for_plume(plume) for plume in tqdm(plumes)]
        else:
            results = [self.calculate_velocity_and_distance_for_plume(plume) for plume in plumes]

        # Separate the velocities and distances into two lists
        plume_positions = np.array([result[0] for result in results])
        plume_distances = np.array([result[1] for result in results])
        plume_velocities = np.array([result[2] for result in results])

        plume_positions = np.array(plume_positions)
        plume_velocities = np.array(plume_velocities)
        plume_distances = np.array(plume_distances)

        return plume_positions, plume_distances, plume_velocities


    def calculate_velocity_and_distance_for_plume(self, plume):
        """
        Calculates the velocity of the plume based on its positions in consecutive frames.
        we only consider the front end (x axis) of the plume since we already normalize the image

        Args:
            plume_positions (list): A list of tuples (x, y) representing the centroid positions of the plume.
            
        Returns:
            list: A list of velocities in pixels per unit time.
        """
        positions = []
        velocities = []
        distances = []

        previous_x = 0
        # set the start position
        if isinstance(self.start_position, tuple):
            previous_x = self.start_position[0]

        for frame in plume:
            x, y = self.get_plume_position(frame, self.threshold)
            positions.append((x, y))

            if distances != []: # not calculate the backward
                # print(len(positions), x, distances[-1])
                if x - previous_x < distances[-1]:
                    x = distances[-1] + previous_x

                # print(len(positions), (x,y), self.start_position, x, distances[-1])

            # print(x, previous_x)
            distances.append(x - previous_x)

        velocities = [(distances[i]-distances[i-1]) / self.time_interval for i in range(1, len(distances))]
        velocities = [0] + velocities # add the first velocity as 0
        return np.array(positions), np.array(distances), np.array(velocities)
    

    def get_plume_position(self, frame, threshold):
        y = self.start_position[1]
        x_start = self.position_range[0]
        # print(x_start, y)

        if threshold == 'flexible': # use the threshold detected by the line profile

            analyzer = HorizontalLineProfileAnalyzer(frame, row=y, line_width=5)
            profile = analyzer.extract_profile()
            position, magnitude = analyzer.detect(target_x=x_start, show_image=False, show_profile=False, show_difference=False)
            if position == None:
                position = x_start
            # print(position, magnitude)
            return position, y
        
        elif isinstance(threshold, int): # use the threshold provided by the user
            # print(frame.shape)
            _, frame_binary = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
            # frame_binary = np.copy(frame)

            # calculate the front end of the plume
            label_img = measure.label(frame_binary)
            regions = regionprops(label_img)
            sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
            if len(sorted_regions) == 0:
                return 0, 0
            minr, minc, maxr, maxc = sorted_regions[0].bbox 
            if maxc < self.position_range[0]:
                maxc = self.position_range[0]
            if maxc > self.position_range[1]:
                maxc = self.position_range[1]

            return maxc, np.mean((minr, maxr))
        
        else:
            raise ValueError('The threshold should be either an integer or "flexible"')
                
    
    def visualize_plume_positions(self, plume, plume_position, frame_range=None, label_time=False, title=None):
        '''
        This is a function used to visualize the plume positions, distances, and velocities.

        :param time: time
        :type time: numpy.ndarray

        :param plume_positions: plume positions
        :type plume_positions: numpy.ndarray

        :param frame_range: frame range
        :type frame_range: tuple
        '''
        if not isinstance(frame_range, type(None)):
            plume = plume[frame_range[0]:frame_range[1]]
            plume_position = plume_position[frame_range[0]:frame_range[1]]

        if label_time:
            time = np.arange(0, plume_position.shape[0]) * self.time_interval
            titles = [f'{t:.2e}s' for t in time]
        else:
            titles = np.arange(0, len(plume_position))

        fig, axes = create_axes_grid(len(plume), n_per_row=8, plot_height=1.1)
        axes = axes.flatten()
        # fig, axes = plt.subplots(5, 8, figsize=(16, 10))
        for i in range(len(plume)):
            ax = axes[i]
            x, y = plume_position[i]
            im = ax.imshow(plume[i])
            ax.plot(x, y, 'r|', markersize=15)
            fig.colorbar(im, ax=ax)
            ax.axis('off')
            ax.set_title(titles[i])
        if title:
            plt.suptitle(title)
        plt.tight_layout()
        plt.show()

            
    def visualize_distance_velocity(self, plume_distance, plume_velocity, frame_range=None, index_time=False, ignore_start=0):
        '''
        This is a function used to visualize the plume positions, distances, and velocities.

        :param plume_distance: plume distance
        :type plume_distance: numpy.ndarray

        :param plume_velocity: plume velocity
        :type plume_velocity: numpy.ndarray

        :param frame_range: frame range
        :type frame_range: tuple
        '''
        if not isinstance(frame_range, type(None)):
            plume_distance = plume_distance[frame_range[0]:frame_range[1]]
            plume_velocity = plume_velocity[frame_range[0]:frame_range[1]]

        if ignore_start:
            plume_distance[:ignore_start] = plume_distance[0]
            plume_velocity[:ignore_start] = plume_velocity[0]

        if index_time:
            time = np.arange(0, plume_distance.shape[0]) * self.time_interval
            indexes = [f'{t:.2e}s' for t in time]
        else:
            indexes = np.arange(0, plume_distance.shape[0])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(indexes, plume_distance, '-o', markersize=3)
        axes[0].set_title('Distance')
        axes[0].grid()

        axes[1].plot(indexes, plume_velocity, '-o', markersize=3)
        axes[1].set_title('Velocity')
        axes[1].grid()
        plt.show()
