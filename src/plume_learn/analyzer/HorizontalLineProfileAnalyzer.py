import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

class HorizontalLineProfileAnalyzer:
    def __init__(self, image, row=None, line_width=None):
        self.image = image
        if len(image.shape) == 3:
            self.image = np.mean(image, axis=2).astype(np.uint8)
        self.height, self.width = self.image.shape
        self.profile = None
        self.row = row
        self.line_width = line_width

    def extract_profile(self, row=None, line_width=None):
        if row is not None:
            self.row = row
        if line_width is not None:
            self.line_width = line_width

        self.row = max(0, min(self.row, self.height - 1))
        
        start_row = max(0, self.row - self.line_width // 2)
        end_row = min(self.height, start_row + self.line_width)
        # print( start_row, end_row)
        
        self.profile = np.mean(self.image[start_row:end_row, :], axis=0)
        return self.profile

    def detect_largest_decrease(self, target_x=None, smoothing=True, prominence=0.05, width=1, distance=1, show_profile=False):

        if self.profile is None:
            raise ValueError("Profile not extracted. Call extract_profile first.")
        
        profile_original = self.profile.copy()
        # Optional: Apply Savitzky-Golay filter for smoothing
        if smoothing:
            window_length = min(len(self.profile) // 2, 51)  # Must be odd and less than data length
            if window_length % 2 == 0:
                window_length -= 1
            self.profile = savgol_filter(self.profile, window_length, 3)

        # Calculate the differences between adjacent points
        differences = np.diff(self.profile)
        diff_original = differences.copy()

        # Optional: Apply Savitzky-Golay filter for smoothing
        if smoothing:
            window_length = min(len(differences) // 2, 51)  # Must be odd and less than data length
            if window_length % 2 == 0:
                window_length -= 1
            differences = savgol_filter(differences, window_length, 3)

        if show_profile:
            fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(5, 5))

            ax0.plot(profile_original, 'k')
            ax0.plot(self.profile, 'b')
            ax1.plot(diff_original, 'k')
            ax1.plot(differences, 'b')

        peaks, properties = find_peaks(-differences, prominence=prominence, width=width, distance=distance)

        # print(peaks, properties)
        
        peak_heights = properties['prominences']
        if target_x is not None:
            peak_heights = peak_heights[peaks>target_x+10]
            peaks = peaks[peaks>target_x+10]
        # print(peaks, properties)

        if len(peaks) == 0:
            plt.show()
            return None
        
        largest_decrease_index = np.argmax(peak_heights)
        largest_decrease_position = peaks[largest_decrease_index]
        
        if show_profile:
            ax0.plot(largest_decrease_position, differences[largest_decrease_position], 'ro', markersize=5)
            ax1.plot(largest_decrease_position, differences[largest_decrease_position], 'ro', markersize=5)
            plt.show()

        return largest_decrease_position + 1, self.profile[largest_decrease_position], diff_original, differences

    def detect(self, target_x=None, show_image=True, show_profile=True, show_difference=True):
        if self.profile is None:
            raise ValueError("Profile not extracted. Call extract_profile first.")

        largest_decrease = self.detect_largest_decrease(target_x=target_x, show_profile=show_profile)
        
        if largest_decrease is None:
            if show_image or show_profile or show_difference:
                print("No decreases found in the profile.")
            return None, None
        
        position, magnitude, diff_original, differences = largest_decrease
        

        if show_image or show_profile or show_difference:
            num_plots = sum([show_image, show_difference])
            # num_plots = sum([show_image, show_profile, show_difference])

            fig, axs = plt.subplots(num_plots, 1, figsize=(5, 3*num_plots))
            if num_plots == 1:
                axs = [axs]
            
            plot_index = 0
            
            if show_image:
                axs[plot_index].imshow(self.image, cmap='gray')
                axs[plot_index].axhline(y=self.row, color=(1, 0, 0, 0.3), linestyle='--', linewidth=self.line_width)
                axs[plot_index].plot(position, self.row, 'ro', markersize=5)
                axs[plot_index].set_title('Image with Horizontal Line')
                plot_index += 1
            
            # if show_profile:
            #     axs[plot_index].plot(self.profile)
            #     axs[plot_index].plot(position, self.profile[position], 'ro', markersize=5)
            #     axs[plot_index].set_title('Profile with Largest Decrease')
            #     axs[plot_index].set_xlabel('Column')
            #     axs[plot_index].set_ylabel('Pixel Value')
            #     plot_index += 1
                # print(f"Largest decrease found at row: {position}")
            
            if show_difference:
                axs[plot_index].plot(diff_original, 'b')
                axs[plot_index].plot(position - 1, diff_original[position - 1], 'ro', markersize=5)
                axs[plot_index].plot(differences, 'g')
                axs[plot_index].plot(position - 1, differences[position - 1], 'ro', markersize=5)
                axs[plot_index].set_title('Profile Differences with Largest Decrease')
                axs[plot_index].set_xlabel('Position')
                axs[plot_index].set_ylabel('Difference in Pixel Value')
                print(f"Magnitude of decrease: {magnitude:.2f}")
        
            plt.tight_layout()
            plt.show()
        
        return position, magnitude

# # Example usage:
# analyzer = HorizontalLineProfileAnalyzer(image)
# profile = analyzer.extract_profile(row=125, line_width=5)
# position, magnitude = analyzer.visualize(show_image=True, show_profile=True, show_difference=False)