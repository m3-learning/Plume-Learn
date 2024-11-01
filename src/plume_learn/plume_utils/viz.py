import numpy as np
import matplotlib.pyplot as plt
from plume_learn.plume_utils.data_processing import NormalizeData
from m3util.viz.layout import layout_fig

# Function to label the violin plot
def label_violinplot(ax, data, label_type='average', text_pos='center', value_format='float',
                     offset_parms={'x_type': 'fixed', 'x_value': 0.02, 'y_type': 'fixed', 'y_value': 0.02}):
    """
    Label a violin plot with flexible offsets.

    Args:
        ax: Axes object.
        data: Data for the violin plot.
        label_type (str, optional): Type of label to use ('average' or 'number'). Defaults to 'average'.
        text_pos (str, optional): Position of the label text ('center' or 'right'). Defaults to 'center'.
        value_format (str, optional): Format of the value ('float', 'int', or 'scientific'). Defaults to 'float'.
        offset_type (str, optional): Type of offset ('fixed' or 'ratio'). Defaults to 'ratio'.
        offset_value (float, optional): Value for the offset (either a fixed value or ratio). Defaults to 0.02.

    Returns:
        None
    """
    x_offset_type = offset_parms['x_type']
    x_offset_value = offset_parms['x_value']
    y_offset_type = offset_parms['y_type']
    y_offset_value = offset_parms['y_value']
    
    # Calculate position for each category
    xloc = range(len(data))
    yloc = data.values  # Since data is already the grouped mean values

    for tick, (value, label) in enumerate(zip(yloc, ax.get_xticklabels())):
        if label_type == 'average_value':
            if value_format == 'float':
                label_text = f"{value:.2f}"
            elif value_format == 'int':
                label_text = f"{int(value)}"
            elif value_format == 'scientific':
                label_text = f"{value:.2e}"
        elif label_type == 'total_number':
            label_text = f"n: {len(data)}"
        
        # Calculate offset based on the type
        if x_offset_type == 'fixed':
            x_offset = tick  + x_offset_value  # Add fixed offset value
        elif x_offset_type == 'ratio':
            x_offset = tick  * (1 + x_offset_value)  # Offset by ratio
        
        if y_offset_type == 'fixed':
            y_offset = value + y_offset_value
        elif y_offset_type == 'ratio':
            y_offset = value * (1 + y_offset_value)
        
        # Position text based on text_pos
        if text_pos == 'center':
            # Label at the center of the violin
            ax.text(tick, y_offset, label_text, horizontalalignment='center', size=14, weight='semibold')
        elif text_pos == 'right':
            # Label slightly to the right of the tick
            ax.text(x_offset, y_offset, label_text, horizontalalignment='left', size=14, weight='semibold')



def evaluate_image_histogram(image, outlier_std=3):
    """
    Generate a histogram of image pixel values with Z-score clipping and label mean, min, max, and std.
    
    Parameters:
    image (numpy array): The input image array. Assumes a grayscale image with values in range 0-255.
    z_thresh (float): The Z-score threshold for clipping.
    """
    # Flatten the image to a 1D array
    pixel_values = image.flatten()
    
    # Calculate mean and standard deviation
    mean_val = np.mean(pixel_values)
    std_val = np.std(pixel_values)
    
    # Clip values based on Z-score threshold
    lower_clip = mean_val - outlier_std * std_val
    upper_clip = mean_val + outlier_std * std_val
    clipped_values = pixel_values[(pixel_values >= lower_clip) & (pixel_values <= upper_clip)]
    
    # Calculate statistics on clipped values
    mean_clipped = np.mean(clipped_values)
    min_clipped = np.min(clipped_values)
    max_clipped = np.max(clipped_values)
    std_clipped = np.std(clipped_values)
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(clipped_values, bins=100, range=(lower_clip, upper_clip), alpha=0.3, edgecolor='black')
    plt.title(f'Image Histogram (removing noise outside ±{outlier_std}σ)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    # Add text for the statistics
    plt.axvline(mean_clipped, color='red', linestyle='dashed', linewidth=2)
    plt.text(mean_clipped + np.abs(mean_clipped)*0.1, plt.ylim()[1] * 0.9, f'Mean: {mean_clipped:.2e}\nStd: {std_clipped:.2e}', color='black')
    
    plt.axvline(min_clipped, color='green', linestyle='dashed', linewidth=2)
    plt.text(min_clipped + np.abs(min_clipped)*0.05, plt.ylim()[1] * 0.8, f'Min:\n{min_clipped:.2e}', color='green')
    
    plt.axvline(max_clipped, color='green', linestyle='dashed', linewidth=2)
    plt.text(max_clipped + np.abs(max_clipped)*0.05, plt.ylim()[1] * 0.8, f'Max:\n{max_clipped:.2e}', color='green')
    
    for i in range(3):
        lb, hb = mean_clipped-std_clipped*(i+1), mean_clipped+std_clipped*(i+1)
        plt.axvline(lb, color='blue', linestyle='dashed', linewidth=2, alpha=0.5)
        plt.text(lb + np.abs(lb)*0.05, plt.ylim()[1] * (0.4+i/10), f'-{i+1}σ:\n{lb:.2e}', color='black', alpha=0.8)
        plt.axvline(hb, color='blue', linestyle='dashed', linewidth=2, alpha=0.5)
        plt.text(hb + np.abs(hb)*0.05, plt.ylim()[1] * (0.4+i/10), f'+{i+1}σ:\n{hb:.2e}', color='black', alpha=0.8)
    
    # Show plot
    plt.show()


def show_images(images, labels=None, img_per_row=8, img_height=1, label_size=12, title=None, show_colorbar=False, 
                clim=3, cmap='viridis', scale_range=False, hist_bins=None, show_axis=False, fig=None, axes=None, save_path=None):
    
    '''
    Plots multiple images in grid.
    
    images
    labels: labels for every images;
    img_per_row: number of images to show per row;
    img_height: height of image in axes;
    show_colorbar: show colorbar;
    clim: int or list of int, value of standard deviation of colorbar range;
    cmap: colormap;
    scale_range: scale image to a range, default is False, if True, scale to 0-1, if a tuple, scale to the range;
    hist_bins: number of bins for histogram;
    show_axis: show axis
    '''
    
    assert type(images) == list or type(images) == np.ndarray, "do not use torch.tensor for hist"
    if type(clim) == list:
        assert len(images) == len(clim), "length of clims is not matched with number of images"

    h = images[0].shape[1] // images[0].shape[0]*img_height + 1
    if isinstance(labels, type(None)):
        labels = range(len(images))
        
    if isinstance(axes, type(None)):
        if hist_bins: # add a row for histogram
            fig, axes = layout_fig(graph=len(images)*2, mod=img_per_row, figsize=(None, img_height*2))
        else:
            fig, axes = layout_fig(graph=len(images), mod=img_per_row, figsize=(None, img_height))
        
    axes = axes.flatten()
    # if hist_bins:
    #     trim_axes(axes, len(images)*2)
    # else:
    #     trim_axes(axes, len(images))


    for i, img in enumerate(images):

        if hist_bins:
            # insert histogram in after the row
            # index = i + (i//img_per_row)*img_per_row
            index = i*2

#         if torch.is_tensor(x_tensor):
#             if img.requires_grad: img = img.detach()
#             img = img.numpy()
        else:
            index = i
            
        if isinstance(scale_range, bool): 
            if scale_range: img = NormalizeData(img)
                    
        # if len(images) <= img_per_row and not hist_bins:
        #     index = i%img_per_row
        # else:
        #     index = (i//img_per_row)*n, i%img_per_row
        # print(i, index)

        axes[index].set_title(labels[i], fontsize=label_size)
        im = axes[index].imshow(img, cmap=cmap)

        if show_colorbar:
            m, s = np.mean(img), np.std(img) 
            if type(clim) == list:
                im.set_clim(m-clim[i]*s, m+clim[i]*s) 
            elif type(clim) == int:
                im.set_clim(m-clim*s, m+clim*s) 
            # else:
            #     im.set_clim(0, 1)

            fig.colorbar(im, ax=axes[index])
            
        if show_axis:
            axes[index].tick_params(axis="x",direction="in", top=True)
            axes[index].tick_params(axis="y",direction="in", right=True)
        else:
            axes[index].axis('off')

        if hist_bins:
            # index_hist = (i//img_per_row)*2+1, i%img_per_row
            # index_hist = index+img_per_row
            # index_hist = index*2+1
            index_hist = index+1
            # print(index, index_hist)
            h = axes[index_hist].hist(img.flatten(), bins=hist_bins)

        if title:
            fig.suptitle(title, fontsize=16, y=1.01)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.95)

    # if save_path and isinstance(axes, type(None)): # this is not effective because axes are defined after the function is called
    #     plt.savefig(save_path+'.svg', dpi=300)
    #     plt.savefig(save_path+'.png', dpi=300)
    
    # print(axes)
    # if isinstance(axes, type(None)): # this is not effective because axes are defined after the function is called
    #     plt.show()