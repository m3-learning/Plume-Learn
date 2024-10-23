import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path, patches, patheffects
from plume_learn.utils import NormalizeData

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


def create_axes_grid(n_plots, n_per_row, plot_height, figsize='auto'):
    """
    Create a grid of axes.

    Args:
        n_plots: Number of plots.
        n_per_row: Number of plots per row.
        plot_height: Height of each plot.
        n_rows: Number of rows. If None, it is calculated from n_plots and n_per_row.
        
    Returns:
        axes: Axes object.
    """
    
    if figsize == 'auto':
        figsize = (16, plot_height*n_plots//n_per_row+1)
    elif isinstance(figsize, tuple):
        pass
    elif figsize != None:
        raise ValueError("figsize must be a tuple or 'auto'")
    
    fig, axes = plt.subplots(n_plots//n_per_row+1*int(n_plots%n_per_row>0), n_per_row, figsize=figsize)
    trim_axes(axes, n_plots)
    return fig, axes
        

def trim_axes(axs, N):

    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    # axs = axs.flatten()
    # for ax in axs[N:]:
    #     ax.remove()
    # return axs[:N]
    for i in range(N, len(axs.flatten())):
        axs.flatten()[i].remove()
    return axs.flatten()[:N]


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
            fig, axes = create_axes_grid(len(images)*2, img_per_row, img_height*2, figsize='auto')
        else:
            fig, axes = create_axes_grid(len(images), img_per_row, img_height, figsize='auto')
        
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
    

    
def labelfigs(ax, number=None, style="wb",
              loc="tl", string_add="", size=8,
              text_pos="center", inset_fraction=(0.15, 0.15), **kwargs):

    # initializes an empty string
    text = ""

    # Sets up various color options
    formatting_key = {
        "wb": dict(color="w", linewidth=.75),
        "b": dict(color="k", linewidth=0),
        "w": dict(color="w", linewidth=0),
    }

    # Stores the selected option
    formatting = formatting_key[style]

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_inset = (xlim[1] - xlim[0]) * inset_fraction[1]
    y_inset = (ylim[1] - ylim[0]) * inset_fraction[0]

    if loc == 'tl':
        x, y = xlim[0] + x_inset, ylim[1] - y_inset
    elif loc == 'tr':
        x, y = xlim[1] - x_inset, ylim[1] - y_inset
    elif loc == 'bl':
        x, y = xlim[0] + x_inset, ylim[0] + y_inset
    elif loc == 'br':
        x, y = xlim[1] - x_inset, ylim[0] + y_inset
    elif loc == 'ct':
        x, y = (xlim[0] + xlim[1]) / 2, ylim[1] - y_inset

    elif loc == 'cb':
        x, y = (xlim[0] + xlim[1]) / 2, ylim[0] + y_inset
    else:
        raise ValueError(
            "Invalid position. Choose from 'tl', 'tr', 'bl', 'br', 'ct', or 'cb'.")

    text += string_add

    if number is not None:
        text += number_to_letters(number)

    text_ = ax.text(x, y, text, va='center', ha='center',
                      path_effects=[patheffects.withStroke(
                      linewidth=formatting["linewidth"], foreground="k")],
                      color=formatting["color"], size=size, **kwargs
                      )

    text_.set_zorder(np.inf)

    
def number_to_letters(num):
    letters = ''
    while num >= 0:
        num, remainder = divmod(num, 26)
        letters = chr(97 + remainder) + letters
        num -= 1  # decrease num by 1 because we have processed the current digit
    return letters



def path_maker(axes, vertices, facecolor='none', edgecolor='k', linestyle='-', lineweight=1, shape='rect'):
    """
    Adds a path (rectangular or custom shape) to the figure.

    Parameters:
    ----------
    axes : matplotlib axes
        Axes to which the path will be added.
    vertices : list of tuples or numpy array
        Vertices of the path. For 'rect', it should have 4 coordinates.
    facecolor : str, optional
        Face color of the path.
    edgecolor : str, optional
        Edge color of the path.
    linestyle : str, optional
        Line style of the path.
    lineweight : float, optional
        Line weight of the path.
    shape : str, optional
        Shape type ('rect' for rectangle or 'custom' for custom polygon).
    """
    if shape == 'rect':
        vertices = [(vertices[0], vertices[2]), (vertices[1], vertices[2]), (vertices[1], vertices[3]), (vertices[0], vertices[3]), (0, 0)]
        codes = [path.Path.MOVETO] + [path.Path.LINETO] * 3 + [path.Path.CLOSEPOLY]
    elif shape == 'custom':
        codes = [path.Path.MOVETO] + [path.Path.LINETO] * (len(vertices) - 2) + [path.Path.CLOSEPOLY]
    
    vertices = np.array(vertices, float)
    path_ = path.Path(vertices, codes)
    path_patch = patches.PathPatch(path_, facecolor=facecolor, edgecolor=edgecolor, ls=linestyle, lw=lineweight)
    axes.add_patch(path_patch)


def scalebar(axes, image_size, scale_size, units='nm', loc='br', custom_position=None):
    """
    Adds a scalebar to the figure with more flexible positioning options.

    Parameters:
    ----------
    axes : matplotlib axes
        Axes to which the scalebar will be added.
    image_size : int
        Size of the image.
    scale_size : float
        Size of the scalebar.
    units : str, optional
        Units of the scalebar.
    loc : str, optional
        Location of the scalebar (default: 'br' - bottom right).
    custom_position : tuple, optional
        Custom position (x_start, x_end, y_start, y_end) for the scalebar.
    """
    # Make sure the axes limits are up to date
    axes.relim()
    axes.autoscale_view()

    if custom_position:
        x_start, x_end, y_start, y_end = custom_position
    else:
        x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]

        # Adjust the position based on image coordinates
        if loc == 'br':
            x_start = x_lim[0] + 0.9 * x_range
            x_end = x_start - (scale_size / image_size) * x_range
            y_start = y_lim[0] + 0.1 * y_range
            y_end = y_start + 0.025 * y_range

    # Draw the scalebar using the path_maker function
    path_maker(axes, [x_start, x_end, y_start, y_end], facecolor='w', edgecolor='k', linestyle='-', lineweight=1)

    # Add an offset to the text label above the scalebar
    text_offset = 0.05 * y_range  # Adjust the text offset based on the vertical range
    axes.text((x_start + x_end) / 2, y_end + text_offset,  # Move the text label above the scalebar
              f'{scale_size} {units}',
              size=14, weight='bold', ha='center', va='center', color='w',
              path_effects=[patheffects.withStroke(linewidth=1.5, foreground="k")])