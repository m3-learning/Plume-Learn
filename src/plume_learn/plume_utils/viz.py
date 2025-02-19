import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plume_learn.plume_utils.data_processing import NormalizeData
from m3util.viz.layout import layout_fig

def set_cbar(fig, ax, cbar_label=None, scientific_notation=True, logscale=False, tick_in=True, ticklabel_fontsize=10, labelpad=4, fontsize=10):
    cbar = fig.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02, shrink=1)

    if scientific_notation:
        if logscale:
            formatter = ticker.LogFormatterMathtext(base=10)  # Logarithmic formatter
        else:
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))

        cbar.ax.yaxis.set_major_formatter(formatter)
    
    if cbar_label:
        cbar.ax.xaxis.set_label_position('bottom')  # Move label to the bottom
        cbar.ax.xaxis.set_ticks_position('bottom')  # Move ticks to the bottom as well
        cbar.set_label(cbar_label, rotation=0, labelpad=labelpad, fontsize=fontsize)  # Horizontal label with padding
        cbar.ax.yaxis.set_label_coords(1.5, -0.04)  # (x, y) relative to the colorbar

    if tick_in:
        cbar.ax.tick_params(direction='in', labelsize=ticklabel_fontsize)  # Set tick direction to 'in'
    else:
        cbar.ax.tick_params(labelsize=ticklabel_fontsize)

def set_labels(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, yaxis_style='sci', label_fontsize=12, title_fontsize=12, ticklabel_fontsize=10, scientific_notation_fontsize=8, logscale=False, legend=None, legend_fontsize=8, legend_loc='best', show_ticks=True, ticks_both_sides=True, tick_padding=10):
    """
    Set labels and other properties of the given axes.

    Args:
        ax: Axes object.
        xlabel (str, optional): X-axis label. Defaults to None.
        ylabel (str, optional): Y-axis label. Defaults to None.
        title (str, optional): Plot title. Defaults to None.
        xlim (tuple, optional): X-axis limits. Defaults to None.
        ylim (tuple, optional): Y-axis limits. Defaults to None.
        yaxis_style (str, optional): Y-axis style. Defaults to 'sci'.
        logscale (bool, optional): Use log scale on the y-axis. Defaults to False.
        legend (list, optional): Legend labels. Defaults to None.
        ticks_both_sides (bool, optional): Display ticks on both sides of the axes. Defaults to True.

    Returns:
        None
    """
    if type(xlabel) != type(None): ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if type(ylabel) != type(None): ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if type(title) != type(None): ax.set_title(title, fontsize=title_fontsize)
    if type(xlim) != type(None): ax.set_xlim(xlim)
    if type(ylim) != type(None): ax.set_ylim(ylim)
    
    if yaxis_style == 'sci':
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # Force scientific notation
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(scientific_notation_fontsize)  # Adjust font size for the magnitude label

    # elif yaxis_style == 'float':
    #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        # ax.ticklabel_format(axis='y', style='plain')   

    if logscale: ax.set_yscale("log") 
    if legend: 
        if legend == 'auto' or legend == True: 
            handles, labels = ax.get_legend_handles_labels()  # Automatically get handles and labels from the plot
            ax.legend(handles, labels, fontsize=legend_fontsize, loc=legend_loc)
        else:
            ax.legend(legend, fontsize=legend_fontsize, loc=legend_loc)
            
    if show_ticks:
        ax.tick_params(axis="x", direction="in", length=5, labelsize=ticklabel_fontsize, pad=tick_padding)
        ax.tick_params(axis="y", direction="in", length=5, labelsize=ticklabel_fontsize, pad=tick_padding)
        if ticks_both_sides:
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
    elif show_ticks == False:
        ax.tick_params(axis="x", direction="in", length=0, labelsize=ticklabel_fontsize, pad=tick_padding)
        ax.tick_params(axis="y", direction="in", length=0, labelsize=ticklabel_fontsize, pad=tick_padding)


def to_scientific_10_power_format(value):
    # Convert the value to scientific notation
    sci_notation = "{:.2e}".format(value)
    
    # Split the base and exponent
    base, exponent = sci_notation.split('e')
    
    # Convert exponent to an integer to remove extra + and leading zeros
    exponent = int(exponent)
    
    # Format in LaTeX style for Matplotlib or Jupyter Notebook
    label_text = f"{base}\\times10^{{{exponent}}}"
    
    return rf"${label_text}$"  # Add $ symbols for LaTeX formatting


# Function to label the violin plot
def label_violinplot(ax, data, label_type='average', text_pos='center', value_format='float', text_size=14, 
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
                label_text = to_scientific_10_power_format(value)
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
            ax.text(tick, y_offset, label_text, horizontalalignment='center', size=text_size, weight='semibold')
        elif text_pos == 'right':
            # Label slightly to the right of the tick
            ax.text(x_offset, y_offset, label_text, horizontalalignment='left', size=text_size, weight='semibold')



def evaluate_image_histogram(image, outlier_std=3):
    """
    Generate a histogram of image pixel values with Z-score clipping and label mean, min, max, and std.
    
    Parameters:
    image (numpy array): The input image array. Assumes a grayscale image with values in range 0-255.
    outlier_std (float): The Z-score threshold for clipping.
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


def show_images(images, labels=None, img_per_row=8, img_height=1, label_size=12, title=None, show_colorbar=False, clim=3, cmap='viridis', scale_range=False, hist_bins=None, show_axis=False, fig=None, axes=None, save_path=None):
    
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
    if labels == 'index':
        labels = range(len(images))
    elif labels == None:
        labels = ['']*len(images)
        
    if isinstance(axes, type(None)):
        if hist_bins: # add a row for histogram
            fig, axes = layout_fig(graph=len(images)*2, mod=img_per_row, figsize=(None, img_height*2))
        else:
            fig, axes = layout_fig(graph=len(images), mod=img_per_row, figsize=(None, img_height))
        
    # axes = axes.flatten()
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

        if labels[i]:
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