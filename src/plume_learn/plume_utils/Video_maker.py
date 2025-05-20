import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def make_video(image_sequences, titles=None, output="video.mp4", fps=5, cmap='viridis', clim='auto'):
    """
    Create an MP4 video from one or more image sequences, with fixed color scaling and colorbars.

    Parameters
    ----------
    image_sequences : list of np.ndarray
        List of 3D arrays (num_frames, H, W) or 4D arrays (num_frames, H, W, 3) for RGB.
        
    titles : list of str, optional
        A list of titles, one per frame. If None, no title is shown.
        
    output : str
        Output MP4 filename.
        
    fps : int
        Frames per second.
        
    cmap : str
        Colormap for grayscale images.

    clim : 'auto' | 'global' | (vmin, vmax)
        Color scale limits. Use 'auto' for per-frame autoscaling,
        'global' to scale based on all frames, or provide (vmin, vmax) tuple.
    """
    num_sequences = len(image_sequences)
    num_frames = image_sequences[0].shape[0]

    fig, axes = plt.subplots(1, num_sequences, figsize=(4 * num_sequences + 1, 3))
    if num_sequences == 1:
        axes = [axes]

    ims = []
    vlims = []

    for seq in image_sequences:
        if clim == 'global':
            mean, std = np.mean(seq), np.std(seq)
            vlims.append((mean - 3 * std, mean + 3 * std))
        elif isinstance(clim, tuple):
            vlims.append(clim)
        else:
            vlims.append(None)  # use auto per-frame

    for ax, seq, vlim in zip(axes, image_sequences, vlims):
        im = ax.imshow(seq[0], cmap=cmap if seq.ndim == 3 else None)
        if vlim is not None:
            im.set_clim(*vlim)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        ax.axis('off')
        ims.append(im)

    title_obj = plt.title(titles[0] if titles else "", fontsize=12)

    def update(frame):
        for i, (im, seq, vlim) in enumerate(zip(ims, image_sequences, vlims)):
            im.set_data(seq[frame])
            if vlim is None:  # auto per-frame scaling
                im.set_clim(np.min(seq[frame]), np.max(seq[frame]))
        if titles:
            title_obj.set_text(titles[frame])
        return ims + [title_obj]

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(output, writer=writer)
    plt.close(fig)
    print(f"Video saved to {output}")
