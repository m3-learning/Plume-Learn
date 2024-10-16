import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from .AutoAlign import align_plumes, visualize_corners
from .PlumeDataset import plume_dataset
from .PlumeMetrics import PlumeMetrics
from .Velocity import VelocityCalculator
# from .plume_utils import smooth_curve
# from viz import show_images
# from .HorizontalLineProfileAnalyzer import HorizontalLineProfileAnalyzer

def load_plumes_and_align(file_path, group_name='PLD_Plumes', plume_name='1-SrRuO3', pre_plume_name=None, frame_view_index=0, plume_view_index=0):
    """
    Load plume data, visualize it, and prepare it for analysis.
    
    """
    plume_ds = plume_dataset(file_path=file_path, group_name=group_name)
    keys = plume_ds.dataset_names()
    print(f"Available datasets: {keys}")
    
    ds_name = file_path.split('/')[-1].split('_')[0]
    plumes = plume_ds.load_plumes(plume_name)
    
    if pre_plume_name == None:
        pre_plume_name = f'{plume_name}_Pre'
        
    frame_view = plume_ds.load_plumes(pre_plume_name)[plume_view_index, frame_view_index]
    fig = px.imshow(frame_view)
    return plumes, ds_name, frame_view, fig

def run_plume_analysis(plumes, ds_name, frame_view, coords, coords_path, standard_coords_path, output_csv_path, viz_parms=None, metric_parms=None, align=True):
    """
    Run the full plume analysis with flexible parameters.
    """
    visualize_corners(frame_view, coords, color='k', marker_size=200, style='both')
    np.save(coords_path, coords)
    
    coords_standard = np.load(standard_coords_path)

    if viz_parms is None:
        print("No visualization parameters provided. Using default parameters.")
        viz_parms = {
            'viz': True, 
            'index': 5, 
            'viz_index': list(np.arange(0, 32, 1)), 
            'plume_name': ds_name
        }
    
    align_parms = {'align': align, 'coords': coords, 'coords_standard': coords_standard}
    
    if metric_parms is None:
        print("No metric parameters provided. Using default parameters.")
        metric_parms = {
            'time_interval': 500e-9,
            'threshold_list': [5, 200, 'flexible'],
            'rename_dataset': True
        }
        
    if align:
        start_position = np.round(np.mean(coords_standard[:2], axis=0)).astype(np.int32) # start position of plume  (x, y)
        position_range = np.min(coords_standard[:,0]), np.max(coords_standard[:,0]) # x position range
        metric_parms['start_position'] = start_position
        metric_parms['position_range'] = position_range
    
    df_all = []
    for threshold in metric_parms['threshold_list']:
        print(f"Running analysis for threshold: {threshold}")
        metric_parms['threshold'] = threshold
        df = analyze_function(plumes, viz_parms, metric_parms, align_parms)
        df_all.append(df)
    df_all = pd.concat(df_all)
    df_all.to_csv(output_csv_path)
    print(f"Results saved to {output_csv_path}")
    
    plot_configs = [
        ('time_index', 'Area', 'Plume Area'),
        ('time_index', 'Velocity', 'Plume Velocity'),
        ('Distance', 'Area', 'Plume Area'),
        ('Distance', 'Velocity', 'Plume Velocity')
    ]
    
    for x, y, title in plot_configs:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(x=x, y=y, hue="Threshold", data=df_all)
        plt.title(title)
        plt.show()

def analyze_function(plumes, viz_parms, metric_parms, align_parms={'align':False, 'coords':None, 'coords_standard':None}):
    # plumes = plumes[x_range[0]:x_range[1]]

    # visualization parameters
    viz = viz_parms['viz']
    index = viz_parms['index']
    viz_index = viz_parms['viz_index']
    plume_name = viz_parms['plume_name']
    progress_bar = viz_parms['progress_bar']

    # metric parameters
    time_interval = metric_parms['time_interval']
    start_position = metric_parms['start_position']
    position_range = metric_parms['position_range']
    threshold = metric_parms['threshold']
    P = PlumeMetrics(time_interval, start_position, position_range, threshold=threshold, progress_bar=progress_bar)
    V = VelocityCalculator(time_interval, start_position, position_range, threshold=threshold, progress_bar=progress_bar)

    # align plumes
    if align_parms['align']:
        if align_parms['coords'] is None or align_parms['coords_standard'] is None:
            raise ValueError('Please provide the coordinates for alignment')
        else:
            plumes = align_plumes(plumes, align_parms['coords'], align_parms['coords_standard'])

    # visualize plumes
    # if viz:
    #     show_images(plumes[index][viz_index], img_per_row=16, img_height=1, title=plume_name)
    #     plt.show()

    # calculate area for plumes
    areas, coords, labeled_images = P.calculate_area_for_plumes(plumes, return_format='dataframe')
    df_area = P.to_df(areas)
    # print(plumes[index][viz_index].shape, areas[index][viz_index].shape, coords[index][viz_index].shape, labeled_images[index][viz_index].shape)
    if viz:
        P.viz_blob_plume(plumes[index][viz_index], areas[index][viz_index], coords[index][viz_index], labeled_images[index][viz_index], title=f'{plume_name}-Area')

    # calculate velocity for plumes
    plume_positions, plume_distances, plume_velocities = V.calculate_distance_area_for_plumes(plumes)
    df_velocity = V.to_df(plume_positions, plume_distances, plume_velocities)
    # print(plumes[index][viz_index].shape, plume_positions[index][viz_index].shape, plume_distances[index][viz_index].shape, plume_velocities[index][viz_index].shape)
    if viz:
        V.visualize_plume_positions(plumes[index][viz_index], plume_positions[index][viz_index], label_time=False, title=f'{plume_name}-plume position')

    df = pd.concat([df_velocity, df_area], axis=1)
    if metric_parms['rename_dataset']:
        # df = df.rename(columns={'Distance': f'Distance({threshold})'})
        # df = df.rename(columns={'Velocity': f'Velocity({threshold})'})
        # df = df.rename(columns={'Area': f'Area({threshold})'})
        df['Threshold'] = str(threshold)

    df['Growth'] = plume_name
    return df