import os
import sys
import time
import cv2
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tkinter as tk
import xml.etree.ElementTree as ET
import subprocess
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Markdown,display
from xml.dom import minidom
from PyQt5.QtWidgets import (
    QApplication, QWidget, QSlider, QLabel, QGridLayout, QScrollArea, 
    QVBoxLayout, QFrame, QRadioButton, QButtonGroup, QCheckBox,
)
from PyQt5.QtCore import Qt
from functools import partial
from scipy.ndimage import binary_dilation
from io import BytesIO
from termcolor import colored
from collections import deque
from datetime import datetime


"""
    sys.path.append('../../package/kinematics_helper/') # for 'transforms'
"""
from transforms import (
    t2p,
    r2quat,
    rpy2r,
)

def trim_scale(x,th):
    """
    Trim the scale of the input array such that its maximum absolute value does not exceed a given threshold.

    Parameters:
        x (np.array): Input array.
        th (float): Threshold value.

    Returns:
        np.array: Scaled array with its maximum absolute value limited to th.
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def compute_view_params(
        camera_pos,
        target_pos,
        up_vector = np.array([0,0,1]),
    ):
    """
    Compute view parameters (azimuth, distance, elevation, and lookat point) for a camera in 3D space.

    Parameters:
        camera_pos (np.ndarray): 3D position of the camera.
        target_pos (np.ndarray): 3D position of the target.
        up_vector (np.ndarray): 3D up vector (default is [0, 0, 1]).

    Returns:
        tuple: (azimuth [deg], distance, elevation [deg], lookat point)
    """
    # Compute camera-to-target vector and distance
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    # Compute azimuth and elevation
    azimuth = np.arctan2(cam_to_target[1], cam_to_target[0])
    azimuth = np.rad2deg(azimuth) # [deg]
    elevation = np.arcsin(cam_to_target[2] / distance)
    elevation = np.rad2deg(elevation) # [deg]

    # Compute lookat point
    lookat = target_pos

    # Compute camera orientation matrix
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # Return computed values
    return azimuth, distance, elevation, lookat

def get_idxs(list_query, list_domain):
    """
    Get the corresponding indices in list_query for the items present in list_domain.

    Parameters:
        list_query (list): The list in which to search for items.
        list_domain (list): The list of items whose indices are desired.

    Returns:
        list: A list of indices from list_query for items found in list_domain.

    Raises:
        TypeError: If either input is not a list.
    """
    if not isinstance(list_query, list) or not isinstance(list_domain, list):
        raise TypeError("[get_idxs] Both inputs should be lists.")
    return [list_query.index(item) for item in list_domain if item in list_query]

def get_idxs_contain(list_query,list_substring):
    """
    Get indices of elements in list_query that contain any of the substrings specified in list_substring.

    Parameters:
        list_query (list): List of strings to search.
        list_substring (list): List of substrings to look for.

    Returns:
        list: Indices of elements in list_query that contain any of the substrings.
    """
    idxs = [i for i, s in enumerate(list_query) if any(sub in s for sub in list_substring)]
    return idxs

def get_colors(n_color=10, cmap_name='gist_rainbow', alpha=1.0):
    """
    Generate a list of diverse colors using a specified colormap.

    Returns:
        list: A list of RGBA tuples.
    """
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.linspace(0, 1, n_color))
    rgba[:, 3] = alpha
    return [tuple(r) for r in rgba]

def sample_xyzs(n_sample=1,x_range=(-1,-1),y_range=(-1,+1),z_range=(0,0),min_dist=0.1,xy_margin=0.0):
    """
    Sample 3D points within specified ranges ensuring a minimum distance between points.

    Parameters:
        n_sample (int): Number of points to sample.
        x_range (list): [min, max] range for x-coordinate.
        y_range (list): [min, max] range for y-coordinate.
        z_range (list): [min, max] range for z-coordinate.
        min_dist (float): Minimum allowed distance between points.
        xy_margin (float): Margin to apply for x and y dimensions.

    Returns:
        np.array: Array of sampled 3D points with shape (n_sample, 3).
    """
    xyzs = np.zeros((n_sample,3))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0]+xy_margin,high=x_range[1]-xy_margin)
            y_rand = np.random.uniform(low=y_range[0]+xy_margin,high=y_range[1]-xy_margin)
            z_rand = np.random.uniform(low=z_range[0],high=z_range[1])
            xyz = np.array([x_rand,y_rand,z_rand])
            if p_idx == 0: break
            devc = cdist(xyz.reshape((-1,3)),xyzs[:p_idx,:].reshape((-1,3)),'euclidean')
            if devc.min() > min_dist: break # minimum distance between objects
        xyzs[p_idx,:] = xyz
    return xyzs

def sample_xys(n_sample=1,x_range=[0,1],y_range=[0,1],min_dist=0.1,xy_margin=0.0):
    """
    Sample 2D points within specified ranges ensuring a minimum distance between points.

    Parameters:
        n_sample (int): Number of points to sample.
        x_range (list): [min, max] range for x-coordinate.
        y_range (list): [min, max] range for y-coordinate.
        min_dist (float): Minimum allowed distance between points.
        xy_margin (float): Margin to apply for x and y dimensions.

    Returns:
        np.array: Array of sampled 2D points with shape (n_sample, 2).
    """
    xys = np.zeros((n_sample,2))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0]+xy_margin,high=x_range[1]-xy_margin)
            y_rand = np.random.uniform(low=y_range[0]+xy_margin,high=y_range[1]-xy_margin)
            xy = np.array([x_rand,y_rand])
            if p_idx == 0: break
            devc = cdist(xy.reshape((-1,2)),xys[:p_idx,:].reshape((-1,2)),'euclidean')
            if devc.min() > min_dist: break # minimum distance between objects
        xys[p_idx,:] = xy
    return xys

def save_png(img,png_path,verbose=False):
    """
    Save an image to a PNG file.

    Parameters:
        img (np.array): Image data.
        png_path (str): File path to save the PNG.
        verbose (bool): If True, print status messages.

    Returns:
        None
    """
    directory = os.path.dirname(png_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        if verbose:
            print ("[%s] generated."%(directory))
    # Save to png
    plt.imsave(png_path,img)
    if verbose:
        print ("[%s] saved."%(png_path))
        
class MultiSliderClass(object):
    """
    GUI class to create and manage multiple sliders using Tkinter.
    """
    def __init__(
            self,
            n_slider      = 10,
            title         = 'Multiple Sliders',
            window_width  = 500,
            window_height = None,
            x_offset      = 0,
            y_offset      = 100,
            slider_width  = 400,
            label_width   = None, # dummy parameter 
            label_texts   = None,
            slider_mins   = None,
            slider_maxs   = None,
            slider_vals   = None,
            resolution    = None,
            resolutions   = None,
            fontsize      = 10,
            verbose       = True
        ):
        """
        Initialize the MultiSliderClass with the specified slider parameters.

        Parameters:
            n_slider (int): Number of sliders.
            title (str): Window title.
            window_width (int): Width of the window.
            window_height (int): Height of the window. If None, it is computed based on n_slider.
            x_offset (int): X offset for window placement.
            y_offset (int): Y offset for window placement.
            slider_width (int): Width of each slider.
            label_width (int): Width of the label (dummy parameter).
            label_texts (list): List of texts for slider labels.
            slider_mins (list): List of minimum values for sliders.
            slider_maxs (list): List of maximum values for sliders.
            slider_vals (list): Initial slider values.
            resolution (float): Resolution for all sliders.
            resolutions (list): List of resolutions for each slider.
            fontsize (int): Font size for labels.
            verbose (bool): If True, print status messages.
        """
        self.n_slider      = n_slider
        self.title         = title
        
        self.window_width  = window_width
        if window_height is None:
            self.window_height = self.n_slider*40
        else:
            self.window_height = window_height
        self.x_offset      = x_offset
        self.y_offset      = y_offset
        self.slider_width  = slider_width
        self.resolution    = resolution
        self.resolutions   = resolutions
        self.fontsize      = fontsize
        self.verbose       = verbose
        
        # Slider values
        self.slider_values = np.zeros(self.n_slider)
        
        # Initial/default slider settings
        self.label_texts   = label_texts
        self.slider_mins   = slider_mins
        self.slider_maxs   = slider_maxs
        self.slider_vals   = slider_vals
        
        # Create main window
        self.gui = tk.Tk()
        
        self.gui.title("%s"%(self.title))
        self.gui.geometry(
            "%dx%d+%d+%d"%
            (self.window_width,self.window_height,self.x_offset,self.y_offset))
        
        # Create vertical scrollbar
        self.scrollbar = tk.Scrollbar(self.gui,orient=tk.VERTICAL)
        
        # Create a Canvas widget with the scrollbar attached
        self.canvas = tk.Canvas(self.gui,yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure the scrollbar to control the canvas
        self.scrollbar.config(command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create a frame inside the canvas to hold the sliders
        self.sliders_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0,0),window=self.sliders_frame,anchor=tk.NW)
        
        # Create sliders
        self.sliders = self.create_sliders()
        
        # Update the canvas scroll region when the sliders_frame changes size
        self.sliders_frame.bind("<Configure>",self.cb_scroll)

        # You may want to do this in the main script
        for _ in range(100): self.update() # to avoid GIL-related error 
        
    def cb_scroll(self,event): 
        """
        Callback function to update the scroll region when the slider frame size changes.

        Parameters:
            event: Tkinter event object.

        Returns:
            None
        """        
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def cb_slider(self,slider_idx,slider_value):
        """
        Callback function for slider value changes.

        Parameters:
            slider_idx (int): Index of the slider.
            slider_value (float): New value of the slider.

        Returns:
            None
        """
        self.slider_values[slider_idx] = slider_value # append
        if self.verbose:
            print ("slider_idx:[%d] slider_value:[%.1f]"%(slider_idx,slider_value))
        
    def create_sliders(self):
        """
        Create slider widgets for the GUI.

        Returns:
            list: List of slider widget objects.
        """
        sliders = []
        for s_idx in range(self.n_slider):
            # Create label
            if self.label_texts is None:
                label_text = "Slider %02d "%(s_idx)
            else:
                label_text = "[%d/%d] %s"%(s_idx,self.n_slider,self.label_texts[s_idx])
            slider_label = tk.Label(self.sliders_frame, text=label_text,font=("Helvetica",self.fontsize))
            slider_label.grid(row=s_idx,column=0,padx=0,pady=0)
            
            # Create slider
            if self.slider_mins is None: slider_min = 0
            else: slider_min = self.slider_mins[s_idx]
            if self.slider_maxs is None: slider_max = 100
            else: slider_max = self.slider_maxs[s_idx]
            if self.slider_vals is None: slider_val = 50
            else: slider_val = self.slider_vals[s_idx]

            # Resolution
            if self.resolution is None: # if none, divide the range with 100
                resolution = (slider_max-slider_min)/100
            else:
                resolution = self.resolution 
            if self.resolutions is not None:
                resolution = self.resolutions[s_idx]

            slider = tk.Scale(
                self.sliders_frame,
                from_      = slider_min,
                to         = slider_max,
                orient     = tk.HORIZONTAL,
                command    = lambda value,idx=s_idx:self.cb_slider(idx,float(value)),
                resolution = resolution,
                length     = self.slider_width,
                font       = ("Helvetica",self.fontsize),
            )
            slider.grid(row=s_idx,column=1,padx=0,pady=0,sticky=tk.W)
            slider.set(slider_val)
            sliders.append(slider)
            
        return sliders
    
    def update(self):
        """
        Update the GUI window if it exists.

        Returns:
            None
        """        
        if self.is_window_exists():
            self.gui.update()
        
    def run(self):
        """
        Start the Tkinter main loop for the slider GUI.

        Returns:
            None
        """        
        self.gui.mainloop()
        
    def is_window_exists(self):
        """
        Check if the GUI window still exists.

        Returns:
            bool: True if the window exists, False otherwise.
        """        
        try:
            return self.gui.winfo_exists()
        except tk.TclError:
            return False
        
    def get_slider_values(self):
        """
        Get the current values of all sliders.

        Returns:
            np.array: Array of slider values.
        """        
        return self.slider_values
    
    def get_values(self):
        """
        Alias for get_slider_values.

        Returns:
            np.array: Array of slider values.
        """        
        return self.slider_values
        
    def set_slider_values(self,slider_values):
        """
        Set the values of all sliders.

        Parameters:
            slider_values (list or np.array): New slider values.

        Returns:
            None
        """        
        self.slider_values = slider_values
        for slider,slider_value in zip(self.sliders,self.slider_values):
            slider.set(slider_value)

    def set_slider_value(self,slider_idx,slider_value):
        """
        Set the value of a specific slider.

        Parameters:
            slider_idx (int): Index of the slider.
            slider_value (float): New value for the slider.

        Returns:
            None
        """        
        self.slider_values[slider_idx] = slider_value
        slider = self.sliders[slider_idx]
        slider.set(slider_value)
    
    def close(self):
        """
        Close the slider GUI.

        Returns:
            None
        """        
        if self.is_window_exists():
            # some loop
            for _ in range(100): self.update() # to avoid GIL-related error 
            # Close 
            self.gui.destroy()
            self.gui.quit()
            self.gui.update()                
            

class MultiSliderWidgetClass(QWidget):
    """
    GUI class to create and manage multiple sliders using PyQt5.
    """    
    def __init__(
            self,
            n_slider      = None, # to make it comaptible with 'MultiSliderClass'
            n_sliders     = 5,
            title         = 'PyQt5 MultiSlider',
            window_width  = 500,
            window_height = None,
            x_offset      = 100,
            y_offset      = 100,
            label_width   = 200,
            slider_width  = 400,
            label_texts   = None,
            slider_mins   = None,
            slider_maxs   = None,
            slider_vals   = None,
            resolutions   = None,
            max_change    = 0.1,
            max_changes   = None,
            fontsize      = 10,
            verbose       = True,
        ):
        """
        Initialize the MultiSliderWidgetClass with the specified slider parameters.

        Parameters:
            n_slider (int): (Optional) Overrides n_sliders if provided.
            n_sliders (int): Number of sliders.
            title (str): Window title.
            window_width (int): Width of the window.
            window_height (int): Height of the window.
            x_offset (int): X offset for window placement.
            y_offset (int): Y offset for window placement.
            label_width (int): Width of the label.
            slider_width (int): Width of each slider.
            label_texts (list): List of texts for slider labels.
            slider_mins (list): List of minimum slider values.
            slider_maxs (list): List of maximum slider values.
            slider_vals (list): List of initial slider values.
            resolutions (list): List of resolutions for each slider.
            max_change (float): Maximum allowed change per update.
            max_changes (list): List of maximum changes for each slider.
            fontsize (int): Font size for labels.
            verbose (bool): If True, print status messages.
        """        
        super().__init__()
        if n_slider is not None: n_sliders = n_slider # override 'n_sliders' if 'n_slider' it not None 
        self.n_sliders     = n_sliders
        self.title         = title
        self.window_width  = window_width
        self.window_height = window_height
        self.x_offset      = x_offset
        self.y_offset      = y_offset
        self.label_width   = label_width
        self.slider_width  = slider_width
        if label_texts is None:
            self.label_texts = [f'Slider {i+1}' for i in range(n_sliders)]
        else:
            self.label_texts = label_texts
        self.slider_mins   = slider_mins
        self.slider_maxs   = slider_maxs
        self.slider_vals   = slider_vals
        if resolutions is None:
            self.resolutions   = [0.01]*self.n_sliders
        else:
            self.resolutions   = resolutions
        self.max_change    = max_change
        if max_changes is None:
            self.max_changes   = [self.max_change]*self.n_sliders
        else:
            self.max_changes   = max_changes
        self.fontsize      = fontsize
        self.verbose       = verbose
        self.sliders        = []
        self.labels_widgets = []
        self.slider_values  = self.slider_vals.copy()
        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface for the slider widget.

        Returns:
            None
        """
        # Main layout
        main_layout = QVBoxLayout(self)

        # Make region scrollable 
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Widget
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        for i in range(self.n_sliders):
            # Slider
            slider = QSlider(Qt.Horizontal, self)

            # Resolution
            scale = 1 / self.resolutions[i]

            # Integer range
            int_min = int(self.slider_mins[i] * scale)
            int_max = int(self.slider_maxs[i] * scale)
            int_val = int(self.slider_vals[i] * scale)

            slider.setMinimum(int_min)
            slider.setMaximum(int_max)
            slider.setValue(int_val)
            slider.setSingleStep(1)
            
            slider.sliderMoved.connect(lambda value, idx=i, s=scale: self.value_changed(idx, value, s))

            # Slider label
            label = QLabel(f'{self.label_texts[i]}: {self.slider_vals[i]:.4f}', self)
            label.setFixedWidth(self.label_width)
            label.setStyleSheet(f"font-size: {self.fontsize}px;")

            self.sliders.append(slider)
            self.labels_widgets.append(label)

            scroll_layout.addWidget(label, i, 0)
            scroll_layout.addWidget(slider, i, 1)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)

        # Scrollable area
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self.setWindowTitle(self.title)
        self.setGeometry(self.x_offset, self.y_offset, self.window_width, self.window_height)
        self.show()
        
        # Pause
        time.sleep(1.0)

    def value_changed(self, index, int_value, scale):
        """
        Callback function when a slider's value changes.

        Parameters:
            index (int): Index of the slider.
            int_value (int): New integer value from the slider.
            scale (float): Scaling factor to convert integer value to float.

        Returns:
            None
        """
        # Change integer to float value
        float_value = int_value / (scale+1e-6)
        
        # Change only if difference is less than max_change
        max_change = self.max_changes[index]
        if abs(float_value - self.slider_values[index]) <= max_change: # small changes
            self.slider_values[index] = float_value
            self.labels_widgets[index].setText(f'{self.label_texts[index]}: {float_value:.2f}')
            if self.verbose:
                print(f'Slider {index} Value: {float_value}')
        else:
            self.set_slider_value(slider_idx=index,slider_value=self.slider_values[index])
            if self.verbose:
                print(f'Slider {index} Ignoring jitter (diff: {abs(float_value - self.slider_values[index]):.4f})')
        
    def get_slider_values(self):
        """
        Get the current slider values.

        Returns:
            np.array: Array of current slider values.
        """
        return self.slider_values

    def set_slider_values(self, slider_values):
        """
        Set the values of all sliders.

        Parameters:
            slider_values (list or np.array): New slider values.

        Returns:
            None
        """
        for i, val in enumerate(slider_values):
            scale = 1 / self.resolutions[i]
            int_val = int(val * scale)
            self.sliders[i].setValue(int_val)

    def set_slider_value(self, slider_idx, slider_value):
        """
        Set the value of a specific slider.

        Parameters:
            slider_idx (int): Index of the slider.
            slider_value (float): New value for the slider.

        Returns:
            None
        """
        scale = 1 / self.resolutions[slider_idx]
        int_val = int(slider_value * scale)
        self.sliders[slider_idx].setValue(int_val)

    def close(self):
        """
        Close the slider widget.

        Returns:
            None
        """
        super().close()

class MultiSliderQtClass(QWidget):
    """ 
        Multi sliders using Qt
    """
    def __init__(
            self,
            title         = None,
            name          = 'Sliders',
            n_slider      = 5,
            x_offset      = 0,
            y_offset      = 0,
            window_width  = 500,
            window_height = 300,
            label_width   = 100,
            slider_width  = 400,
            label_texts   = None,
            label_colors  = None,
            slider_mins   = None,
            slider_maxs   = None,
            slider_steps  = 1001,
            slider_vals   = None,
            resolutions   = None,
            max_changes   = None, # not used, to make it compatible with 'MultiSliderClass'
            fontsize      = 10,
            verbose       = False,
        ):
        """
        Initialize the MultiSliderQtClass with specified slider parameters.
        
        Parameters:
            title (str, optional): Window title (overrides name if provided).
            name (str): Identifier for the slider widget.
            n_slider (int): Number of sliders.
            x_offset (int): Horizontal offset of the window.
            y_offset (int): Vertical offset of the window.
            window_width (int): Window width.
            window_height (int): Window height.
            label_width (int): Width of slider labels.
            slider_width (int): Width of slider widgets.
            label_texts (list, optional): Texts for each slider label.
            slider_mins (list, optional): Minimum values for sliders.
            slider_maxs (list, optional): Maximum values for sliders.
            slider_vals (list, optional): Initial slider values.
            resolutions (list, optional): Resolution for each slider.
            max_changes: Not used (for compatibility).
            fontsize (int): Font size for labels.
            verbose (bool): Verbosity flag.
        
        Returns:
            None
        """
        super().__init__()
        if title is not None: name = title
        self.name          = name
        self.n_slider      = n_slider
        self.x_offset      = x_offset
        self.y_offset      = y_offset
        self.window_width  = window_width
        self.window_height = window_height
        self.label_width   = label_width
        self.slider_width  = slider_width
        if label_texts is None:
            label_texts = list(np.arange(0,self.n_slider,1))
        self.label_texts   = label_texts
        self.color_map = {
            'r': 'red',
            'g': 'green',
            'b': 'blue',
            'w': 'white',
            'k': 'black',
        }
        if label_colors is None:
            label_colors = ['w'] * self.n_slider  # default to white
        self.label_colors = label_colors
        if slider_mins is None:
            slider_mins = (0.0*np.ones(self.n_slider)) # no need to be 'list'
        self.slider_mins   = np.array(slider_mins)
        if slider_maxs is None:
            slider_maxs = (100.0*np.ones(self.n_slider))
        self.slider_maxs   = np.array(slider_maxs)
        if slider_vals is None:
            slider_vals = (0.0*np.ones(self.n_slider))
        self.slider_vals   = np.array(slider_vals)
        self.slider_steps  = slider_steps
        if resolutions is None:
            resolutions = (self.slider_maxs-self.slider_mins)/self.slider_steps # default resolution
            resolutions = np.maximum(resolutions, 1e-6)  # set the minimum to 1e-6
        self.resolutions   = np.array(resolutions)
        self.fontsize      = fontsize
        self.verbose       = verbose

        # Buffers
        self.sliders = []
        self.labels  = []

        # Initialize user interface
        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface for the slider widget.
        
        Parameters:
            None
        
        Returns:
            None
        """
        # Layout
        main_layout = QVBoxLayout(self)
        # Scrollable area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        # Initialize sliders
        for s_idx in range(self.n_slider): # for each slider
            # Slider
            slider = QSlider(Qt.Horizontal,self)
            time.sleep(1e-6)
            # QtSlider values can only have integers
            resolution = self.resolutions[s_idx]
            slider_min = self.slider_mins[s_idx]
            slider_max = self.slider_maxs[s_idx]
            slider_val = self.slider_vals[s_idx]
            # Only integers are allowed
            slider.setMinimum(int(0)) # set minimum
            slider.setMaximum( # set maximum
                self.float_to_int(
                    val_float     = slider_max,
                    min_val_float = slider_min,
                    resolution    = resolution,
                )
            )
            slider.setValue( # set value
                self.float_to_int(
                    val_float     = slider_val,
                    min_val_float = slider_min,
                    resolution    = resolution,
                )
            )
            slider.setSingleStep(int(1))
            # Slider value change callback
            slider.valueChanged.connect(lambda value,idx=s_idx:self.cb_value_changed(idx,value))
            # Append slider
            self.sliders.append(slider)
            # Label
            label = QLabel(f'{self.label_texts[s_idx]}:{self.slider_vals[s_idx]:.2f}',self)
            label_color_key = self.label_colors[s_idx]
            label_color = self.color_map.get(label_color_key, label_color_key)
            time.sleep(1e-6)
            label.setFixedWidth(self.label_width)
            label.setStyleSheet(f"color: {label_color}; font-size: {self.fontsize}px;")
            # Append label
            self.labels.append(label)
            # Add layout
            scroll_layout.addWidget(label,s_idx,0)
            scroll_layout.addWidget(slider,s_idx,1)
        # Add layout
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        # Set layout
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        self.setWindowTitle(self.name) # set title
        self.setGeometry(self.x_offset,self.y_offset,self.window_width,self.window_height) # set size
        # Show
        self.show()
        time.sleep(1e-6)
        # (Optional) Resize
        
    def reset(self,n_reset=1):
        """
        Reset the slider display.
        
        Parameters:
            n_reset (int): Number of resets to perform.
        
        Returns:
            None
        """
        for _ in range(n_reset):
            self.resize(self.width(),self.height())
            self.repaint()

    def cb_value_changed(self,idx,value):
        """
        Callback function for slider value change events.
        
        Parameters:
            idx (int): Index of the slider.
            value (int): New slider value.
        
        Returns:
            None
        """
        label = self.labels[idx]
        label_text = self.label_texts[idx]
        value_float = self.int_to_float(
            val_int       = value,
            min_val_float = self.slider_mins[idx],
            resolution    = self.resolutions[idx],
        )
        label.setText(f'{label_text}:{value_float:.2f}')
        if self.verbose:
            print ("slider index:[%d] value:[%d] value_float:[%.2f]"%
                   (idx,value,value_float))

    def float_to_int(self,val_float,min_val_float,resolution):
        """
        Convert a float value to an integer based on resolution.
        
        Parameters:
            val_float (float): Float value.
            min_val_float (float): Minimum float value.
            resolution (float): Resolution factor.
        
        Returns:
            int: Converted integer value.
        """
        return int((val_float-min_val_float)/resolution)

    def int_to_float(self,val_int,min_val_float,resolution):
        """
        Convert an integer value back to a float based on resolution.
        
        Parameters:
            val_int (int): Integer value.
            min_val_float (float): Minimum float value.
            resolution (float): Resolution factor.
        
        Returns:
            float: Converted float value.
        """
        return val_int*resolution+min_val_float

    def get_values(self):
        """
        Retrieve the current slider values as a NumPy array.
        
        Parameters:
            None
        
        Returns:
            np.array: Current slider values.
        """
        values = np.zeros(self.n_slider)
        for s_idx in range(self.n_slider):
            slider = self.sliders[s_idx]
            values[s_idx] = self.int_to_float(
                val_int       = slider.value(),
                min_val_float = self.slider_mins[s_idx],
                resolution    = self.resolutions[s_idx],
            )
        return values
    
    def set_value(self,s_idx,value):
        """
        Set the value of a specific slider.
        
        Parameters:
            s_idx (int): Slider index.
            value (float): New slider value.
        
        Returns:
            None
        """
        slider    = self.sliders[s_idx]
        label     = self.labels[s_idx]
        val_float = value
        val_int   = self.float_to_int(
            val_float     = val_float,
            min_val_float = self.slider_mins[s_idx],
            resolution    = self.resolutions[s_idx],
        )
        slider.setValue(val_int) # set slider value
        label_text = self.label_texts[s_idx]
        label.setText(f'{label_text}:{val_float:.2f}')
    
    def set_values(self,values):
        """
        Set values for all sliders.
        
        Parameters:
            values (list or array): New slider values.
        
        Returns:
            None
        """
        for s_idx in range(self.n_slider):
            self.set_value(s_idx=s_idx,value=values[s_idx])
    
    def get_slider_values(self):
        """
        Get slider values (compatible with MultiSliderClass).
        
        Parameters:
            None
        
        Returns:
            np.array: Slider values.
        """
        return self.get_values()
    
    def set_slider_values(self,values):
        """
        Set slider values (compatible with MultiSliderClass).
        
        Parameters:
            values (list or array): New slider values.
        
        Returns:
            None
        """
        return self.set_values(values)
        
    def close(self):
        """
        Close the slider widget.
        
        Parameters:
            None
        
        Returns:
            None
        """
        super().close()
        
    def update(self):
        """
        Dummy update function for compatibility with MultiSliderClass.
        
        Parameters:
            None
        
        Returns:
            None
        """
        return

class MultiRadioQtClass(QWidget):
    """
    Multi radio buttons using Qt.
    """
    def __init__(
            self, 
            label_texts, 
            option_texts, 
            window_width  = 500, 
            window_height = 300, 
            x_offset      = 0,
            y_offset      = 0,
            title         = "Multi Radio Buttons",
            fontsize      = 10,
            initial_texts = None,
        ):
        """
        Initialize the MultiRadioQtClass with specified parameters.

        Parameters:
            label_texts (list of str): List of label texts for each row.
            option_texts (list of list of str): List of option texts for each row, aligned with label_texts.
            window_width (int): Width of the window.
            window_height (int): Height of the window.
            x_offset (int): Horizontal position of the window on screen.
            y_offset (int): Vertical position of the window on screen.
            title (str): Window title.
            fontsize (int): Font size for labels and buttons.
            initial_texts (list of str or None, optional): Initial label to select per row.
        """
        super().__init__()
        self.label_texts   = label_texts
        self.option_texts  = option_texts
        self.n_row         = len(option_texts)
        self.window_width  = window_width
        self.window_height = window_height
        self.x_offset      = x_offset
        self.y_offset      = y_offset
        self.title         = title
        self.fontsize      = fontsize
        self.initial_texts = initial_texts if initial_texts else [None] * self.n_row

        self.button_groups = []  # QButtonGroup for each row
        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface for the multi-radio button widget.
        """
        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        for row in range(self.n_row):
            label = QLabel(self.label_texts[row])
            label.setStyleSheet(f"font-size: {self.fontsize}px;")
            scroll_layout.addWidget(label, row, 0)

            button_group = QButtonGroup(self)
            self.button_groups.append(button_group)

            for col, text in enumerate(self.option_texts[row]):
                radio = QRadioButton(text)
                radio.setStyleSheet(f"font-size: {self.fontsize}px;")
                button_group.addButton(radio, col)
                scroll_layout.addWidget(radio, row, col + 1)

                if self.initial_texts[row] == text:
                    radio.setChecked(True)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self.setWindowTitle(self.title)
        self.setGeometry(self.x_offset, self.y_offset, self.window_width, self.window_height)
        self.show()

    def get_button_indices(self):
        """
        Get the selected indices of radio buttons for each row.

        Returns:
            list of int: Selected index for each row, or -1 if none is selected.
        """
        selected = []
        for group in self.button_groups:
            checked = group.checkedId()
            selected.append(checked if checked >= 0 else -1)
        return selected

    def get_button_option(self, label_text):
        """
        Get the selected option text for a specific row identified by label text.

        Parameters:
            label_text (str): The label identifying the row.

        Returns:
            str or None: The selected option text, or None if nothing is selected or label not found.
        """
        for row, row_label in enumerate(self.label_texts):
            if row_label == label_text:
                checked_idx = self.button_groups[row].checkedId()
                if checked_idx >= 0:
                    return self.option_texts[row][checked_idx]
                return None
        return None

    def set_button_option(self, label_text, option_text):
        """
        Set a specific radio button as selected in the row with the given label.

        Parameters:
            label_text (str): The label identifying the row.
            option_text (str): The radio button text to be selected.

        Returns:
            bool: True if the option was successfully set, False otherwise.
        """
        for row, row_label in enumerate(self.label_texts):
            if row_label == label_text:
                if option_text in self.option_texts[row]:
                    col = self.option_texts[row].index(option_text)
                    button = self.button_groups[row].button(col)
                    if button:
                        button.setChecked(True)
                        return True
                else:
                    print(f"[Warning] option_text '{option_text}' not found in row '{label_text}'.")
                return False
        print(f"[Warning] label_text '{label_text}' not found.")
        return False

def finite_difference_matrix(n, dt, order):
    """
    Compute a finite difference matrix for numerical differentiation.
    
    Parameters:
        n (int): Number of points.
        dt (float): Time interval between points.
        order (int): Derivative order (1: velocity, 2: acceleration, 3: jerk).
    
    Returns:
        np.array: Finite difference matrix scaled by dt^order.
    """
    # Order
    if order == 1:  # velocity
        coeffs = np.array([-1, 1])
    elif order == 2:  # acceleration
        coeffs = np.array([1, -2, 1])
    elif order == 3:  # jerk
        coeffs = np.array([-1, 3, -3, 1])
    else:
        raise ValueError("Order must be 1, 2, or 3.")

    # Fill-in matrix
    mat = np.zeros((n, n))
    for i in range(n - order):
        for j, c in enumerate(coeffs):
            mat[i, i + j] = c

    # (optional) Handling boundary conditions with backward differences
    if order == 1:  # velocity
        mat[-1, -2:] = np.array([-1, 1])  # backward difference
    elif order == 2:  # acceleration
        mat[-1, -3:] = np.array([1, -2, 1])  # backward difference
        mat[-2, -3:] = np.array([1, -2, 1])  # backward difference
    elif order == 3:  # jerk
        mat[-1, -4:] = np.array([-1, 3, -3, 1])  # backward difference
        mat[-2, -4:] = np.array([-1, 3, -3, 1])  # backward difference
        mat[-3, -4:] = np.array([-1, 3, -3, 1])  # backward difference

    # Return 
    return mat / (dt ** order)

def get_A_vel_acc_jerk(n=100,dt=1e-2):
    """
    Generate finite difference matrices for velocity, acceleration, and jerk.
    
    Parameters:
        n (int): Number of points.
        dt (float): Time interval.
    
    Returns:
        tuple: (A_vel, A_acc, A_jerk) matrices.
    """
    A_vel  = finite_difference_matrix(n,dt,order=1)
    A_acc  = finite_difference_matrix(n,dt,order=2)
    A_jerk = finite_difference_matrix(n,dt,order=3)
    return A_vel,A_acc,A_jerk

def smooth_optm_1d(
        traj,
        dt          = 0.1,
        x_init      = None,
        x_final     = None,
        vel_init    = None,
        vel_final   = None,
        x_lower     = None,
        x_upper     = None,
        vel_limit   = None,
        acc_limit   = None,
        jerk_limit  = None,
        idxs_remain = None,
        vals_remain = None,
        p_norm      = 2,
        verbose     = True,
    ):
    """
    Perform 1-D smoothing of a trajectory using optimization.
    
    Parameters:
        traj (array): Original trajectory.
        dt (float): Time interval.
        x_init (float, optional): Initial position.
        x_final (float, optional): Final position.
        vel_init (float, optional): Initial velocity.
        vel_final (float, optional): Final velocity.
        x_lower (float, optional): Lower position bound.
        x_upper (float, optional): Upper position bound.
        vel_limit (float, optional): Maximum velocity.
        acc_limit (float, optional): Maximum acceleration.
        jerk_limit (float, optional): Maximum jerk.
        idxs_remain (array, optional): Fixed indices.
        vals_remain (array, optional): Values at fixed indices.
        p_norm (int): Norm degree for objective.
        verbose (bool): Verbosity flag.
    
    Returns:
        np.array: Smoothed trajectory.
    """
    import cvxpy as cp
    n = len(traj)
    A_pos = np.eye(n,n)
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=n,dt=dt)
    
    # Objective 
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x-traj,p_norm))
    
    # Equality constraints
    A_list,b_list = [],[]
    if x_init is not None:
        A_list.append(A_pos[0,:])
        b_list.append(x_init)
    if x_final is not None:
        A_list.append(A_pos[-1,:])
        b_list.append(x_final)
    if vel_init is not None:
        A_list.append(A_vel[0,:])
        b_list.append(vel_init)
    if vel_final is not None:
        A_list.append(A_vel[-1,:])
        b_list.append(vel_final)
    if idxs_remain is not None:
        A_list.append(A_pos[idxs_remain,:])
        if vals_remain is not None:
            b_list.append(vals_remain)
        else:
            b_list.append(traj[idxs_remain])

    # Inequality constraints
    C_list,d_list = [],[]
    if x_lower is not None:
        C_list.append(-A_pos)
        d_list.append(-x_lower*np.ones(n))
    if x_upper is not None:
        C_list.append(A_pos)
        d_list.append(x_upper*np.ones(n))
    if vel_limit is not None:
        C_list.append(A_vel)
        C_list.append(-A_vel)
        d_list.append(vel_limit*np.ones(n))
        d_list.append(vel_limit*np.ones(n))
    if acc_limit is not None:
        C_list.append(A_acc)
        C_list.append(-A_acc)
        d_list.append(acc_limit*np.ones(n))
        d_list.append(acc_limit*np.ones(n))
    if jerk_limit is not None:
        C_list.append(A_jerk)
        C_list.append(-A_jerk)
        d_list.append(jerk_limit*np.ones(n))
        d_list.append(jerk_limit*np.ones(n))
    constraints = []
    if A_list:
        A = np.vstack(A_list)
        b = np.hstack(b_list).squeeze()
        constraints.append(A @ x == b) 
    if C_list:
        C = np.vstack(C_list)
        d = np.hstack(d_list).squeeze()
        constraints.append(C @ x <= d)
    
    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    # Return
    traj_smt = x.value

    # Null check
    if traj_smt is None and verbose:
        print ("[smooth_optm_1d] Optimization failed.")
    return traj_smt

def smooth_gaussian_1d(traj,sigma=5.0,mode='nearest',radius=5):
    """
    Smooth a 1-D trajectory using a Gaussian filter.
    
    Parameters:
        traj (array): Input trajectory.
        sigma (float): Gaussian standard deviation.
        mode (str): Filter mode.
        radius (int): Radius for filtering.
    
    Returns:
        np.array: Smoothed trajectory.
    """
    from scipy.ndimage import gaussian_filter1d
    traj_smt = gaussian_filter1d(
        input  = traj,
        sigma  = sigma,
        mode   = 'nearest',
        radius = int(radius),
    )
    return traj_smt
    
def plot_traj_vel_acc_jerk(
        t,
        traj,
        traj_smt = None,
        figsize  = (6,6),
        title    = 'Trajectory',
        ):
    """
    Plot trajectory, velocity, acceleration, and jerk.
    
    Parameters:
        t (array): Time vector.
        traj (array): Original trajectory.
        traj_smt (array, optional): Smoothed trajectory.
        figsize (tuple): Figure size.
        title (str): Plot title.
    
    Returns:
        None
    """
    n  = len(t)
    dt = t[1]-t[0]
    # Compute velocity, acceleration, and jerk
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=n,dt=dt)
    vel  = A_vel @ traj
    acc  = A_acc @ traj
    jerk = A_jerk @ traj
    if traj_smt is not None:
        vel_smt  = A_vel @ traj_smt
        acc_smt  = A_acc @ traj_smt
        jerk_smt = A_jerk @ traj_smt
    # Plot
    plt.figure(figsize=figsize)
    plt.subplot(4, 1, 1)
    plt.plot(t,traj,'.-',ms=1,color='k',lw=1/5,label='Trajectory')
    if traj_smt is not None:
        plt.plot(t,traj_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Trajectory')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 2)
    plt.plot(t,vel,'.-',ms=1,color='k',lw=1/5,label='Velocity')
    if traj_smt is not None:
        plt.plot(t,vel_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Velocity')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 3)
    plt.plot(t,acc,'.-',ms=1,color='k',lw=1/5,label='Acceleration')
    if traj_smt is not None:
        plt.plot(t,acc_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Acceleration')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 4)
    plt.plot(t,jerk,'.-',ms=1,color='k',lw=1/5,label='Jerk')
    if traj_smt is not None:
        plt.plot(t,jerk_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Jerk')
    plt.legend(fontsize=8,loc='upper right')
    plt.suptitle(title,fontsize=10)
    plt.subplots_adjust(hspace=0.2,top=0.95)
    plt.show()

def kernel_se(X1,X2,hyp={'g':1,'l':1,'w':0}):
    """
    Compute the squared exponential (SE) kernel.
    
    Parameters:
        X1 (array): First input set.
        X2 (array): Second input set.
        hyp (dict): Kernel hyperparameters.
    
    Returns:
        np.array: Kernel matrix.
    """
    if len(X1.shape) == 1: 
        X1_use = X1.reshape(-1,1)
    else:
        X1_use = X1
    if len(X2.shape) == 1: 
        X2_use = X2.reshape(-1,1)
    else:
        X2_use = X2
    K = hyp['g']*hyp['g']*np.exp(-cdist(X1_use,X2_use,'sqeuclidean')/(2*hyp['l']*hyp['l'])) 
    if (hyp['w']>0) and (K.shape[0]==K.shape[1]):
        K = K + hyp['w']*np.eye(K.shape[0]) # add diag
    return K

def kernel_levse(X1,X2,L1,L2,hyp={'g':1,'l':1}):
    """
    Compute the leveraged squared exponential (SE) kernel.
    
    Parameters:
        X1 (array): First input set.
        X2 (array): Second input set.
        L1 (array): Leverage values for X1.
        L2 (array): Leverage values for X2.
        hyp (dict): Kernel hyperparameters.
    
    Returns:
        np.array: Leveraged kernel matrix.
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    L = np.cos(np.pi/2.0*cdist(L1,L2,'cityblock'))
    return np.multiply(K,L)

def safe_chol(A,upper=False,max_iter=100,eps=1e-20,verbose=False):
    """
    Perform safe Cholesky decomposition with jitter.
    
    Parameters:
        A (np.array): Matrix to decompose.
        max_iter (int): Maximum iterations.
        eps (float): Initial epsilon value.
        verbose (bool): Verbosity flag.
    
    Returns:
        np.array or None: Cholesky factor or None if failed.
    """
    A_use = A.copy()
    for iter in range(max_iter):
        try:
            L = np.linalg.cholesky(A_use,upper=upper)
            if verbose:
                print ("[safe_chol] Cholesky succeeded. iter:[%d] eps:[%.2e]"%(iter,eps))
            return L 
        except np.linalg.LinAlgError:
            A_use = A_use + eps*np.eye(A.shape[0])
            eps *= 10
    print ("[safe_chol] Cholesky failed. iter:[%d] eps:[%.2e]"%(iter,eps))
    return None

def soft_squash(x,x_min=-1,x_max=+1,margin=0.1):
    """
    Apply soft squashing to constrain array values.
    
    Parameters:
        x (np.array): Input array.
        x_min (float): Minimum value.
        x_max (float): Maximum value.
        margin (float): Margin for squashing.
    
    Returns:
        np.array: Squashed array.
    """
    def th(z,m=0.0):
        # thresholding function 
        return (m)*(np.exp(2/m*z)-1)/(np.exp(2/m*z)+1)
    x_in = np.copy(x)
    idxs_upper = np.where(x_in>(x_max-margin))
    x_in[idxs_upper] = th(x_in[idxs_upper]-(x_max-margin),m=margin) + (x_max-margin)
    idxs_lower = np.where(x_in<(x_min+margin))
    x_in[idxs_lower] = th(x_in[idxs_lower]-(x_min+margin),m=margin) + (x_min+margin)
    return x_in    

def soft_squash_multidim(
        x      = np.random.randn(100,5),
        x_min  = -np.ones(5),
        x_max  = np.ones(5),
        margin = 0.1,
    ):
    """
    Apply soft squashing dimension-wise for multi-dimensional arrays.
    
    Parameters:
        x (np.array): Input array.
        x_min (array): Minimum values per dimension.
        x_max (array): Maximum values per dimension.
        margin (float): Margin for squashing.
    
    Returns:
        np.array: Squashed multi-dimensional array.
    """
    x_squash = np.copy(x)
    dim      = x.shape[1]
    for d_idx in range(dim):
        x_squash[:,d_idx] = soft_squash(
            x=x[:,d_idx],x_min=x_min[d_idx],x_max=x_max[d_idx],margin=margin)
    return x_squash 

def get_idxs_closest_ndarray(ndarray_query,ndarray_domain):
    """
    Get indices of closest elements in ndarray_domain for each query element.
    
    Parameters:
        ndarray_query (array): Query array.
        ndarray_domain (array): Domain array.
    
    Returns:
        list: Indices of closest matches.
    """    
    return [np.argmin(np.abs(ndarray_query-x)) for x in ndarray_domain]

def get_interp_const_vel_traj_nd(
        anchors, # [L x D]
        vel = 1.0,
        HZ  = 100,
        ord = np.inf,
    ):
    """
    Generate a linearly interpolated (with constant velocity) trajectory.
    
    Parameters:
        anchors (array): Anchor points [L x D].
        vel (float): Constant velocity.
        HZ (int): Sampling frequency.
        ord (float): Norm order for distance.
    
    Returns:
        tuple: (times_interp, anchors_interp, times_anchor, idxs_anchor)
    """
    L = anchors.shape[0]
    D = anchors.shape[1]
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev,p_curr = anchors[tick-1,:],anchors[tick,:]
            dists[tick] = np.linalg.norm(p_prev-p_curr,ord=ord)
    times_anchor = np.cumsum(dists/vel) # [L]
    L_interp     = int(times_anchor[-1]*HZ)
    times_interp = np.linspace(0,times_anchor[-1],L_interp) # [L_interp]
    anchors_interp  = np.zeros((L_interp,D)) # [L_interp x D]
    for d_idx in range(D): # for each dim
        anchors_interp[:,d_idx] = np.interp(times_interp,times_anchor,anchors[:,d_idx])
    idxs_anchor = get_idxs_closest_ndarray(times_interp,times_anchor)
    return times_interp,anchors_interp,times_anchor,idxs_anchor

def get_smoothed_equidistant_xy_traj(
        xy_traj_raw, # [L x 2]
        inter_dist = 0.1, # inter distance (m)
        acc_limit  = 10,
    ):
    """
    Obtain a smoothed, equidistant XY trajectory.
    
    Parameters:
        xy_traj_raw (array): Raw XY trajectory [L x 2].
        inter_dist (float): Interpolation distance.
        acc_limit (float): Acceleration limit.
    
    Returns:
        np.array: Smoothed XY trajectory.
    """
    D  = xy_traj_raw.shape[1] # dim=2
    HZ = int(1.0/inter_dist)
    # Interpolate the trajectory
    _,xy_traj_interp,_,_ = get_interp_const_vel_traj_nd(
        anchors = xy_traj_raw,
        vel     = 1.0,
        HZ      = HZ,
    ) # interpolate the trajectory with constant velocity
    # Smooth the trajectory
    xy_traj_smt = np.zeros_like(xy_traj_interp) # [L x 2]
    for d_idx in range(D): # smooth interpolated trajectory
        traj = xy_traj_interp[:,d_idx]
        xy_traj_smt[:,d_idx] = smooth_optm_1d(
            traj,
            dt        = 1/HZ,
            x_init    = traj[0],
            x_final   = traj[-1],
            acc_limit = acc_limit, # smoothing acceleration limit
        )
    # Return
    return xy_traj_smt
    

def interpolate_and_smooth_nd(
        anchors, # List or [N x D]
        HZ             = 50,
        vel_init       = 0.0,
        vel_final      = 0.0,
        x_lowers       = None, # [D]
        x_uppers       = None, # [D]
        vel_limit      = None, # [1]
        acc_limit      = None, # [1]
        jerk_limit     = None, # [1]
        vel_interp_max = np.deg2rad(180),
        vel_interp_min = np.deg2rad(10),
        n_interp       = 10,
        verbose        = False,
    ):
    """
    Interpolate and smooth multi-dimensional anchor points.
    
    Parameters:
        anchors (array or list): Anchor points [N x D].
        HZ (int): Sampling frequency.
        vel_init (float): Initial velocity.
        vel_final (float): Final velocity.
        x_lowers (array, optional): Lower bounds.
        x_uppers (array, optional): Upper bounds.
        vel_limit (float, optional): Velocity limit.
        acc_limit (float, optional): Acceleration limit.
        jerk_limit (float, optional): Jerk limit.
        vel_interp_max (float): Maximum interpolation velocity.
        vel_interp_min (float): Minimum interpolation velocity.
        n_interp (int): Number of interpolation steps.
        verbose (bool): Verbosity flag.
    
    Returns:
        tuple: (times, traj_interp, traj_smt, times_anchor)
    """
    if isinstance(anchors, list):
        # If 'anchors' is given as a list, make it an ndarray
        anchors = np.vstack(anchors)
    
    D = anchors.shape[1]
    vels = np.linspace(start=vel_interp_max,stop=vel_interp_min,num=n_interp)
    for v_idx,vel_interp in enumerate(vels):
        # First, interploate
        times,traj_interp,times_anchor,idxs_anchor = get_interp_const_vel_traj_nd(
            anchors = anchors,
            vel     = vel_interp,
            HZ      = HZ,
        )
        dt = times[1] - times[0]
        # Second, smooth
        traj_smt = np.zeros_like(traj_interp)
        is_success = True
        for d_idx in range(D):
            traj_d = traj_interp[:,d_idx]
            if x_lowers is not None: x_lower_d = x_lowers[d_idx]
            else: x_lower_d = None
            if x_uppers is not None: x_upper_d = x_uppers[d_idx]
            else: x_upper_d = None
            traj_smt_d = smooth_optm_1d(
                traj        = traj_d,
                dt          = dt,
                idxs_remain = idxs_anchor,
                vals_remain = anchors[:,d_idx],
                vel_init    = vel_init,
                vel_final   = vel_final,
                x_lower     = x_lower_d,
                x_upper     = x_upper_d,
                vel_limit   = vel_limit,
                acc_limit   = acc_limit,
                jerk_limit  = jerk_limit,
                p_norm      = 2,
                verbose     = False,
            )
            if traj_smt_d is None:
                is_success = False
                break
            # Append
            traj_smt[:,d_idx] = traj_smt_d

        # Check success
        if is_success:
            if verbose:
                print ("Optimization succeeded. vel_interp:[%.3f]"%(vel_interp))
            return times,traj_interp,traj_smt,times_anchor
        else:
            if verbose:
                print (" v_idx:[%d/%d] vel_interp:[%.2f] failed."%(v_idx,n_interp,vel_interp))
    
    # Optimization failed
    if verbose:
        print ("Optimization failed.")
    return times,traj_interp,traj_smt,times_anchor

def check_vel_acc_jerk_nd(
        times, # [L]
        traj, # [L x D]
        verbose = True,
        factor  = 1.0,
    ):
    """
    Check velocity, acceleration, and jerk of an n-dimensional trajectory.
    
    Parameters:
        times (array): Time vector.
        traj (array): Trajectory [L x D].
        verbose (bool): Verbosity flag.
        factor (float): Scaling factor for display.
    
    Returns:
        tuple: (vel_inits, vel_finals, max_vels, max_accs, max_jerks)
    """
    L,D = traj.shape[0],traj.shape[1]
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=len(times),dt=times[1]-times[0])
    vel_inits,vel_finals,max_vels,max_accs,max_jerks = [],[],[],[],[]
    for d_idx in range(D):
        traj_d = traj[:,d_idx]
        vel = A_vel @ traj_d
        acc = A_acc @ traj_d
        jerk = A_jerk @ traj_d
        vel_inits.append(vel[0])
        vel_finals.append(vel[-1])
        max_vels.append(np.abs(vel).max())
        max_accs.append(np.abs(acc).max())
        max_jerks.append(np.abs(jerk).max())

    # Print
    if verbose:
        print ("Checking velocity, acceleration, and jerk of a L:[%d]xD:[%d] trajectory (factor:[%.2f])."%
               (L,D,factor))
        for d_idx in range(D):
            print (" dim:[%d/%d]: v_init:[%.2e] v_final:[%.2e] v_max:[%.2f] a_max:[%.2f] j_max:[%.2f]"%
                   (d_idx,D,
                    factor*vel_inits[d_idx],factor*vel_finals[d_idx],
                    factor*max_vels[d_idx],factor*max_accs[d_idx],factor*max_jerks[d_idx])
                )
            
    # Return
    return vel_inits,vel_finals,max_vels,max_accs,max_jerks

def animate_chains_slider(
        env,
        secs,
        chains,
        transparent       = True,
        black_sky         = True,
        r_link            = 0.005,
        rgba_link         = (0.05,0.05,0.05,0.9),
        plot_joint        = True,
        plot_joint_axis   = True,
        plot_joint_sphere = False,
        plot_joint_name   = False,
        axis_len_joint    = 0.05,
        axis_width_joint  = 0.005,
        plot_rev_axis     = True,
    ):
    """
    Animate a sequence of chains with a slider interface.
    
    Parameters:
        env: Simulation environment.
        secs (array): Time stamps.
        chains (list): List of chain objects.
        transparent (bool): Use transparent viewer background.
        black_sky (bool): Use black sky background.
        r_link (float): Link radius.
        rgba_link (tuple): RGBA color for links.
        plot_joint (bool): Plot joints.
        plot_joint_axis (bool): Plot joint axes.
        plot_joint_sphere (bool): Plot joint spheres.
        plot_joint_name (bool): Display joint names.
        axis_len_joint (float): Length of joint axes.
        axis_width_joint (float): Width of joint axes.
        plot_rev_axis (bool): Plot reverse axes.
    
    Returns:
        None
    """
    # Reset
    env.reset(step=True)
    
    # Initialize slider
    L = len(secs)
    sliders = MultiSliderClass(
        n_slider      = 2,
        title         = 'Slider Tick',
        window_width  = 900,
        window_height = 100,
        x_offset      = 100,
        y_offset      = 100,
        slider_width  = 600,
        label_texts   = ['tick','mode (0:play,1:slider,2:reverse)'],
        slider_mins   = [0,0],
        slider_maxs   = [L-1,2],
        slider_vals   = [0,1.0],
        resolutions   = [0.1,1.0],
        verbose       = False,
    )
    
    # Loop
    env.init_viewer(
        transparent = transparent,
        black_sky   = black_sky,
    )
    tick,mode = 0,'slider' # 'play' / 'slider'
    while env.is_viewer_alive():
        # Update
        env.increase_tick()
        sliders.update() # update slider
        chain = chains[tick]
        sec = secs[tick]

        # Mode change
        if sliders.get_slider_values()[1] == 0.0: mode = 'play'
        elif sliders.get_slider_values()[1] == 1.0: mode = 'slider'
        elif sliders.get_slider_values()[1] == 2.0: mode = 'reverse'

        # Render
        if env.loop_every(tick_every=20) or (mode=='play') or (mode=='reverse'):
            chain.plot_chain_mujoco(
                env,
                r_link            = r_link,
                rgba_link         = rgba_link,
                plot_joint        = plot_joint,
                plot_joint_axis   = plot_joint_axis,
                plot_joint_sphere = plot_joint_sphere,
                plot_joint_name   = plot_joint_name,
                axis_len_joint    = axis_len_joint,
                axis_width_joint  = axis_width_joint,
                plot_rev_axis     = plot_rev_axis,
            )
            env.plot_T(p=np.array([0,0,0]),R=np.eye(3,3))
            env.plot_text(
                p     = np.array([0,0,1]),
                label = '[%d] tick:[%d] time:[%.2f]sec mode:[%s]'%(env.tick,tick,sec,mode)
            )
            env.render()        

        # Proceed
        if mode == 'play':
            if tick < len(chains)-1: tick = tick + 1
            sliders.set_slider_value(slider_idx=0,slider_value=tick)
        elif mode == 'slider':
            tick = int(sliders.get_slider_values()[0])
        elif mode == 'reverse':
            if tick > 0: tick = tick - 1
            sliders.set_slider_value(slider_idx=0,slider_value=tick)
            
    # Close viewer and slider
    env.close_viewer() 
    sliders.close()
    
def animate_chains_slider_qt(
        env,
        chains,
        secs,
    ):
    """
    Animate a sequence of chains with a slider interface (Qt).
    """
    env.reset()
    env.init_viewer(height=env.monitor_height,transparent=True)
    L = len(chains)
    sliders = MultiSliderQtClass(
        n_slider      = 1,
        window_width  = 500,
        window_height = 100,
        slider_width  = 300,
        label_width   = 150,
        label_texts   = ['Tick'],
        label_colors  = ['w'],
        slider_mins   = np.array([0]),
        slider_maxs   = np.array([L-1]),
        slider_vals   = np.array([0,0,0,0,0,0]),
        verbose       = False,
    )

    # Loop
    while env.is_viewer_alive():
        tick = int(sliders.get_values()[0])
        chains[tick].plot_chain_mujoco(env)
        env.viewer_text_overlay('Tick','%d'%(tick))
        env.viewer_text_overlay('Time','%.2fsec'%(secs[tick]))
        env.plot_joint_axis()
        env.render()

    # Close viewer and slider
    env.close_viewer()
    sliders.close()


def animate_env_qpos_list(
        env,
        secs,
        qpos_list,
        viewer_title      = '',
        plot_contact_info = True,
        transparent       = True,
        black_sky         = True,
    ):
    """
    Animate the environment using a list of configuration positions.
    
    Parameters:
        env: Simulation environment.
        secs (array): Time stamps.
        qpos_list (list): List of qpos configurations.
        viewer_title (str): Viewer window title.
        plot_contact_info (bool): Plot contact information.
        transparent (bool): Use transparent viewer background.
        black_sky (bool): Use black sky background.
    
    Returns:
        None
    """
    # Reset
    env.reset(step=True)
    # Initialize slider
    L = len(secs)
    sliders = MultiSliderClass(
        n_slider      = 2,
        title         = 'Slider Tick',
        window_width  = 900,
        window_height = 100,
        x_offset      = 100,
        y_offset      = 100,
        slider_width  = 600,
        label_texts   = ['tick','mode (0:play,1:slider,2:reverse)'],
        slider_mins   = [0,0],
        slider_maxs   = [L-1,2],
        slider_vals   = [0,1.0],
        resolutions   = [0.1,1.0],
        verbose       = False,
    )
    # Loop
    env.init_viewer(
        transparent = transparent,
        title       = viewer_title,
        black_sky   = black_sky,
    )
    tick,mode = 0,'slider' # 'play' / 'slider'
    while env.is_viewer_alive():
        # Update
        sliders.update() # update slider
        qpos = qpos_list[tick]
        env.forward(q=qpos)
        sec = secs[tick]

        # Mode change
        if sliders.get_slider_values()[1] == 0.0: mode = 'play'
        elif sliders.get_slider_values()[1] == 1.0: mode = 'slider'
        elif sliders.get_slider_values()[1] == 2.0: mode = 'reverse'

        # Render
        if env.loop_every(tick_every=20) or (mode=='play') or (mode=='reverse'):
            env.plot_T(p=np.array([0,0,0]),R=np.eye(3,3))
            env.viewer.add_overlay(loc='bottom left',text1='tick',text2='%d'%(env.tick))
            env.viewer.add_overlay(loc='bottom left',text1='sim time (sec)',text2='%.2f'%(sec))
            env.viewer.add_overlay(loc='bottom left',text1='mode',text2='%s'%(mode))
            if plot_contact_info:
                env.plot_contact_info()
            env.render()

        # Proceed
        if mode == 'play':
            if tick < len(qpos_list)-1: tick = tick + 1
            sliders.set_slider_value(slider_idx=0,slider_value=tick)
            if tick == (L-1): mode = 'slider'
        elif mode == 'slider':
            tick = int(sliders.get_slider_values()[0])
        elif mode == 'reverse':
            if tick > 0: tick = tick - 1
            sliders.set_slider_value(slider_idx=0,slider_value=tick)
            
    # Close viewer and slider
    env.close_viewer() 
    sliders.close()
        
def np_uv(vec):
    """
    Compute the unit vector of the given vector.
    
    Parameters:
        vec (array): Input vector.
    
    Returns:
        np.array: Unit vector.
    """
    x = np.array(vec)
    len = np.linalg.norm(x)
    if len <= 1e-6:
        return np.array([0,0,1]) # if zero vector, return default unit vector
    else:
        return x/len    
    
def uv_T_joi(T_joi,joi_fr,joi_to):
    """
    Compute the unit vector between two JOI poses.
    
    Parameters:
        T_joi (array): Transformation matrices.
        joi_fr: Starting index.
        joi_to: Ending index.
    
    Returns:
        np.array: Unit vector from joi_fr to joi_to.
    """
    return np_uv(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))

def len_T_joi(T_joi,joi_fr,joi_to):
    """
    Compute the Euclidean distance between two JOI poses.
    
    Parameters:
        T_joi (array): Transformation matrices.
        joi_fr: Starting index.
        joi_to: Ending index.
    
    Returns:
        float: Distance between the poses.
    """
    return np.linalg.norm(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))

def get_consecutive_subarrays(array,min_element=1):
    """
    Extract consecutive subarrays from an array.
    
    Parameters:
        array (array): Input array.
        min_element (int): Minimum number of elements per subarray.
    
    Returns:
        list: List of consecutive subarrays.
    """
    split_points = np.where(np.diff(array) != 1)[0] + 1
    subarrays = np.split(array,split_points)    
    return [subarray for subarray in subarrays if len(subarray) >= min_element]

def load_image(image_path):
    """
    Load an image from a file and return it as a NumPy array.
    
    Parameters:
        image_path (str): Path to the image file.
    
    Returns:
        np.array: Loaded image in uint8 format.
    """
    return np.array(Image.open(image_path))

def imshows(img_list,title_list,figsize=(8,2),fontsize=8):
    """
    Display multiple images in a row with titles.
    
    Parameters:
        img_list (list): List of images.
        title_list (list): List of titles.
        figsize (tuple): Figure size.
        fontsize (int): Font size for titles.
    
    Returns:
        None
    """
    n_img = len(img_list)
    plt.figure(figsize=(8,2))
    for img_idx in range(n_img):
        img   = img_list[img_idx]
        title = title_list[img_idx]
        plt.subplot(1,n_img,img_idx+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title,fontsize=fontsize)
    plt.show()
    
def depth_to_gray_img(depth,max_val=10.0):
    """
    Convert a 1-channel float depth image to a 3-channel uint8 gray image.
    
    Parameters:
        depth (array): Input depth image.
        max_val (float): Maximum value for normalization.
    
    Returns:
        np.array: 3-channel gray image.
    """
    depth_clip = np.clip(depth,a_min=0.0,a_max=max_val) # float-type
    # Normalize depth image
    img = np.tile(255*depth_clip[:,:,np.newaxis]/depth_clip.max(),(1,1,3)).astype(np.uint8) # unit8-type
    return img

def get_monitor_size():
    """
    Retrieve the current monitor size.
    
    Parameters:
        None
    
    Returns:
        tuple: (width, height). Falls back to (1920, 1080) when pyautogui is not
        installed or no display is available (e.g. headless machines).
    """
    try:
        import pyautogui
        w,h = pyautogui.size()
    except Exception:
        w,h = 1920,1080
    return w,h
    
def get_xml_string_from_path(xml_path):
    """
    Parse an XML file and return its content as a string.
    
    Parameters:
        xml_path (str): Path to the XML file.
    
    Returns:
        str: XML content.
    """    
    # Parse the XML file
    tree = ET.parse(xml_path)
    
    # Get the root element of the XML
    root = tree.getroot()
    
    # Convert the ElementTree object to a string
    xml_string = ET.tostring(root, encoding='unicode', method='xml')
    
    return xml_string

def printmd(string):
    """
    Display a markdown formatted string.
    
    Parameters:
        string (str): Markdown string.
    
    Returns:
        None
    """    
    display(Markdown(string)) 
    
def prettify(elem):
    """
    Return a pretty-printed XML string for the given element.
    
    Parameters:
        elem: XML element.
    
    Returns:
        str: Pretty XML string.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    # Remove empty lines
    lines = [line for line in pretty_xml.splitlines() if line.strip()]
    return "\n".join(lines)

class TicTocClass(object):
    """
    Utility class for measuring elapsed time.
    Usage:
        tt = TicTocClass()
        tt.tic()
        ~~
        tt.toc()
    """    
    def __init__(self,name='tictoc',print_every=1):
        """
        Initialize the TicTocClass.
        
        Parameters:
            name (str): Name identifier.
            print_every (int): Frequency for printing elapsed time.
        
        Returns:
            None
        """
        self.name         = name
        self.time_start   = time.time()
        self.time_end     = time.time()
        self.print_every  = print_every
        self.time_elapsed = 0.0
        self.cnt          = 0 

    def tic(self):
        """
        Record the current time as the start time.
        
        Parameters:
            None
        
        Returns:
            None
        """
        self.time_start = time.time()

    def toc(self,str=None,cnt=None,print_every=None,verbose=False):
        """
        Compute and print the elapsed time since the last tic.
        
        Parameters:
            str (str, optional): Custom label.
            cnt (int, optional): Current count for periodic printing.
            print_every (int, optional): Print frequency.
            verbose (bool): Verbosity flag.
        
        Returns:
            float: Elapsed time.
        """
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start
        if print_every is not None: 
            self.print_every = print_every
        if verbose:
            if self.time_elapsed <1.0:
                time_show = self.time_elapsed*1000.0
                time_unit = 'ms'
            elif self.time_elapsed <60.0:
                time_show = self.time_elapsed
                time_unit = 's'
            else:
                time_show = self.time_elapsed/60.0
                time_unit = 'min'
            if cnt is not None: self.cnt = cnt
            if (self.cnt % self.print_every) == 0:
                if str is None:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (self.name,time_show,time_unit))
                else:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (str,time_show,time_unit))
        self.cnt = self.cnt + 1
        # Return
        return self.time_elapsed
    
def sleep(sec):
    """
    Pause execution for a specified number of seconds.
    
    Parameters:
        sec (float): Number of seconds to sleep.
    
    Returns:
        None
    """
    time.sleep(sec)
    
class OccupancyGridMapClass:
    """
    Occupancy grid map for environment representation.
    """
    def __init__(
            self,
            x_range, 
            y_range, 
            resolution,
            verbose = False,
        ):
        """
        Initialize the occupancy grid map.
        
        Parameters:
            x_range (tuple): X-axis range.
            y_range (tuple): Y-axis range.
            resolution (float): Grid resolution.
            verbose (bool): Verbosity flag.
        
        Returns:
            None
        """
        self.x_range    = x_range
        self.y_range    = y_range
        self.resolution = resolution
        self.verbose    = verbose
        self.width      = int((x_range[1] - x_range[0]) / resolution)
        self.height     = int((y_range[1] - y_range[0]) / resolution)
        self.extent     = [self.y_range[0], self.y_range[1], self.x_range[1], self.x_range[0]]
        # Reset
        self.reset()
        self.reset_local_grid()
        # Print
        if self.verbose:
            print ("grid:[%s]"%(self.grid.shape,))
        
    def reset(self):
        """
        Reset the occupancy grid to all zeros.
        
        Parameters:
            None
        
        Returns:
            None
        """
        self.grid = np.zeros((self.height,self.width), dtype=np.int8)
        
    def reset_local_grid(self):
        """
        Reset the local occupancy grid.
        
        Parameters:
            None
        
        Returns:
            None
        """
        self.local_grid = np.zeros((self.height,self.width), dtype=np.int8)
    
    def xy_to_grid(self,x,y):
        """
        Convert (x,y) coordinates to grid indices.
        
        Parameters:
            x (float): X-coordinate.
            y (float): Y-coordinate.
        
        Returns:
            tuple: (i, j) grid indices.
        """
        i = int((y - self.y_range[0]) / self.resolution)
        j = int((x - self.x_range[0]) / self.resolution)
        return i,j

    def grid_to_xy(self,i,j):
        """
        Convert grid indices to (x,y) coordinates.
        
        Parameters:
            i (int): Row index.
            j (int): Column index.
        
        Returns:
            tuple: (x, y) coordinates.
        """
        y = i * self.resolution + self.y_range[0]
        x = j * self.resolution + self.x_range[0]
        return x,y
    
    def batch_xy_to_grid(self,xy_list):
        """
        Batch convert (x,y) pairs to grid indices.
        
        Parameters:
            xy_list (array): List or array of (x,y) pairs.
        
        Returns:
            tuple: Arrays of grid indices.
        """
        xy_array = np.asarray(xy_list)
        x,y = xy_array[..., 0], xy_array[..., 1] # [L] or [N x L]
        i = ((y - self.y_range[0]) / self.resolution).astype(int)
        j = ((x - self.x_range[0]) / self.resolution).astype(int)
        return i,j
    
    def check_point_occupancy(self,x,y,use_margin=True,use_local_grid=False):
        """
        Check if a point is occupied in the grid.
        
        Parameters:
            x (float): X-coordinate.
            y (float): Y-coordinate.
            use_margin (bool): Whether to use margin.
            use_local_grid (bool): Whether to check the local grid.
        
        Returns:
            int: Occupancy value.
        """
        i,j = self.xy_to_grid(x=x,y=y)
        if use_margin:
            occupied = self.grid_with_margin[i,j]
        else:
            occupied = self.grid[i,j]
        if use_local_grid: # (optional) use local grid
            if use_margin: # margin-added local grid
                occupied = max(occupied,self.local_grid_with_margin[i,j])
            else: # local grid without margin
                occupied = max(occupied,self.local_grid[i,j])
        return occupied
    
    def batch_check_point_occupancy(self,xy_list,use_margin=True,use_local_grid=False):
        """
        Batch check occupancy for multiple points.
        
        Parameters:
            xy_list (array): Array of (x,y) pairs.
            use_margin (bool): Whether to use margin.
            use_local_grid (bool): Whether to check the local grid.
        
        Returns:
            np.array: Occupancy values.
        """
        xy_array = np.asarray(xy_list)
        shape = xy_array.shape[:-1]  # [L] or [N,L]
        i,j = self.batch_xy_to_grid(xy_list=xy_array)
        
        i = np.clip(i, 0, self.grid.shape[0] - 1)  # Limit row indices
        j = np.clip(j, 0, self.grid.shape[1] - 1)  # Limit column indices
    
        if use_margin:
            occupied = self.grid_with_margin[i,j]
        else:
            occupied = self.grid[i,j]
        if use_local_grid: # (optional) use local grid
            if use_margin: # margin-added local grid
                occupied = np.maximum(occupied,self.local_grid_with_margin[i,j])
            else: # local grid without margin
                occupied = np.maximum(occupied,self.local_grid[i,j])
        # Return
        return occupied.reshape(shape) # [L] or [N x L]
    
    def sample_feasible_xy(self,reduce_rate=None,use_margin=True):
        """
        Sample a feasible (x,y) point from the grid.
        
        Parameters:
            reduce_rate (float, optional): Reduction rate for sampling.
            use_margin (bool): Whether to use margin.
        
        Returns:
            np.array: Sampled (x,y) point.
        """
        while True:
            x_min,x_max = self.x_range[0],self.x_range[1]
            y_min,y_max = self.y_range[0],self.y_range[1]
            x_span,y_span = x_max-x_min,y_max-y_min
            if reduce_rate is not None: # reduce_rate=1.0: no reduce / 0.0: no span
                x_min = x_min + 0.5*(1-reduce_rate)*x_span
                y_min = y_min + 0.5*(1-reduce_rate)*y_span
                x_span = reduce_rate*x_span
                y_span = reduce_rate*y_span
            x_rand = x_min + x_span*np.random.rand()
            y_rand = y_min + y_span*np.random.rand()
            if not self.check_point_occupancy(x=x_rand,y=y_rand,use_margin=use_margin):
                break
        # Return
        return np.array([x_rand,y_rand])
    
    def check_line_occupancy(self, x_fr, y_fr, x_to, y_to, step_size=0.1, use_margin=True):
        """
        Check if the line between two points is free of obstacles.
        
        Parameters:
            x_fr (float): Starting x-coordinate.
            y_fr (float): Starting y-coordinate.
            x_to (float): Ending x-coordinate.
            y_to (float): Ending y-coordinate.
            step_size (float): Step size for checking.
            use_margin (bool): Whether to use margin.
        
        Returns:
            int: 1 if occupied, 0 if free.
        """
        distance  = np.hypot(x_to-x_fr,y_to-y_fr)
        num_steps = int(distance/step_size)
        
        # Generate points along the line
        x_values = np.linspace(x_fr,x_to,num_steps)
        y_values = np.linspace(y_fr,y_to,num_steps)
        
        # Check each point's occupancy
        for x, y in zip(x_values,y_values):
            if self.check_point_occupancy(x,y,use_margin=use_margin):
                return 1 # occupied
        return 0 # free

    def update_grid(self,xy_scan):
        """
        Update the global occupancy grid using scan data.
        
        Parameters:
            xy_scan (list): List of (x,y) scan points.
        
        Returns:
            None
        """
        for (x,y) in xy_scan:
            i, j = self.xy_to_grid(x,y)
            if 0 <= i < self.height and 0 <= j < self.width:
                self.grid[i, j] = 1 
                
    def update_local_grid(self,xy_scan,reset_first=True,add_margin=False):
        """
        Update the local occupancy grid with scan data.
        
        Parameters:
            xy_scan (list): List of (x,y) points.
            reset_first (bool): Whether to reset the local grid first.
            add_margin (bool): Whether to add margin.
        
        Returns:
            None
        """
        if reset_first:
            self.reset_local_grid()
        for (x,y) in xy_scan:
            i, j = self.xy_to_grid(x,y)
            if 0 <= i < self.height and 0 <= j < self.width:
                self.local_grid[i, j] = 1 
                
        # Add margin
        if add_margin:
            self.add_margin_to_local_grid(margin=self.safety_margin)
                
    def add_margin(self,margin):
        """
        Add margin around occupied cells in the global grid.
        
        Parameters:
            margin (float): Margin distance.
        
        Returns:
            None
        """
        # Set mask
        margin_cells = int(margin / self.resolution)
        mask = np.zeros((2 * margin_cells + 1, 2 * margin_cells + 1), dtype=bool)
        y, x = np.ogrid[-margin_cells:margin_cells+1, -margin_cells:margin_cells+1]
        mask[y**2 + x**2 <= margin_cells**2] = True
        # Add mask to grid
        self.grid_with_margin = binary_dilation(self.grid,structure=mask).astype(np.int8)
        
    def add_margin_to_local_grid(self,margin):
        """
        Add margin around occupied cells in the local grid.
        
        Parameters:
            margin (float): Margin distance.
        
        Returns:
            None
        """
        # Set mask
        margin_cells = int(margin / self.resolution)
        mask = np.zeros((2 * margin_cells + 1, 2 * margin_cells + 1), dtype=bool)
        y, x = np.ogrid[-margin_cells:margin_cells+1, -margin_cells:margin_cells+1]
        mask[y**2 + x**2 <= margin_cells**2] = True
        # Add mask to grid
        self.local_grid_with_margin = binary_dilation(self.local_grid,structure=mask).astype(np.int8)

    def plot(
            self,
            grid      = None,
            figsize   = (6,4),
            title_str = 'Occupancy Grid Map',
            no_show   = False,
            return_ax = False,
        ):
        """
        Plot the occupancy grid map.
        
        Parameters:
            grid (array, optional): Grid to plot.
            figsize (tuple): Figure size.
            title_str (str): Plot title.
            no_show (bool): Do not display immediately.
            return_ax (bool): Return axis object.
        
        Returns:
            None
        """
        if grid is None: grid = self.grid
        plt.figure(figsize=figsize)
        self.ax = plt.axes()
        plt.imshow(grid.T,cmap="gray", origin="upper",extent=self.extent) # transpose the map (swap X and Y axis)
        plt.grid(color='w', linestyle='-', linewidth=0.2)
        plt.gca().set_xticks(np.arange(self.y_range[0],self.y_range[1],self.resolution),minor=True)
        plt.gca().set_yticks(np.arange(self.x_range[0],self.x_range[1],self.resolution),minor=True)
        plt.gca().grid(which='minor',color='w', linestyle='-',linewidth=0.1)
        plt.xlabel("Y (m)",fontsize=8);plt.ylabel("X (m)",fontsize=8);plt.title(title_str,fontsize=10)
        if not no_show:
            plt.show()
            
    def plot_point(
            self,
            p        = np.array([0,0,]),
            marker   = 'o',
            ms       = 10,
            mec      = 'r',
            mew      = 2,
            mfc      = 'none',
            label    = None,
            fs       = 8,
            p_offset = 0.2,
            no_show  = True,
        ):
        """
        Plot a single point on the grid.
        
        Parameters:
            p (np.array): Point coordinates.
            marker (str): Marker style.
            ms (int): Marker size.
            mec (str): Marker edge color.
            mew (int): Marker edge width.
            mfc (str): Marker face color.
            label (str, optional): Label text.
            fs (int): Font size.
            p_offset (float): Label offset.
            no_show (bool): Do not display immediately.
        
        Returns:
            None
        """
        plt.plot(p[1],p[0],marker=marker,ms=ms,mec=mec,mew=mew,mfc=mfc)
        
        if label is not None:
            plt.text(
                p_offset+p[1],p[0],label,va='center',fontsize=fs,
                bbox=dict(fc='white',alpha=0.5,ec='k')
                )
        
        if not no_show:
            plt.show()
            
    def plot_points(
            self,
            ps       = np.zeros((10,2)),
            marker   = 'o',
            ms       = 3,
            mfc      = 'none',
            mec      = 'k',
            mew      = 1/2,
            ls       = 'none',
            lw       = 0.5,
            color    = 'b',
            no_show  = True,
        ):
        """
        Plot multiple points on the grid.
        
        Parameters:
            ps (array): Array of points.
            marker (str): Marker style.
            ms (int): Marker size.
            mfc (str): Marker face color.
            mec (str): Marker edge color.
            mew (float): Marker edge width.
            ls (str): Line style.
            lw (float): Line width.
            color (str): Color for points.
            no_show (bool): Do not display immediately.
        
        Returns:
            None
        """
        plt.plot(ps[:,1],ps[:,0],
                 marker=marker,ms=ms,mec=mec,mew=mew,mfc=mfc,
                 ls=ls,lw=lw,color=color) # swap x and y
        if not no_show:
            plt.show()
            
    def plot_rrt_results(
            self,
            rrt,
            figsize    = (8,6),
            title_str  = 'RRT Results',
            edge_color = (1,1,0,1),
            edge_lw    = 1/4,
            node_color = (1,1,0,0.5),
            node_ec    = (0,0,0,1),
            node_ms    = 3,
            path_color = (0,0,1,1),
            path_lw    = 1,
            path_ms    = 3,
            save_png   = False,
            png_path   = '',
            verbose    = True,
        ):
        """
        Plot RRT results on the occupancy grid map.
        
        Parameters:
            rrt: RRT object with nodes and edges.
            figsize (tuple): Figure size.
            title_str (str): Plot title.
            edge_color (tuple): Edge color.
            edge_lw (float): Edge line width.
            node_color (tuple): Node color.
            node_ec (tuple): Node edge color.
            node_ms (int): Node marker size.
            path_color (tuple): Path color.
            path_lw (float): Path line width.
            path_ms (int): Path marker size.
            save_png (bool): Save plot as PNG.
            png_path (str): File path for PNG.
            verbose (bool): Verbosity flag.
        
        Returns:
            None
        """
        # Plot occupancy grid map
        self.plot(
            grid      = self.grid_with_margin,
            figsize   = figsize,
            title_str = title_str,
            no_show   = True,
            return_ax = True,
        )
        
        # Get RRT nodes
        p_nodes = np.array([rrt.get_node_point(node) for node in rrt.get_nodes()]) # [N x 2]
        p_nodes_swap = p_nodes[:,[1,0]] # X-Y swapped
        
        # Plot RRT edges
        edgelist = list(rrt.get_edges())
        edge_pos = np.asarray(
            [(p_nodes_swap[e[0]],p_nodes_swap[e[1]]) for e in edgelist]) # [M x 2 x 2]
        edge_collection = mpl.collections.LineCollection(
            edge_pos,colors=edge_color[:3],linewidths=edge_lw,alpha=edge_color[3])
        self.ax.add_collection(edge_collection)
        
        # Plot RRT nodes
        self.plot_points(ps=p_nodes,mfc=node_color,mec=node_ec,ms=node_ms)
        
        # Path to goal
        path_to_goal,node_list = rrt.get_path_to_goal()
        if path_to_goal is not None:
            plt.plot(path_to_goal[:,1],path_to_goal[:,0],'o-',
                    color=path_color,lw=path_lw,mec=path_color,ms=path_ms,mfc='none')
        
        # Plot root and goal
        self.plot_point(p=rrt.point_root,mec='r',label='Start')
        self.plot_point(p=rrt.point_goal,mec='b',label='Final')
        
        # Show
        if save_png:
            if verbose:
                print ("[%s] saved."%(png_path))
            
            directory = os.path.dirname(png_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
                if verbose:
                    print ("[%s] generated."%(directory))
                
            plt.savefig(png_path, bbox_inches='tight')
            plt.close()
        else:
            # Show
            plt.show()
            
    def get_rgb_array_from_grid(
            self,
            grid       = None,
            grid_local = None,
            figsize    = (8,6),
            xy         = None,
            marker     = 'o',
            ms         = 10,
            mec        = 'r',
            mew        = 2,
            mfc        = 'none',
            label      = None,
            fs         = 8,
            p_offset   = 0.2,
            no_show    = True,
        ):
        """
        Generate an RGB image array from the occupancy grid.
        
        Parameters:
            grid (array, optional): Global grid.
            grid_local (array, optional): Local grid.
            figsize (tuple): Figure size.
            xy (np.array, optional): Point to overlay.
            marker (str): Marker style.
            ms (int): Marker size.
            mec (str): Marker edge color.
            mew (int): Marker edge width.
            mfc (str): Marker face color.
            label (str, optional): Label text.
            fs (int): Font size.
            p_offset (float): Label offset.
            no_show (bool): Do not display immediately.
        
        Returns:
            np.array: RGB image array.
        """
        fig,ax = plt.subplots(figsize=figsize)
        
        # Initialize 'color_array'. Note that both 'grid' and 'grid_local' are transposed
        color_array = np.zeros((grid.shape[1], grid.shape[0], 3), dtype=np.uint8)
        color_array[grid.T == 1] = [255,255,255] # white
        if grid_local is not None:
            color_array[(grid_local.T==1) & (grid.T==0)] = [0,255,0] # only local occpupied: green
            color_array[(grid_local.T==1) & (grid.T==1)] = [255,255,0] # both local and global occupied: yellow
        
        # Show image
        ax.imshow(color_array, origin='upper', extent=self.extent)
            
        # Plot point
        if xy is not None:
            self.plot_point(
                p        = xy,
                marker   = marker,
                ms       = ms,
                mec      = mec,
                mew      = mew,
                mfc      = mfc,
                label    = label,
                fs       = fs,
                p_offset = p_offset,
                no_show  = no_show,
            )
        plt.axis('off')  # remove axis
        # Buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        # RGBA array
        rgba_array = np.frombuffer(buf.getvalue(),dtype=np.uint8)
        rgba_array = plt.imread(buf, format='png')
        rgb_array  = rgba_array[..., :3]  # remove alpha channel
        rgb_array  = 255*rgb_array.astype(np.int32)
        # Close
        plt.close(fig)
        return rgb_array

def get_xy_meshgrid(x_range,y_range,res):
    """
    Generate a meshgrid of (x,y) pairs given ranges and resolution.
    
    Parameters:
        x_range (tuple): X-axis range.
        y_range (tuple): Y-axis range.
        res (float): Resolution.
    
    Returns:
        np.array: Array of (x,y) pairs.
    """
    x_values    = np.arange(x_range[0], x_range[1] + res, res)
    y_values    = np.arange(y_range[0], y_range[1] + res, res)
    xx, yy      = np.meshgrid(x_values, y_values)
    xy_meshgrid = np.column_stack([xx.ravel(), yy.ravel()])
    return xy_meshgrid

def rot_mtx(deg):
    """
    Return a 2x2 rotation matrix for the given angle in degrees.
    
    Parameters:
        deg (float): Angle in degrees.
    
    Returns:
        np.array: 2x2 rotation matrix.
    """
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def is_point_in_polygon(point,polygon):
    """
    Check if a point is inside a polygon.
    
    Parameters:
        point (np.array or Point): Point to check.
        polygon: Polygon object.
    
    Returns:
        bool: True if inside, False otherwise.
    """
    import shapely as sp
    from shapely import Point
    if isinstance(point,np.ndarray):
        point_check = Point(point)
    else:
        point_check = point
    return sp.contains(polygon,point_check)

def is_point_feasible(point,obs_list):
    """
    Determine if a point is feasible with respect to obstacles.
    
    Parameters:
        point (np.array): Point coordinates.
        obs_list: List of obstacles.
    
    Returns:
        bool: True if feasible, False otherwise.
    """
    result = is_point_in_polygon(point,obs_list) # is the point inside each obstacle?
    if sum(result) == 0:
        return True
    else:
        return False                    
    
def is_point_to_point_connectable(point1,point2,obs_list):
    """
    Check if a straight line between two points is free of obstacles.
    
    Parameters:
        point1 (np.array): Start point.
        point2 (np.array): End point.
        obs_list: List of obstacles.
    
    Returns:
        bool: True if connectable, False otherwise.
    """
    import shapely as sp
    from shapely import LineString
    result = sp.intersects(LineString([point1,point2]),obs_list)
    if sum(result) == 0:
        return True
    else:
        return False
    
def get_ogm_from_env(
        env,
        rf_body_name    = 'body_lidar',
        rf_sensor_names = [],
        x_range         = (-5,5),
        y_range         = (-5,5),
        scan_resolution = 1.0,
        grid_resolution = 0.05,
        safety_margin   = 0.2,
        plot            = True,
        figsize         = (5,4),
        verbose         = True,
    ):
    """
    Generate an occupancy grid map from environment sensor data.
    
    Parameters:
        env: Simulation environment.
        rf_body_name (str): Reference body name.
        rf_sensor_names (list): Sensor names.
        x_range (tuple): X-axis range.
        y_range (tuple): Y-axis range.
        scan_resolution (float): Scan resolution.
        grid_resolution (float): Grid resolution.
        safety_margin (float): Safety margin.
        plot (bool): Whether to plot the grid.
        figsize (tuple): Plot figure size.
        verbose (bool): Verbosity flag.
    
    Returns:
        OccupancyGridMapClass: Occupancy grid map object.
    """
    # Scan with changing the sensor
    xy_meshgrid = get_xy_meshgrid(x_range=x_range,y_range=y_range,res=scan_resolution)
    xy_data_list = []
    for xy in xy_meshgrid:
        env.set_p_body(body_name=rf_body_name,p=np.append(xy,0)) # move lidar
        p_rf_obs_list = env.get_p_rf_obs_list(sensor_names=rf_sensor_names) # scan
        # Append range finder results 
        if len(p_rf_obs_list) > 0: # if list is not empty
            xy_data = np.array(p_rf_obs_list)[:,:2]
            xy_data_list.append(xy_data)
    # Set occupancy grid map
    ogm = OccupancyGridMapClass(
        x_range    = x_range,
        y_range    = y_range,
        resolution = grid_resolution,
    )
    if verbose:
        print ("Occuapncy grid map instantiated from [%s] env"%(env.name))
        print ("x_range:[%.1f]m~[%.1f]m y_range:[%.1f]m~[%.1f]m"%
               (x_range[0],x_range[1],y_range[0],y_range[1]))
        print ("resolution:[%.2f]m"%(grid_resolution))
        print ("grid.shape:%s"%(ogm.grid.shape,))
    ogm.safety_margin = safety_margin # safety margin
    for xy_data in xy_data_list:
        ogm.update_grid(xy_scan=xy_data) # update grid
    ogm.add_margin(margin=safety_margin) # add margin
    # Plot
    if plot:
        ogm.plot(grid=ogm.grid,figsize=figsize,title_str='Occupancy Grid Map')
        ogm.plot(grid=ogm.grid_with_margin,figsize=figsize,title_str='Occupancy Grid Map with Margin')
    return ogm

def get_wheel_vels(
        dir_vel, # directional velocity of the base (m/s)
        ang_vel, # angular velocity of the base (rad/s)
        w_radius, # wheel radius (m)
        w2w_dist, # wheel to wheel distance (m)
    ):
    """
    Compute left and right wheel velocities for a differential wheeled robot.
    
    Parameters:
        dir_vel (float): Base directional velocity.
        ang_vel (float): Base angular velocity.
        w_radius (float): Wheel radius.
        w2w_dist (float): Wheel-to-wheel distance.
    
    Returns:
        tuple: (left_vel, right_vel)
    """
    left_vel  = (dir_vel - (ang_vel*w2w_dist/2)) / w_radius
    right_vel = (dir_vel + (ang_vel*w2w_dist/2)) / w_radius
    return left_vel,right_vel

def noramlize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi].
    
    Parameters:
        angle (float): Input angle in radians.
    
    Returns:
        float: Normalized angle.
    """
    while angle > np.pi:
        angle = angle - 2*np.pi
    while angle < -np.pi:
        angle = angle + 2*np.pi
    return angle

def compute_xy_heading(p,R):
    """
    Compute (x,y) coordinates and heading from position and rotation.
    
    Parameters:
        p (np.array): Position vector.
        R (np.array): Rotation matrix.
    
    Returns:
        tuple: (xy, heading)
    """
    xy = p[:2]
    heading = np.arctan2(R[1,0],R[0,0])
    return xy,heading

def compute_lookahead_traj(xy,heading,dir_vel,ang_vel,dt,T):
    """
    Compute a lookahead trajectory based on current state and velocities.
    
    Parameters:
        xy (np.array): Current (x,y) position.
        heading (float): Current heading angle.
        dir_vel (float): Directional velocity.
        ang_vel (float): Angular velocity.
        dt (float): Time step.
        T (float): Total lookahead time.
    
    Returns:
        tuple: (secs, xy_traj, heading_traj)
    """
    L = int(T/dt)
    xy_traj,heading_traj,secs = np.zeros((L,2)),np.zeros(L),np.zeros(L)
    # Append current xy and heading
    xy_traj[0,:]    = xy
    heading_traj[0] = heading
    secs[0]         = 0.0
    # Loop
    xy_curr = xy.copy()
    heading_curr = heading
    for tick in range(L-1):
        # Update
        c,s = np.cos(heading_curr),np.sin(heading_curr)
        xy_curr = xy_curr + dir_vel*dt*np.array([c,s])
        heading_curr = heading_curr + ang_vel*dt 
        # Append
        xy_traj[tick+1,:]    = xy_curr.copy()
        heading_traj[tick+1] = heading_curr
        secs[tick+1]         = tick*dt
    # Return
    return secs,xy_traj,heading_traj

def compute_dir_ang_vels(sec,p,R):
    """
    Compute directional and angular velocities from time, position, and orientation.
    
    Parameters:
        sec (float): Current time.
        p (np.array): Current position.
        R (np.array): Current rotation matrix.
    
    Returns:
        tuple: (dir_vel, ang_vel)
    """
    # First call flag
    if not hasattr(compute_dir_ang_vels,'first_flag'):
        compute_dir_ang_vels.first_flag = True
    else:
        compute_dir_ang_vels.first_flag = False
    
    if compute_dir_ang_vels.first_flag:
        dir_vel = 0.0
        ang_vel = 0.0
    else:
        sec_prev = compute_dir_ang_vels.sec_prev
        p_prev   = compute_dir_ang_vels.p_prev.copy()
        R_prev   = compute_dir_ang_vels.R_prev.copy()
        delta_t  = sec - sec_prev
        delta_x  = p[0] - p_prev[0]
        delta_y  = p[1] - p_prev[1]
        dir_vel  = np.sqrt(delta_x**2 + delta_y**2) / delta_t
        R_rel    = R_prev.T @ R
        theta    = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        theta    = noramlize_angle(theta)
        ang_vel  = theta / delta_t

    # Backup
    compute_dir_ang_vels.sec_prev = sec
    compute_dir_ang_vels.p_prev   = p
    compute_dir_ang_vels.R_prev   = R
    return dir_vel,ang_vel

def get_closest_xy_idx(xy,xy_traj):
    """
    Get the index of the closest point in a trajectory to a given point.
    
    Parameters:
        xy (np.array): Query point.
        xy_traj (array): Trajectory of points.
    
    Returns:
        int: Index of the closest point.
    """
    if len(xy.shape) == 1:
        xy_query = xy[np.newaxis,:]
    else:
        xy_query = xy.copy()
    dists = cdist(xy[np.newaxis,:],xy_traj)
    idx_closest = np.argmin(dists)
    return idx_closest

def get_closest_xy(xy,xy_traj):
    """
    Get the closest point and its index from a trajectory.
    
    Parameters:
        xy (np.array): Query point.
        xy_traj (array): Trajectory of points.
    
    Returns:
        tuple: (index, closest point)
    """
    idx_closest = get_closest_xy_idx(xy,xy_traj)
    xy_closest  = xy_traj[idx_closest]
    return idx_closest,xy_closest

def compute_distance_from_xy_to_xy_traj(xy,xy_traj):
    """
    Compute the shortest distance from a point to a trajectory.
    
    Parameters:
        xy (np.array): Query point.
        xy_traj (array): Trajectory of points.
    
    Returns:
        float: Minimum distance.
    """
    idx_closest,xy_closest = get_closest_xy(xy,xy_traj)
    dist = np.linalg.norm(xy-xy_closest) # distance
    return dist

def compute_angle_from_xy(xy_fr,xy_to):
    """
    Compute the angle (in radians) from one point to another.
    
    Parameters:
        xy_fr (np.array): Starting point.
        xy_to (np.array): Ending point.
    
    Returns:
        float: Angle in radians.
    """
    delta = xy_to - xy_fr
    angle = np.arctan2(delta[1],delta[0])
    return angle

def compute_point_to_segment_distance(p,p_a,p_b):
    """
    Compute the distance from a point to a line segment and return the closest point.
    
    Parameters:
        p (np.array): Query point.
        p_a (np.array): Segment start.
        p_b (np.array): Segment end.
    
    Returns:
        tuple: (distance, closest point)
    """
    # Vectors
    ab = p_b - p_a
    ap = p - p_a
    bp = p - p_b
    
    # ab squared for reuse
    ab_squared = np.dot(ab, ab)
    
    # If p_a==p_b
    if ab_squared == 0:
        return np.linalg.norm(ap),p_a.copy()
    
    # Check whether the closest point is on the segment
    t = np.dot(ap, ab) / ab_squared
    if 0 <= t <= 1:
        # on the segment
        closest_point = p_a + t * ab
        distance = np.linalg.norm(p - closest_point)
    else:
        # not on the segment
        dist_a = np.dot(ap, ap)
        dist_b = np.dot(bp, bp)
        if dist_a < dist_b:
            closest_point = p_a
            distance = np.sqrt(dist_a)
        else:
            closest_point = p_b
            distance = np.sqrt(dist_b)
    # Return
    return distance, closest_point.copy()
    
def compute_point_to_segments_distance(p,p_traj):
    """
    Compute the minimum distance from a point to a sequence of line segments.
    
    Parameters:
        p (np.array): Query point.
        p_traj (array): Sequence of points defining segments.
    
    Returns:
        tuple: (minimum distance, closest point, segment index)
    """  
    L = p_traj.shape[0]
    dists = np.zeros(L-1)
    p_closest_list = []
    for idx in range(L-1):
        p_a = p_traj[idx,:] # [d]
        p_b = p_traj[idx+1,:] # [d]
        dist,p_closest = compute_point_to_segment_distance(p=p,p_a=p_a,p_b=p_b)
        # Append
        dists[idx] = dist
        p_closest_list.append(p_closest)
    # Return
    idx_seg = np.argmin(dists)
    distance = dists[idx_seg]
    p_closest = p_closest_list[idx_seg].copy()
    return distance,p_closest,idx_seg

def get_colors_from_costs(
        costs,
        cmap='autumn',
        alpha=None,
        cost_max=np.inf,
        color_min=(1, 0, 0, 1.0),
        color_max=(0, 0, 0, 0.02),
    ):
    """
    Map cost values to RGBA colors using a colormap and thresholds.
    
    Parameters:
        costs (array): Array of cost values.
        cmap (str): Colormap name.
        alpha (float, optional): Base alpha value.
        cost_max (float): Maximum cost threshold.
        color_min (tuple): Color for minimum cost.
        color_max (tuple): Color for infeasible costs.
    
    Returns:
        np.array: Array of RGBA colors.
    """
    # Initialize colors array
    colors = np.zeros((len(costs), 4))
    
    # Set infeasible indices to color_max
    infeas_mask = costs >= cost_max
    colors[infeas_mask] = color_max
    
    # Process feasible indices
    feas_mask = costs < cost_max
    if np.any(feas_mask):
        costs_feas = costs[feas_mask]
        costs_feas_nzd = (costs_feas - costs_feas.min()) / (costs_feas.max() - costs_feas.min() + 1e-6)
        cmap_colors = plt.get_cmap(cmap)(costs_feas_nzd)
        if alpha is not None:
            cmap_colors[:, 3] = alpha  # Set alpha channel
        colors[feas_mask] = cmap_colors

    # Handle minimum cost color
    if color_min is not None:
        min_mask = (costs == costs.min()) & (costs < cost_max)
        colors[min_mask] = color_min

    return colors

def xy2xyz(xy, z=0.0):
    """
    Convert 2D coordinates to 3D by appending a z-value.
    
    Parameters:
        xy (np.array): 2D coordinates. 
                       For a single coordinate, shape=(2,).
                       For multiple coordinates, shape=(L, 2).
        z (float): Z-coordinate to append (default is 0.0).
    
    Returns:
        np.array: 3D coordinates. 
                  For a single coordinate, shape=(3,).
                  For multiple coordinates, shape=(L, 3).
    """
    xy = np.asarray(xy)
    if xy.ndim == 1:
        if xy.shape[0] != 2:
            raise ValueError("1D array must have exactly 2 elements.")
        return np.concatenate((xy, np.array([z])))
    elif xy.ndim == 2:
        if xy.shape[1] != 2:
            raise ValueError("2D array must have 2 columns.")
        z_col = np.full((xy.shape[0], 1), z)
        return np.concatenate((xy, z_col), axis=1)
    else:
        raise ValueError("Input must be either a 1D or 2D array.")

def xyz2xy(xyz):
    """
    Convert 3D coordinates to 2D by dropping the z-component.
    
    Parameters:
        xyz (np.array): 3D coordinates. 
                        For a single coordinate, shape=(3,).
                        For multiple coordinates, shape=(L, 3).
    
    Returns:
        np.array: 2D coordinates. 
                  For a single coordinate, shape=(2,).
                  For multiple coordinates, shape=(L, 2).
    """
    xyz = np.asarray(xyz)
    if xyz.ndim == 1:
        if xyz.shape[0] != 3:
            raise ValueError("1D array must have exactly 3 elements.")
        return xyz[:2]
    elif xyz.ndim == 2:
        if xyz.shape[1] != 3:
            raise ValueError("2D array must have 3 columns.")
        return xyz[:, :2]
    else:
        raise ValueError("Input must be either a 1D or 2D array.")

def lookahead_planner(
        xy_curr,
        angle_curr,
        xy_pursuit_traj,
        ogm, # occupancy grid map to check feasibility
        la_dir_vel_list      = np.array([0.1,0.5]), # directional velocities (m/sec)
        la_ang_vel_list      = np.deg2rad([0,-30,-20,-10,-5,+5,+10,+20,+30]), # angular velocities (rad/sec)
        la_dt                = 0.1, # (sec)
        la_T                 = 2.0, # (sec)
        la_pursuit_dist      = 0.5, # lookahead distance (m)
        la_sec_feas_check    = 2.0, # feasibility check time (sec)
        la_sec_pursuit_check = 1.0, # pursuit check time (sec)        
        la_dist_threshold    = 0.15, # pursuit trajectory deviation distance margin (m)
        la_angle_threshold   = np.deg2rad(20), # deviation angle margin (rad)
        la_turn_ang_vel      = np.deg2rad(45), # in-place rotation angular velocity (rad/sec)
        use_local_grid       = False,
    ):
    """
    Plan a lookahead motion by sampling trajectories and selecting the best command.
    
    Parameters:
        xy_curr (np.array): Current (x,y) position.
        angle_curr (float): Current heading angle.
        xy_pursuit_traj (array): Pursuit trajectory.
        ogm: Occupancy grid map for feasibility checking.
        la_dir_vel_list (array): List of directional velocities.
        la_ang_vel_list (array): List of angular velocities.
        la_dt (float): Time step for lookahead.
        la_T (float): Total lookahead time.
        la_pursuit_dist (float): Desired pursuit distance.
        la_sec_feas_check (float): Time for feasibility check.
        la_sec_pursuit_check (float): Time for pursuit check.
        la_dist_threshold (float): Deviation distance threshold.
        la_angle_threshold (float): Deviation angle threshold.
        la_turn_ang_vel (float): In-place turning angular velocity.
        use_local_grid (bool): Whether to use local grid for feasibility.
    
    Returns:
        tuple: (dir_vel_cmd, ang_vel_cmd, lap_info)
    """
    
    # Configuration
    n_la_traj = len(la_dir_vel_list)*len(la_ang_vel_list)
    
    # Buffers
    la_dir_ang_vels_list     = [] # lookahead directional and angular velocities
    xy_heading_la_trajs_list = [] # lookahead trajs
    xy_la_pursuit_check_list = [] # pursuit check list (possibly short horizon)
    xy_la_feas_check_list    = [] # feasibility check list (possibly long horizon)
    
    # Costs
    costs = np.zeros(n_la_traj) # initialize costs
    
    # Get the current pursuit point
    dist2traj,xy_closest,idx_seg = compute_point_to_segments_distance(
        p=xy_curr,p_traj=xy_pursuit_traj)
    
    # Check finished
    L = len(xy_pursuit_traj) # number of pursuit points
    if idx_seg >= L-2: # if we only have two pursuit points left
        is_finished = True
    else:
        is_finished = False
        
    # Lookahead planner
    if is_finished: # if finished, stop
        xy_pursuit = xy_pursuit_traj[-1,:] # when finished, let the pursuit point to be the final point
        dir_vel_cmd,ang_vel_cmd = 0.0,0.0 # make robot stop
        angle_diff = 0.0
    else:
        # Cmopute the current pursuit point using 'la_pursuit_dist'
        dist_sum = np.linalg.norm(xy_pursuit_traj[idx_seg+1,:]-xy_closest)
        loop_cnt = 0
        while True:
            # Increase loop counter
            loop_cnt = loop_cnt + 1
            # Segment index to check
            idx_seg_check = idx_seg+loop_cnt+1
            if idx_seg_check >= (L-1): 
                xy_pursuit = xy_pursuit_traj[-1,:] # last pursuit point
                break
            else:
                xy_fr = xy_pursuit_traj[idx_seg_check-1,:]
                xy_to = xy_pursuit_traj[idx_seg_check,:]
                dist_sum_check = dist_sum + np.linalg.norm(xy_to-xy_fr)
                if dist_sum_check > la_pursuit_dist: # if 'xy_pursuit' lies within the next segment
                    dist_remain = la_pursuit_dist - dist_sum 
                    xy_pursuit = xy_fr + dist_remain*np_uv(xy_to-xy_fr)
                    break
                else: dist_sum = dist_sum_check # update 'dist_sum'
            
        # If robot devitate too much from 'xy_pursuit_traj', frist pursuit 'xy_closest'
        # otherwise, pursuit 'xy_pursuit_traj' using lookahead trajectories
        if dist2traj > la_dist_threshold: # if robot deviates too far from 'xy_pursuit_traj', move to 'xy_closest'
            angle_curr2closest = compute_angle_from_xy(xy_fr=xy_curr,xy_to=xy_closest)
            angle_diff = noramlize_angle(angle_curr2closest-angle_curr)
            if np.abs(angle_diff) > np.deg2rad(5): # turn to 'xy_closest'
                dir_vel_cmd,ang_vel_cmd = 0.0,la_turn_ang_vel*np.sign(angle_diff) # in-place rotation
            else: # go to 'xy_closest'
                dir_vel_cmd,ang_vel_cmd = 0.3,np.deg2rad(5)*np.sign(angle_diff)
        else: # robot is close enough to 'xy_pursuit_traj', follow the trajectory by pursuiting 'xy_pursuit'
            angle_closest2pursuit = compute_angle_from_xy(xy_fr=xy_closest,xy_to=xy_pursuit)
            angle_diff = noramlize_angle(angle_closest2pursuit-angle_curr)
            if np.abs(angle_diff) > la_angle_threshold: # angle deviates
                dir_vel_cmd,ang_vel_cmd = 0.0,la_turn_ang_vel*np.sign(angle_diff) # in-place rotation
            else:
                # lookahead planner
                for la_dir_vel in la_dir_vel_list:
                    for la_ang_vel in la_ang_vel_list:
                        la_secs,la_xy_traj,la_heading_traj = compute_lookahead_traj(
                            xy      = xy_curr,
                            heading = angle_curr,
                            dir_vel = la_dir_vel,
                            ang_vel = la_ang_vel,
                            dt      = la_dt,
                            T       = la_T,
                        )
                        # Append
                        la_dir_ang_vels_list.append((la_dir_vel,la_ang_vel))
                        xy_heading_la_trajs_list.append((la_xy_traj,la_heading_traj))
                        
                # Compute costs of the lookahead trajectories
                for t_idx in range(n_la_traj): # for each lookahead trajectory
                    (la_xy_traj,la_heading_traj) = xy_heading_la_trajs_list[t_idx]
                    
                    # Check the feasibility (check occupied)
                    idx_feas_check = np.argmin(np.abs(la_secs-la_sec_feas_check))
                    xy_feas_check  = la_xy_traj[idx_feas_check,:]
                    is_occupied = ogm.check_point_occupancy(
                            x              = xy_feas_check[0],
                            y              = xy_feas_check[1],
                            use_margin     = True,
                            use_local_grid = use_local_grid
                        )
                    
                    # Check the feasibility of the half-way point
                    idx_feas_check_half = np.argmin(np.abs(la_secs-la_sec_feas_check/2.0))
                    xy_feas_check_half  = la_xy_traj[idx_feas_check_half,:]
                    is_occupied_half    = ogm.check_point_occupancy(
                            x              = xy_feas_check_half[0],
                            y              = xy_feas_check_half[1],
                            use_margin     = True,
                            use_local_grid = use_local_grid
                        )
                    
                    if is_occupied or is_occupied_half: # if occupied
                        costs[t_idx] = costs[t_idx] + 1.0 # add big cost to infeasible lookahead trajectory
                    else: # if not occupied (feasible)
                        xy_la_pursuit_check = la_xy_traj[np.argmin(np.abs(la_secs-la_sec_pursuit_check)),:]
                        costs[t_idx] = costs[t_idx] + np.linalg.norm(xy_la_pursuit_check-xy_pursuit)
                        # Append
                        xy_la_feas_check_list.append(xy_feas_check)
                        xy_la_pursuit_check_list.append(xy_la_pursuit_check)
                        
                # Get command with the smallest cost
                idx_min = np.argmin(costs)
                (dir_vel_cmd,ang_vel_cmd) = la_dir_ang_vels_list[idx_min]
            
    # Append 'lap_info'
    lap_info = {}
    lap_info['xy_curr']                  = xy_curr # current point
    lap_info['xy_pursuit_traj']          = xy_pursuit_traj # pursuit trajectory
    lap_info['costs']                    = costs # costs of the lookahead trajectories
    lap_info['dist2traj']                = dist2traj
    lap_info['xy_closest']               = xy_closest
    lap_info['xy_pursuit']               = xy_pursuit
    lap_info['angle_diff']               = angle_diff
    lap_info['la_dir_ang_vels_list']     = la_dir_ang_vels_list
    lap_info['xy_heading_la_trajs_list'] = xy_heading_la_trajs_list
    lap_info['xy_la_pursuit_check_list'] = xy_la_pursuit_check_list # pursuit check positions
    lap_info['xy_la_feas_check_list']    = xy_la_feas_check_list # feasibility check positions
    
    # Return
    return dir_vel_cmd,ang_vel_cmd,lap_info

def plot_lookahead_planner_on_env(
        env,
        lap_info,
        plot_la_traj              = True, # this takes a lot of time
        plot_pursuit_check_points = True,
        plot_feas_check_points    = True,
        plot_pursuit_traj         = True,
        plot_pursuit_point        = True,
        plot_current_point        = True,
        plot_closest_point        = True,
    ):
    """
    Plot lookahead planner trajectories and check points on the environment.
    
    Parameters:
        env: Simulation environment.
        lap_info (dict): Lookahead planner information.
        plot_la_traj (bool): Plot lookahead trajectories.
        plot_pursuit_check_points (bool): Plot pursuit check points.
        plot_feas_check_points (bool): Plot feasibility check points.
        plot_pursuit_traj (bool): Plot pursuit trajectory.
        plot_pursuit_point (bool): Plot pursuit point.
        plot_current_point (bool): Plot current position.
        plot_closest_point (bool): Plot closest point on pursuit trajectory.
    
    Returns:
        None
    """
    # Parse
    costs = lap_info['costs']
    xy_heading_la_trajs_list = lap_info['xy_heading_la_trajs_list']
    xy_la_pursuit_check_list = lap_info['xy_la_pursuit_check_list']
    xy_la_feas_check_list    = lap_info['xy_la_feas_check_list']
    xy_curr                  = lap_info['xy_curr']
    xy_closest               = lap_info['xy_closest']
    xy_pursuit               = lap_info['xy_pursuit']
    xy_pursuit_traj          = lap_info['xy_pursuit_traj']
    
    # Colors
    colors = get_colors_from_costs(
        costs     = costs,
        cmap      = 'summer',
        alpha     = 0.5,
        cost_max  = 1.0,
        color_max = (0.5,0.5,0.5,0.1),  # max_cost==infeasible 
        color_min = (1,0,0,0.5),
    )
    
    # Plot look-ahead trajectories
    if plot_la_traj:
        for traj_idx,(la_xy_traj,la_heading_traj) in enumerate(xy_heading_la_trajs_list):
            color = colors[traj_idx]
            if traj_idx == np.argmin(costs): # minimum cost trajectory
                env.plot_xy_heading_traj(la_xy_traj,la_heading_traj,
                                        r=0.005,plot_arrow=False,plot_cylinder=True,rgba=color)
            else: # other trajectories
                env.plot_xy_heading_traj(la_xy_traj,la_heading_traj,
                                        r=0.002,plot_arrow=False,plot_cylinder=True,rgba=color)
    
    # Plot pursuit check points
    if plot_pursuit_check_points:
        for xy in xy_la_pursuit_check_list: # pursuit check
            env.plot_sphere(p=xy2xyz(xy),r=0.005,rgba=(1,0,0,0.5)) 
    
    # Plot feasibility check points
    if plot_feas_check_points:
        for xy in xy_la_feas_check_list: # feasibility check
            env.plot_sphere(p=xy2xyz(xy),r=0.005,rgba=(0,0,0,0.5))
        
    # Plot pursuit trajectory
    if plot_pursuit_traj:
        env.plot_traj(
            xy_pursuit_traj,
            rgba          = (0,0,0,0.25),
            plot_line     = False,
            plot_cylinder = True,
            cylinder_r    = 0.01,
            plot_sphere   = True,
            sphere_r      = 0.025,
        )
        
    # Plot pursuit point 'xy_pursuit
    if plot_pursuit_point:
        env.plot_sphere(p=xy2xyz(xy_pursuit),r=0.03,rgba=(1,0,0,0.5),label='')
    
    # Plot the current point
    if plot_current_point:
        env.plot_sphere(p=xy2xyz(xy_curr),r=0.02,rgba=(0,0,1,0.5),label='')
    
    # Plot the closest point 'xy_closest'
    if plot_closest_point:
        env.plot_cylinder_fr2to(
            p_fr = xy2xyz(xy_curr),
            p_to = xy2xyz(xy_closest),
            r    = 0.002,
            rgba = (0,0,1,0.5),
        )
        env.plot_sphere(p=xy2xyz(xy_closest),r=0.02,rgba=(0,0,1,0.5),label='') 
    
def get_xy_scan_from_env(env,sensor_name_prefix='rf_'):
    """
    Retrieve a 2D scan (XY coordinates) from the environment sensors.
    
    Parameters:
        env: Simulation environment.
        sensor_name_prefix (str): Prefix for sensor names.
    
    Returns:
        np.array: Array of scanned (x,y) points.
    """
    p_rf_list = env.get_p_rf_obs_list(sensor_names=env.get_sensor_names(prefix=sensor_name_prefix))
    xy_scan = np.array(p_rf_list)[:,:2] # [L x 2]
    return xy_scan
    
class GaussianRandomPathClass(object):
    """
    Gaussian Random Path (GRP) class for generating random paths via Gaussian processes.
    """
    def __init__(
        self,
        name     = 'GRP',
        kernel   = kernel_levse,
        hyp_mean = {'g':1.0,'l':1.0,'w':1e-6},
        hyp_var  = {'g':1.0,'l':1.0,'w':1e-6}
    ):
        self.name     = name
        self.kernel   = kernel
        self.hyp_mean = hyp_mean
        self.hyp_var  = hyp_var
        # Initialize with default anchors/test
        self.set_data()

    def set_data(
        self,
        t_anchor     = np.linspace(0.0, 1.0, 10).reshape((-1,1)),
        x_anchor     = np.zeros((10,2)),
        l_anchor     = np.ones((10,1)),
        t_test       = np.linspace(0.0, 1.0, 100).reshape((-1,1)),
        l_test       = np.ones((100,1)),
        hyp_mean     = None,
        hyp_var      = None,
        w_chol       = 1e-10,
        APPLY_EPSRU  = False,
        t_eps        = 0.0001,
        l_eps        = 1.0,
        x_diff_start = None,
        x_diff_end   = None,
        SKIP_GP_VAR  = False
    ):
        """
            Set anchor and test data, compute GP mean/variance.
        """
        # Anchors
        self.t_anchor = t_anchor.astype(float)
        self.x_anchor = x_anchor.astype(float)
        self.l_anchor = l_anchor.astype(float)
        self.n_anchor = self.x_anchor.shape[0]
        self.d_anchor = self.x_anchor.shape[1]
        # Test
        self.t_test  = t_test.astype(float)
        self.l_test  = l_test.astype(float)
        self.n_test  = self.t_test.shape[0]

        if hyp_mean is not None: self.hyp_mean = hyp_mean
        if hyp_var  is not None: self.hyp_var  = hyp_var

        # Epsilon run-up
        self.APPLY_EPSRU = APPLY_EPSRU
        if APPLY_EPSRU:
            # compute x_diff_start/end if needed and insert runup points
            # (omitted here for brevity)
            pass

        # GP mean
        if self.kernel.__name__ == 'kernel_levse':
            l_anchor_mean = np.ones((self.n_anchor,1))
            K_tt = self.kernel(self.t_test, self.t_anchor,
                                self.l_test, l_anchor_mean,
                                self.hyp_mean)
            K_aa = self.kernel(self.t_anchor, self.t_anchor,
                                l_anchor_mean, l_anchor_mean,
                                self.hyp_mean) + self.hyp_mean['w']*np.eye(self.n_anchor)
        else:
            K_tt = self.kernel(self.t_test, self.t_anchor, self.hyp_mean)
            K_aa = self.kernel(self.t_anchor, self.t_anchor, self.hyp_mean) + self.hyp_mean['w']*np.eye(self.n_anchor)

        self.x_anchor_mean = self.x_anchor.mean(axis=0)
        self.gamma_test    = np.linalg.solve(K_aa, self.x_anchor - self.x_anchor_mean)
        self.mean_test     = K_tt @ self.gamma_test + self.x_anchor_mean

        # GP variance
        if SKIP_GP_VAR:
            return
        if self.kernel.__name__ == 'kernel_levse':
            K_tt_var = self.kernel(self.t_test, self.t_test,
                                    self.l_test, self.l_test,
                                    self.hyp_var)
            K_ta_var = self.kernel(self.t_test, self.t_anchor,
                                    self.l_test, self.l_anchor,
                                    self.hyp_var)
            K_aa_var = self.kernel(self.t_anchor, self.t_anchor,
                                    self.l_anchor, self.l_anchor,
                                    self.hyp_var) + self.hyp_mean['w']*np.eye(self.n_anchor)
        else:
            K_tt_var = self.kernel(self.t_test, self.t_test, self.hyp_var)
            K_ta_var = self.kernel(self.t_test, self.t_anchor, self.hyp_var)
            K_aa_var = self.kernel(self.t_anchor, self.t_anchor, self.hyp_var) + self.hyp_mean['w']*np.eye(self.n_anchor)

        self.var_test      = K_tt_var - K_ta_var @ np.linalg.solve(K_aa_var, K_ta_var.T)
        self.std_diag_test = np.sqrt(np.diag(self.var_test)).reshape(-1,1)
        self.var_chol_test = np.linalg.cholesky(self.var_test + w_chol*np.eye(self.n_test))

    def sample(self, n_sample=10, rand_type='Gaussian'):
        """
            Sample multiple trajectories (mean + variance).
        """
        samples = []
        for _ in range(n_sample):
            if rand_type=='Gaussian':
                R = np.random.randn(self.n_test, self.d_anchor)
            else:
                R = 2*(np.random.rand(self.n_test, self.d_anchor)-0.5)
            samples.append(self.mean_test + self.var_chol_test @ R)
        return samples, self.t_test

    def sample_one_traj(
        self,
        rand_type='Gaussian',
        ORG_PERTURB=False,
        perturb_gain=1.0,
        ss_x_min=None,
        ss_x_max=None,
        ss_margin=None
    ):
        """
            Sample a single trajectory with optional perturb/squash.
        """
        samples, _ = self.sample(n_sample=1, rand_type=rand_type)
        traj = samples[0]
        if ORG_PERTURB:
            traj += (2*perturb_gain*np.random.rand(1,self.d_anchor)-perturb_gain)
        if ss_x_min is not None and ss_x_max is not None and ss_margin is not None:
            traj = soft_squash_multidim(traj, ss_x_min, ss_x_max, ss_margin)
        return traj, self.t_test

    def plot(
        self,
        n_sample   = 10,
        ss_x_min   = None,
        ss_x_max   = None,
        ss_margin  = None,
        figsize    = (6,3),
        subplot_rc = None,
        lw_sample  = 0.25,
        ylim       = None,
        title_str  = None
    ):
        """
            Plot mean and samples with 2-sigma.
        """
        samples, t = self.sample(n_sample=n_sample)
        for d in range(self.d_anchor):
            plt.figure(figsize=figsize)
            for traj in samples:
                plt.plot(t, traj[:,d], '-', alpha=0.4)
            plt.plot(t, self.mean_test[:,d], '-', lw=2)
            plt.fill_between(t.squeeze(),
                             self.mean_test[:,d]-2*self.std_diag_test.squeeze(),
                             self.mean_test[:,d]+2*self.std_diag_test.squeeze(),
                             alpha=0.2)
            if ylim: plt.ylim(ylim)
            plt.title(title_str or f"Dim-{d}")
            plt.show()

    def set_prior(
        self,
        q_init  = None,
        dur_sec = 2.0,
        HZ      = 50,
        hyp     = None,
        eps_sec = 0.01
    ):
        """
            Initialize prior with uniform or run-up anchors.
        """
        if q_init is None: q_init = np.zeros(self.d_anchor)
        if eps_sec==0:
            t_anchor = np.array([[0.0,dur_sec]]).T
            x_anchor = np.tile(q_init, (2,1))
            l_anchor = np.ones((2,1))
        else:
            t_anchor = np.array([[0.0,eps_sec,dur_sec-eps_sec,dur_sec]]).T
            x_anchor = np.tile(q_init, (4,1))
            l_anchor = np.ones((4,1))
        t_test = np.linspace(0.0, dur_sec, int(dur_sec*HZ)).reshape(-1,1)
        l_test = np.ones_like(t_test)
        self.set_data(t_anchor, x_anchor, l_anchor,
                      t_test, l_test, hyp_mean=hyp, hyp_var=hyp)

    def interpolate(
        self,
        t_anchor,
        x_anchor,
        t_test,
        hyp={'g':1,'l':1/5,'w':1e-8},
        APPLY_EPSRU=False,
        t_eps=0.05,
        x_diff_start=None,
        x_diff_end=None,
        x_min=None,
        x_max=None,
        margin=1.0
    ):
        """
            Interpolate trajectory given anchors.
        """
        self.set_data(
            t_anchor=t_anchor,
            x_anchor=x_anchor,
            l_anchor=np.ones((len(t_anchor),1)),
            t_test=t_test,
            l_test=np.ones((len(t_test),1)),
            hyp_mean=hyp,
            hyp_var=hyp,
            APPLY_EPSRU=APPLY_EPSRU,
            t_eps=t_eps,
            x_diff_start=x_diff_start,
            x_diff_end=x_diff_end,
            SKIP_GP_VAR=True
        )
        traj = self.mean_test
        if x_min is not None and x_max is not None:
            traj = soft_squash_multidim(traj, x_min, x_max, margin)
        return traj

    def lev_interpolate(
        self,
        t_anchor,
        x_anchor,
        t_test,
        hyp={'g':1,'l':1/5,'w':1e-8},
        lev_val=0.9,
        APPLY_EPSRU=False,
        t_eps=0.05,
        x_diff_start=None,
        x_diff_end=None,
        x_min=None,
        x_max=None,
        margin=1.0
    ):
        """
            Leveraged interpolation with variance.
        """
        l_anchor = lev_val*np.ones((len(t_anchor),1))
        self.set_data(
            t_anchor=t_anchor,
            x_anchor=x_anchor,
            l_anchor=l_anchor,
            t_test=t_test,
            l_test=np.ones((len(t_test),1)),
            hyp_mean=hyp,
            hyp_var=hyp,
            APPLY_EPSRU=APPLY_EPSRU,
            t_eps=t_eps,
            x_diff_start=x_diff_start,
            x_diff_end=x_diff_end,
            SKIP_GP_VAR=False
        )
        traj, _ = self.sample_one_traj()
        if x_min is not None and x_max is not None:
            traj = soft_squash_multidim(traj, x_min, x_max, margin)
        return traj
        
def doubly_log_scale(start,stop,num,t_min=1.0,t_max=10.0):
    """
    Generate a doubly logarithmically scaled sequence.
    
    Parameters:
        start (float): Start value.
        stop (float): End value.
        num (int): Number of points.
        t_min (float): Minimum scale value.
        t_max (float): Maximum scale value.
    
    Returns:
        np.array: Doubly log-scaled sequence.
    """
    # Mid point
    t_mid = t_min + (t_max-t_min)/2.0
    # Front half
    log_min = np.log10(t_min)
    log_mid = np.log10(t_mid)
    dense_front = np.logspace(log_min,log_mid,num//2,endpoint=False)
    # Latter half
    dense_back = np.logspace(log_min,log_mid,num//2,endpoint=True)
    dense_back = t_max - (dense_back[::-1] - t_min)
    # Concat
    result = np.concatenate([dense_front, dense_back])
    # Un-normalize
    result = start + (result-t_min) * (stop-start) / (t_max-t_min)
    # Return
    return result

def get_jnt_range_for_free_jnt(jnt_type, jnt_range, names=None):
    """
    Get joint range settings for free joints using roll-pitch-yaw configuration.
    
    Parameters:
        jnt_type (list): List of joint types.
        jnt_range (list): List of joint range arrays.
        names (list, optional): Joint names.
    
    Returns:
        np.array: Combined joint ranges.
    """
    joint_ranges = []  # store the data in a list
    for i, t in enumerate(jnt_type):
        if t == 0:
            rng = np.array([[-3.0, 3.0],
                            [-3.0, 3.0],
                            [-3.0, 3.0],
                            [-3.14, 3.14],
                            [-3.14, 3.14],
                            [-3.14, 3.14]])
            joint_ranges.append(rng)
        elif t == 3 or t==1:
            joint_ranges.append(jnt_range[i].reshape(-1, 2))
    # finally concatenate the list into a NumPy array
    return np.vstack(joint_ranges) if joint_ranges else np.array([]).reshape(-1, 2)

def post_processing_jnt_value_for_free_jnt(output, jnt_type):
    """
    Post-process joint values for free joints (convert RPY to quaternion).
    
    Parameters:
        output (array): Raw joint values.
        jnt_type (list): List of joint types.
    
    Returns:
        np.array: Processed joint values.
    """
    processed_output = []  # store the data in a list
    stacked_idx = 0
    for i, t in enumerate(jnt_type):
        current_jnt_idx = i+stacked_idx
        # for free_joint
        if t == 0:
            temp_xyz = output[current_jnt_idx:current_jnt_idx+3]
            temp_quat = r2quat(rpy2r(output[current_jnt_idx+3:current_jnt_idx+6]))
            processed_output.append(temp_xyz)
            processed_output.append(temp_quat)
            stacked_idx += 5
        # for revolute_joint
        else:
            temp_output = output[current_jnt_idx:current_jnt_idx+1]
            processed_output.append(temp_output)
    # finally concatenate the list into a NumPy array
    return np.hstack(processed_output) if processed_output else np.array([])

def post_processing_ctrl_value_for_free_jnt(output, ctrl_type):
    """
    Post-process control values for free joints (convert RPY to quaternion).
    
    Parameters:
        output (array): Raw control values.
        ctrl_type (list): List of control types.
    
    Returns:
        np.array: Processed control values.
    """
    processed_output = []  # store the data in a list
    stacked_idx = 0
    for i, t in enumerate(ctrl_type):
        current_jnt_idx = i+stacked_idx
        # for free_joint
        if t == 0:
            temp_xyz = output[current_jnt_idx:current_jnt_idx+3]
            temp_quat = r2quat(rpy2r(output[current_jnt_idx+3:current_jnt_idx+6]))
            processed_output.append(temp_xyz)
            processed_output.append(temp_quat)
            stacked_idx += 5
        # for revolute_joint
        else:
            temp_output = output[current_jnt_idx:current_jnt_idx+1]
            processed_output.append(temp_output)
    # finally concatenate the list into a NumPy array
    return np.hstack(processed_output) if processed_output else np.array([])

def print_red(str):
    """
    Print a string in red color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'red'))
    
def print_yellow(str):
    """
    Print a string in yellow color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'yellow'))    

def print_green(str):
    """
    Print a string in green color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'green'))   
    
def print_blue(str):
    """
    Print a string in blue color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'blue'))    

def print_light_green(str):
    """
    Print a string in light green color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'light_green'))    

def print_light_blue(str):
    """
    Print a string in light blue color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'light_blue'))    

def generate_xy_trajs(
        xy_curr,              # current position (m), shape: (2,)
        angle_curr,           # current heading angle (rad), scalar
        v_list = [0.2],       # directional velocities (m/s), list or array
        w_list = [-0.5*np.pi, 0.0, +0.5*np.pi], # angular velocities (rad/s), list or array
        dt     = 0.05,        # time step size (s), scalar
        T      = 1.0,         # total duration (s), scalar
        sparse_interval = 1,  # sparse interval
    ):
    """ 
    Generate sample trajectories based on given velocities and angular velocities.

    Parameters:
        xy_curr (array-like): Current position [x, y] in meters, shape (2,).
        angle_curr (float): Current heading angle in radians.
        v_list (list or array-like): List of linear velocities in m/s.
        w_list (list or array-like): List of angular velocities in rad/s.
        dt (float): Time interval between trajectory points in seconds.
        T (float): Total duration of trajectories in seconds.

    Returns:
        xy_trajs (np.ndarray): Array of trajectory points, shape (num_traj, L, 2), 
                               where num_traj = len(v_list)*len(w_list), L = T/dt.
        heading_trajs (np.ndarray): Array of heading angles, shape (num_traj, L).
        times (np.ndarray): Array of time points, shape (L,).
    """
    # Generate a grid of linear and angular velocities
    v_grid, w_grid = np.meshgrid(v_list, w_list, indexing='ij')
    v_flat = v_grid.ravel()  # shape: (num_traj,)
    w_flat = w_grid.ravel()  # shape: (num_traj,)

    L = int(T / dt)  # Number of time steps
    times = np.arange(0, T, dt)  # shape: (L,)

    # Compute heading angles for each trajectory and timestep
    angles = angle_curr + np.outer(w_flat, times)  # shape: (num_traj, L)

    eps = 1e-8
    x_offsets = np.zeros((v_flat.size, L))  # shape: (num_traj, L)
    y_offsets = np.zeros((v_flat.size, L))  # shape: (num_traj, L)

    # Separate trajectories based on whether angular velocity is zero or not
    nonzero_mask = np.abs(w_flat) > eps
    zero_mask = ~nonzero_mask

    # Closed-form solution for nonzero angular velocities
    if np.any(nonzero_mask):
        w_nz = w_flat[nonzero_mask]
        v_nz = v_flat[nonzero_mask]

        x_offsets[nonzero_mask, :] = (v_nz / w_nz)[:, None] * (
            np.sin(angle_curr + np.outer(w_nz, times)) - np.sin(angle_curr)
        )
        y_offsets[nonzero_mask, :] = - (v_nz / w_nz)[:, None] * (
            np.cos(angle_curr + np.outer(w_nz, times)) - np.cos(angle_curr)
        )

    # Linear movement for zero angular velocities
    if np.any(zero_mask):
        v_z = v_flat[zero_mask]
        x_offsets[zero_mask, :] = v_z[:, None] * times * np.cos(angle_curr)
        y_offsets[zero_mask, :] = v_z[:, None] * times * np.sin(angle_curr)

    # Combine initial positions with calculated offsets to form trajectories
    xy_trajs = np.zeros((v_flat.size, L, 2))  # shape: (num_traj, L, 2)
    xy_trajs[:, :, 0] = xy_curr[0] + x_offsets
    xy_trajs[:, :, 1] = xy_curr[1] + y_offsets

    heading_trajs = angles  # shape: (num_traj, L)

    # Sparse sampling indeces
    if sparse_interval > 1:
        sparse_indices = np.arange(0, L, sparse_interval)
        if sparse_indices[-1] != L - 1:
            sparse_indices = np.append(sparse_indices, L - 1)
        xy_trajs      = xy_trajs[:, sparse_indices, :]
        heading_trajs = heading_trajs[:, sparse_indices]
        times         = times[sparse_indices]

    return xy_trajs, heading_trajs, times

def sample_trajs(
        xy_curr,              # current position (m), shape: (2,)
        angle_curr,           # current heading angle (rad), scalar
        v_list = [0.2],       # directional velocities (m/s), list or array
        w_list = [-0.5*np.pi, 0.0, +0.5*np.pi], # angular velocities (rad/s), list or array
        dt     = 0.05,        # time step size (s), scalar
        T      = 1.0,         # total duration (s), scalar
        sparse_interval = 1,  # sparse interval
    ):
    """ 
    Generate sample trajectories based on given velocities and angular velocities.

    Parameters:
        xy_curr (array-like): Current position [x, y] in meters, shape (2,).
        angle_curr (float): Current heading angle in radians.
        v_list (list or array-like): List of linear velocities in m/s.
        w_list (list or array-like): List of angular velocities in rad/s.
        dt (float): Time interval between trajectory points in seconds.
        T (float): Total duration of trajectories in seconds.

    Returns:
        xy_trajs (np.ndarray): Array of trajectory points, shape (num_traj, L, 2), 
                               where num_traj = len(v_list)*len(w_list), L = T/dt.
        heading_trajs (np.ndarray): Array of heading angles, shape (num_traj, L).
        times (np.ndarray): Array of time points, shape (L,).
    """
    # Generate a grid of linear and angular velocities
    v_grid, w_grid = np.meshgrid(v_list, w_list, indexing='ij')
    v_flat = v_grid.ravel()  # shape: (num_traj,)
    w_flat = w_grid.ravel()  # shape: (num_traj,)

    L = int(T / dt)  # Number of time steps
    times = np.arange(0, T, dt)  # shape: (L,)

    # Compute heading angles for each trajectory and timestep
    angles = angle_curr + np.outer(w_flat, times)  # shape: (num_traj, L)

    eps = 1e-8
    x_offsets = np.zeros((v_flat.size, L))  # shape: (num_traj, L)
    y_offsets = np.zeros((v_flat.size, L))  # shape: (num_traj, L)

    # Separate trajectories based on whether angular velocity is zero or not
    nonzero_mask = np.abs(w_flat) > eps
    zero_mask = ~nonzero_mask

    # Closed-form solution for nonzero angular velocities
    if np.any(nonzero_mask):
        w_nz = w_flat[nonzero_mask]
        v_nz = v_flat[nonzero_mask]

        x_offsets[nonzero_mask, :] = (v_nz / w_nz)[:, None] * (
            np.sin(angle_curr + np.outer(w_nz, times)) - np.sin(angle_curr)
        )
        y_offsets[nonzero_mask, :] = - (v_nz / w_nz)[:, None] * (
            np.cos(angle_curr + np.outer(w_nz, times)) - np.cos(angle_curr)
        )

    # Linear movement for zero angular velocities
    if np.any(zero_mask):
        v_z = v_flat[zero_mask]
        x_offsets[zero_mask, :] = v_z[:, None] * times * np.cos(angle_curr)
        y_offsets[zero_mask, :] = v_z[:, None] * times * np.sin(angle_curr)

    # Combine initial positions with calculated offsets to form trajectories
    xy_trajs = np.zeros((v_flat.size, L, 2))  # shape: (num_traj, L, 2)
    xy_trajs[:, :, 0] = xy_curr[0] + x_offsets
    xy_trajs[:, :, 1] = xy_curr[1] + y_offsets

    heading_trajs = angles  # shape: (num_traj, L)

    # Sparse sampling indeces
    if sparse_interval > 1:
        sparse_indices = np.arange(0, L, sparse_interval)
        if sparse_indices[-1] != L - 1:
            sparse_indices = np.append(sparse_indices, L - 1)
        xy_trajs      = xy_trajs[:, sparse_indices, :]
        heading_trajs = heading_trajs[:, sparse_indices]
        times         = times[sparse_indices]

    return xy_trajs, heading_trajs, times

def norm(x):
    """
    Compute the L2 norm of a vector or matrix.

    Parameters:
        x (array-like): Input vector or matrix.

    Returns:
        float: L2 norm of the input.
    """
    return np.linalg.norm(x)

def print_type_shape(name, x):
    """
    print type and shape
    """
    # if x has a shape attribute, use it as is
    if hasattr(x, 'shape'):
        shape = x.shape
    else:
        try:
            shape = np.array(x).shape
        except Exception:
            shape = "N/A"
    print("['%s'] type:%s shape:%s" % (name, type(x), shape))

def print_gym_env_shape(env):
    """
    print gym observation and action shape
    """
    obs,_ = env.reset()
    action = env.action_space.sample()
    print_type_shape('obs',obs)
    print_type_shape('action',action)
    next_obs,rewards,terminated,truncated,infos = env.step(action=action)
    print (" env.step()")
    print_type_shape('next_obs',next_obs)
    print_type_shape('rewards',rewards)
    print_type_shape('terminated',terminated)
    print_type_shape('truncated',truncated)

def farthest_point_sampling(pcd, n_sample):
    """
    Performs Farthest Point Sampling on a given point cloud.

    Args:
        pcd (numpy.ndarray): The input point cloud of shape (N, D), where N is the number of points and D is the dimensionality.
        n_sample (int): The number of points to sample.

    Returns:
        numpy.ndarray: The indices of the sampled points in the original point cloud.
    """
    
    assert len(pcd.shape) == 2, "Input pcd must be a 2D array."
    N, D = pcd.shape

    # If the number of points is less than or equal to the number of samples, include all points and sample additional ones
    if n_sample >= N:
        additional_indices = np.random.choice(N, n_sample - N, replace=True)
        return np.concatenate([np.arange(N, dtype=np.int32), additional_indices])

    # Initialize arrays
    sampled_indices = []  # List to store the indices of sampled points
    distances = np.ones(N) * 1e7  # Initialize distances to a large value

    # Randomly pick the first point
    first_index = np.random.randint(0, N)
    sampled_indices.append(first_index)

    for _ in range(n_sample - 1):
        # Get the last sampled point
        last_sampled_point = pcd[sampled_indices[-1]].reshape(1, -1)

        # Compute distances from the last sampled point to all other points
        current_distances = np.linalg.norm(pcd - last_sampled_point, axis=1)

        # Update the minimum distances for each point
        distances = np.minimum(distances, current_distances)

        # Select the farthest point from the already sampled points
        next_index = np.argmax(distances)
        sampled_indices.append(next_index)

    return np.array(sampled_indices, dtype=np.int32)    

def add_title_to_img(img,text='Title',margin_top=30,color=(0,0,0),font_size=20,resize=True,shape=(300,300)):
    """
    Add title to image
    """
    # Resize
    img_copied = img.copy()
    if resize:
        img_copied = cv2.resize(img_copied,shape,interpolation=cv2.INTER_NEAREST)
    # Convert to PIL image
    pil_img = Image.fromarray(img_copied)# 
    width, height = pil_img.size
    new_height = margin_top + height
    # Create new image with top margin
    new_img = Image.new("RGB", (width, new_height),color=(255,255,255))
    # Paste the original image
    new_img.paste(pil_img, (0, margin_top))
    # Draw text
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default(size=font_size)
    bbox = draw.textbbox((0,0),text,font=font)
    # Center text
    text_width  = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (margin_top - text_height) // 2
    # Draw text
    draw.text((x, y), text, font=font, fill=color)
    img_with_title = np.array(new_img)
    # Return
    return img_with_title

def block_mtx(M11,M12,M21,M22):
    M_upper = np.concatenate((M11,M12),axis=1)
    M_lower = np.concatenate((M21,M22),axis=1)
    M = np.concatenate((M_upper,M_lower),axis=0)
    return M

def inv_inc(inv_A,b,c):
    """
        Incremental inverse using matrix inverse lemma
    """
    k   = c - b.T @ inv_A @ b
    M11 = inv_A + 1/k * inv_A @ b @ b.T @ inv_A
    M12 = -1/k * inv_A @ b
    M21 = -1/k * b.T @ inv_A
    M22 = 1/k
    M   = block_mtx(M11=M11,M12=M12,M21=M21,M22=M22)
    return M

def det_inc(det_A,inv_A,b,c):
    """
        Incremental determinant computation
    """
    out = det_A * (c - b.T @ inv_A @ b)
    return out

def ikdpp_sampling(
        xs_total,
        qs_total = None,
        n_select = 10,
        n_trunc  = np.inf,
        hyp      = {'g':1.0,'l':0.001,'w':1e-8}
    ):
    """
    Performs (truncated) Incremental k-DPP (i-kDPP) sampling from a given dataset.

    This function incrementally selects a diverse subset of `n_select` points from `xs_total`, 
    using a determinant-based diversity measure. It optionally supports truncation to limit 
    computational cost by considering only a subset of the remaining candidates at each step.

    Args:
        xs_total (np.ndarray): Input data of shape (N, D), where N is the number of data points and D is the dimension.
        qs_total (np.ndarray or None): Optional quality scores for each data point. Default is None.
        n_select (int): Number of points to sample (excluding the initial random one).
        n_trunc (int or float): Maximum number of candidates to consider per iteration (for truncation). Default is np.inf (no truncation).
        hyp (dict): Hyperparameters for the RBF kernel function. Default is {'g':1.0, 'l':1.0, 'w':1e-8}.

    Returns:
        xs_ikdpp (np.ndarray): Sampled subset of shape (n_select, D).
        idxs_ikdpp (np.ndarray): Indices of selected points in the original dataset.
    """
    n_total     = xs_total.shape[0]
    idxs_remain = np.arange(0,n_total,1,dtype=np.int32)
    idxs_select = []
    for i_idx in range(n_select+1): # loop
        n_remain = len(idxs_remain)
        if i_idx == 0: # random first
            idx_select = np.random.permutation(n_total)[0]
            if qs_total is not None:
                q = 1.0+qs_total[idx_select]
            else:
                q = 1.0
            det_K_prev = q
            K_inv_prev = 1/q*np.ones(shape=(1,1))
        else:
            xs_select = xs_total[idxs_select,:]
            # Compute determinants
            dets = np.zeros(shape=n_remain)
            for r_idx in np.random.permutation(n_remain)[:min(n_remain,n_trunc)]:
                # Compute the determinant of the appended kernel matrix 
                k_vec     = kernel_se(
                    X1  = xs_select,
                    X2  = xs_total[idxs_remain[r_idx],:].reshape(1,-1),
                    hyp = hyp)
                if qs_total is not None:
                    q = qs_total[idxs_remain[r_idx]]
                else:
                    q = 1.0
                det_check = det_inc(
                    det_A = det_K_prev,
                    inv_A = K_inv_prev,
                    b     = k_vec,
                    c     = q)
                # Append the determinant
                dets[r_idx] = det_check
            # Get the index with the highest determinant
            idx_temp   = np.where(dets == np.amax(dets))[0][0]
            idx_select = idxs_remain[idx_temp]
            
            # Compute 'det_K_prev' and 'K_inv_prev'
            det_K_prev = dets[idx_temp]
            k_vec      = kernel_se(
                xs_select,
                xs_total[idx_select,:].reshape(1,-1),
                hyp=hyp)
            if qs_total is not None:
                q = 1+qs_total[idx_select]
            else:
                q = 1.0
            K_inv_prev = inv_inc(
                inv_A = K_inv_prev,
                b     = k_vec,
                c     = q)
        # Remove currently selected index from 'idxs_remain'
        idxs_remain = idxs_remain[idxs_remain != idx_select]
        # Append currently selected index to 'idxs_select'
        idxs_select.append(idx_select)
    # Select the subset from 'xs_total' with removing the first sample
    idxs_select = idxs_select[1:] # excluding the first one
    idxs_ikdpp  = np.array(idxs_select)
    xs_ikdpp    = xs_total[idxs_ikdpp]
    return xs_ikdpp,idxs_ikdpp

def find_all_file_paths(root_dir, target_filename, verbose=False):
    """
    Recursively searches for all occurrences of a specified file within a directory tree.

    This function traverses all subdirectories of the given root directory and collects 
    the full paths of files that exactly match the specified file name. It returns a list 
    of matched file paths relative to the environment's filesystem.

    Args:
        root_dir (str): Path to the root directory where the search begins. 
                        Can be an absolute or relative path (e.g., from the Jupyter Notebook).
        target_filename (str): Name of the file to search for (e.g., 'data.json', 'results.csv').

    Returns:
        List[str]: A list of full file paths (including directories) where the target file is found.

    Example:
        >>> find_all_file_paths('projects/', 'config.yaml')
        ['projects/run1/config.yaml', 'projects/experiment/config.yaml']
    """    
    matched_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir): # walk through all subdirectories
        # Check if the target filename exists in the current directory
        if target_filename in filenames:
            matched_paths.append(os.path.join(dirpath, target_filename))
    # Print
    if verbose:
        for path_idx,matched_path in enumerate(matched_paths):
            print("[%d/%d] object_path:[%s]"%(path_idx,len(matched_paths),matched_path))
    
    # Return
    return matched_paths

def indent_xml(elem):
    """
    Convert an XML Element into a neatly formatted, indented XML string.

    Parameters:
        elem (xml.etree.ElementTree.Element): XML element to format.

    Returns:
        str: Pretty-printed XML string without empty lines.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    # Remove empty lines
    lines = [line for line in pretty_xml.splitlines() if line.strip()]
    return "\n".join(lines)

def merge_mjcfs(
        mjcf_model_name     = 'scene', 
        included_mjcf_files = ['a.xml', 'b.xml'], 
        output_xml_path     = 'merged_scene.xml',
        timestep            = 0.002,
        integrator          = 'implicitfast',
        memory              = None,  # "500M", "1G", etc.
        armature            = None,  # e.g., 1.0
        damping             = None,  # e.g., 2.0
        cone                = None,  # e.g., 'elliptic'
        verbose             = True,
    ):
    """
    Merges multiple MJCF files into a single MuJoCo model XML, adds optional simulation settings,
    pretty-prints it, saves it to a file, and returns the file path.

    Args:
        mjcf_model_name (str): The value of the 'model' attribute for the <mujoco> root element.
        included_mjcf_files (List[str]): List of MJCF file paths to include via <include file="..."/> tags.
        output_xml_path (str): File path where the generated XML will be saved.
        timestep (float): Simulation timestep for the <option> tag. Default is 0.002.
        integrator (str): Integrator type for the <option> tag. Default is "implicitfast".
        memory (str or None): Memory setting for the <size> tag (e.g., "500M", "1G"). 
                              If None, the <size> tag is not added. If True, defaults to "1G".
        armature (float or None): Default armature for joints. If None, not added. (e.g., 1.0)
        damping (float or None): Default damping for joints. If None, not added. (e.g., 2.0)
        cone (str or None): Cone friction model. If None, not added. (e.g., 'elliptic')
        verbose (bool): If True, print included files and output path. Default is False.

    Returns:
        str: The path to the saved XML file.
    """
    # Create root element
    mujoco_elem = ET.Element("mujoco", attrib={"model": mjcf_model_name})

    output_dir = os.path.dirname(os.path.abspath(output_xml_path))

    # Add include elements
    for filepath in included_mjcf_files:
        abs_path = os.path.abspath(filepath)
        rel_path = os.path.relpath(abs_path, output_dir)
        ET.SubElement(mujoco_elem, "include", attrib={"file": rel_path})

    # Add <default> tag if needed
    if armature is not None or damping is not None:
        default_elem = ET.SubElement(mujoco_elem, "default")
        joint_attrs = {}
        if armature is not None:
            joint_attrs["armature"] = str(armature)
        if damping is not None:
            joint_attrs["damping"] = str(damping)
        ET.SubElement(default_elem, "joint", attrib=joint_attrs)

    # Add <option> tag with extra attributes if given
    option_attrs = {
        "timestep": str(timestep),
        "integrator": integrator,
    }
    if cone is not None:
        option_attrs["cone"] = cone
    ET.SubElement(mujoco_elem, "option", attrib=option_attrs)

    # Add <size> tag if memory is given
    if memory is not None:
        mem_value = memory if isinstance(memory, str) else "1G"
        ET.SubElement(mujoco_elem, "size", attrib={"memory": mem_value})

    # Convert to pretty string
    pretty_xml = indent_xml(mujoco_elem)

    # Ensure the directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write to file
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    # Verbose output
    if verbose:
        print("[merge_mjcfs] Merging [%d] MJCF files:"%(len(included_mjcf_files)))
        for f_idx,f in enumerate(included_mjcf_files):
            print("  - [%d] [%s]"%(f_idx,f))
        print(f"[merge_mjcfs] Saved merged XML to:[%s]"%(output_xml_path))

    return output_xml_path

def sample_list(list, n_sample, verbose=False):
    """
    Randomly sample n elements from a list.
    
    Parameters:
        list (list): Input list to sample from.
        n_sample (int): Number of samples to draw.
    
    Returns:
        list: Sampled elements.
    """
    subset = random.sample(list,n_sample)

    # Print
    if verbose:
        print("Sampled [%d] elements from the list consists of [%d] items:"%(n_sample,len(list)))
        for i, item in enumerate(subset):
            print(" [%d] [%s]"%(i,item))
    # Return
    return subset

class PID_ControllerClass(object):
    def __init__(
            self,
            name      = 'PID',
            k_p       = 0.01,
            k_i       = 0.0,
            k_d       = 0.001,
            dt        = 0.01,
            dim       = 1,
            dt_min    = 1e-6,
            out_min   = -np.inf,
            out_max   = np.inf,
            ANTIWU    = True,   # anti-windup
            out_alpha = 0.0,    # output EMA (0: no EMA)
        ):
        """
        Initialize PID Controller.
        
        Parameters:
            name (str): Name of the PID controller.
            k_p (float): Proportional gain.
            k_i (float): Integral gain.
            k_d (float): Derivative gain.
            dt (float): Default timestep.
            dim (int): Dimension of control signals.
            dt_min (float): Minimum allowed timestep.
            out_min (float): Minimum control output.
            out_max (float): Maximum control output.
            ANTIWU (bool): Flag to enable anti-windup.
            out_alpha (float): EMA smoothing factor for output (0 for no smoothing).
        """
        self.name      = name
        self.k_p       = k_p
        self.k_i       = k_i
        self.k_d       = k_d
        self.dt        = dt
        self.dim       = dim
        self.dt_min    = dt_min
        self.out_min   = np.array(out_min)
        self.out_max   = np.array(out_max)
        self.ANTIWU    = ANTIWU
        self.out_alpha = out_alpha
        # Buffers
        self.cnt      = 0
        self.x_trgt   = np.zeros(shape=self.dim)
        self.x_curr   = np.zeros(shape=self.dim)
        self.out_val  = np.zeros(shape=self.dim)
        self.out_val_prev = np.zeros(shape=self.dim)
        self.t_curr   = 0.0
        self.t_prev   = 0.0
        self.err_curr = np.zeros(shape=self.dim)
        self.err_intg = np.zeros(shape=self.dim)
        self.err_prev = np.zeros(shape=self.dim)
        self.p_term   = np.zeros(shape=self.dim)
        self.d_term   = np.zeros(shape=self.dim)
        self.err_out  = np.zeros(shape=self.dim)
        
    def reset(self,t_curr=0.0):
        """
        Reset PID controller states.
        
        Parameters:
            t_curr (float): Current timestamp to reset controller.
        
        Returns:
            None
        """
        self.cnt      = 0
        self.x_trgt   = np.zeros(shape=self.dim)
        self.x_curr   = np.zeros(shape=self.dim)
        self.out_val  = np.zeros(shape=self.dim)
        self.out_val_prev = np.zeros(shape=self.dim)
        self.t_curr   = t_curr
        self.t_prev   = t_curr
        self.err_curr = np.zeros(shape=self.dim)
        self.err_intg = np.zeros(shape=self.dim)
        self.err_prev = np.zeros(shape=self.dim)
        self.p_term   = np.zeros(shape=self.dim)
        self.d_term   = np.zeros(shape=self.dim)
        self.err_out  = np.zeros(shape=self.dim)
        
    def update(
        self,
        t_curr  = None,
        x_trgt  = None,
        x_curr  = None,
        verbose = False
        ):
        """
        Update PID controller with new measurements.
        
        Parameters:
            t_curr (float): Current timestamp.
            x_trgt (np.array): Target setpoint.
            x_curr (np.array): Current measured value.
            verbose (bool): Flag to enable detailed logging.
        
        Returns:
            None
        """
        if x_trgt is not None:
            self.x_trgt  = np.array(x_trgt)
        if t_curr is not None:
            self.t_curr  = t_curr
        if x_curr is not None:
            self.x_curr  = np.array(x_curr)
            # PID controller updates here
            self.dt       = max(self.t_curr - self.t_prev,self.dt_min)
            self.err_curr = self.x_trgt - self.x_curr     
            self.err_intg = self.err_intg + (self.err_curr*self.dt)
            self.err_diff = self.err_curr - self.err_prev
            
            if self.ANTIWU: # anti-windup
                self.err_out = self.err_curr * self.out_val
                self.err_intg[self.err_out<0.0] = 0.0
            
            if self.dt > self.dt_min:
                self.p_term   = self.k_p * self.err_curr
                self.i_term   = self.k_i * self.err_intg
                self.d_term   = self.k_d * self.err_diff / self.dt
                self.out_val  = np.clip(
                    a     = self.p_term + self.i_term + self.d_term,
                    a_min = self.out_min,
                    a_max = self.out_max)
                # Smooth the output control value using EMA
                self.out_val = self.out_alpha*self.out_val_prev + \
                    (1.0-self.out_alpha)*self.out_val
                self.out_val_prev = self.out_val

                if verbose:
                    print ("[%s] cnt:[%d] t_curr:[%.5f] dt:[%.5f]"%
                           (self.name,self.cnt,self.t_curr,self.dt))
                    print (" x_trgt:   %s"%(self.x_trgt))
                    print (" x_curr:   %s"%(self.x_curr))
                    print (" err_curr: %s"%(self.err_curr))
                    print (" err_intg: %s"%(self.err_intg))
                    print (" p_term:   %s"%(self.p_term))
                    print (" i_term:   %s"%(self.i_term))
                    print (" d_term:   %s"%(self.d_term))
                    print (" out_val:  %s"%(self.out_val))
                    print (" err_out:  %s"%(self.err_out))
            # Backup
            self.t_prev   = self.t_curr
            self.err_prev = self.err_curr
        # Counter
        if (t_curr is not None) and (x_curr is not None):
            self.cnt = self.cnt + 1
            
    def out(self):
        """
        Retrieve current PID control output.
        
        Returns:
            np.array: Current PID controller output.
        """
        return self.out_val.copy()

class RecentBufferClass:
    def __init__(self, dim, maxlen):
        """
        Initializes a buffer to store the most recent fixed-length data points.

        Args:
            dim (int): The dimension of each data entry.
            maxlen (int): Maximum number of recent data entries to retain.
        """
        self.dim = dim
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)

    def append(self, data):
        """
        Appends a new data entry to the buffer.

        Args:
            data (array-like): New data entry of dimension `dim`.

        Raises:
            ValueError: If input data dimension does not match the specified `dim`.
        """
        if len(data) != self.dim:
            raise ValueError(f"Input data dimension must be {self.dim}")
        self.buffer.append(np.array(data))

    def data(self):
        """
        Retrieves all stored data entries as a numpy array.

        Returns:
            numpy.ndarray: Array of shape `(current_buffer_length, dim)`.
        """
        return np.array(self.buffer)

    def is_full(self):
        """
        Checks whether the buffer has reached its maximum capacity.

        Returns:
            bool: True if buffer is full, False otherwise.
        """
        return len(self.buffer) == self.maxlen

    def clear(self):
        """
        Clears all data from the buffer.
        """
        self.buffer.clear()

class VelocityTrackerClass:
    def __init__(
            self,
            kp_v=0.1, 
            kp_w=0.1,
            v_clip=(-2.0, 2.0), 
            w_clip=(-np.pi, np.pi),
            delta_clip=0.05,
            wheel_radius=0.143,
            wheel_distance=0.556,
        ):
        self.kp_v = kp_v
        self.kp_w = kp_w
        self.v_min, self.v_max = v_clip
        self.w_min, self.w_max = w_clip
        self.delta_clip = delta_clip
        self.r = wheel_radius
        self.a = wheel_distance
        self.v_trgt = 0.0
        self.w_trgt = 0.0

    def reset(self):
        self.v_trgt = 0.0
        self.w_trgt = 0.0

    def update(self, v_cmd, w_cmd, qvel_base, R_base):
        v_curr_raw = qvel_base[:2]
        v_curr_mag = np.linalg.norm(v_curr_raw)
        forward_direction = R_base[:2, 0]
        sign_dir = np.sign(np.dot(v_curr_raw, forward_direction))
        v_curr = sign_dir * v_curr_mag
        w_curr = qvel_base[5]

        v_err = v_cmd - v_curr
        w_err = w_cmd - w_curr

        self.v_trgt += np.clip(self.kp_v * v_err, -self.delta_clip, self.delta_clip)
        self.w_trgt += np.clip(self.kp_w * w_err, -self.delta_clip, self.delta_clip)

        self.v_trgt = np.clip(self.v_trgt, self.v_min, self.v_max)
        self.w_trgt = np.clip(self.w_trgt, self.w_min, self.w_max)

        if np.abs(v_cmd) < 1e-1:
            self.v_trgt = 0.0
        if np.abs(w_cmd) < 1e-1:
            self.w_trgt = 0.0

        wheel_velocities = self._compute_wheel_velocities()
        info = {
            'v_trgt': self.v_trgt,
            'w_trgt': self.w_trgt,
            'v_curr': v_curr,
            'w_curr': w_curr,
        }

        return wheel_velocities,info

    def _compute_wheel_velocities(self):
        rrv = (self.v_trgt + (self.a / 2.0) * self.w_trgt) / self.r
        rlv = (self.v_trgt - (self.a / 2.0) * self.w_trgt) / self.r
        frv = rrv
        flv = rlv

        return {'rrv': rrv, 'rlv': rlv, 'frv': frv, 'flv': flv}

def rand():
    """
    Generate a random float between 0 and 1.
    
    Returns:
        float: Random float in the range [0, 1).
    """
    return np.random.rand()

def damped_pinv_svd(J, damping=1e-3):
    """
    Compute damped pseudo-inverse using SVD (more stable for all J shapes)
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S**2 + damping**2)
    return Vt.T @ np.diag(S_damped) @ U.T

def get_arc(p_center, axis, uv_front, r, theta_min, theta_max, L):
    """
    Generates a set of 3D points forming an arc on a plane

    Parameters:
        p_center (np.ndarray): A (3,) array representing the center of the arc.
        axis (np.ndarray): A (3,) unit vector representing the axis of rotation
        uv_front (np.ndarray): A (3,) unit vector representing the forward direction
        r (float): Radius of the arc.
        theta_min (float): Start angle of the arc in radians, relative to `uv_front` direction.
        theta_max (float): End angle of the arc in radians, relative to `uv_front` direction.
        L (int): Number of points to sample along the arc.

    Returns:
        np.ndarray: An (L, 3) array of 3D points along the arc.
    """
    axis = axis / np.linalg.norm(axis)
    uv_front = uv_front / np.linalg.norm(uv_front)

    # Construct an orthonormal basis on the rotation plane
    uv_right = np.cross(axis, uv_front)
    uv_right /= np.linalg.norm(uv_right)
    uv_front = np.cross(uv_right, axis)  # Ensure exact orthogonality

    # Create angles
    thetas = np.linspace(theta_min, theta_max, L).reshape(-1, 1)  # (L, 1)

    # Compute direction vectors
    directions = np.cos(thetas) * uv_front + np.sin(thetas) * uv_right  # (L, 3)

    # Compute arc points
    points = p_center + r * directions  # (L, 3)

    return points


class TimerClass(object):
    """ 
    TimerClass is a simple timer class to control the execution time of a loop.
    It can be used to run a loop at a specific frequency (Hz) and can also handle maximum execution time.
    It is useful for controlling the execution of tasks in a simulation or real-time system. 

    Example usage:
        tmr_pub = TimerClass(name='Person publisher (20Hz)', Hz=20, max_sec=np.inf, verbose=True)
        tmr_pub.start()
        while tmr_pub.is_notfinished():  # loop
            if tmr_pub.do_run():  # publish persons to state machines (20Hz)
                A.publishPersons()
                tmr_pub.end()
    """
    def __init__(self,name='timer',Hz=10,max_sec=np.inf,verbose=True):
        """
        Initialize the TimerClass with a name, frequency (Hz), maximum execution time, and verbosity.

        Parameters:
            name (str): Name of the timer instance.
            Hz (int): Frequency in Hertz (Hz) at which the timer should run.
            max_sec (float): Maximum execution time in seconds. If exceeded, the timer will stop.
            verbose (bool): If True, print initialization messages.
        """
        self.name           = name
        self.Hz             = Hz
        self.max_sec        = max_sec
        self.verbose        = verbose
        self.sec_next       = 0.0
        self.sec_period     = 1.0 / self.Hz
        self.sec_elps       = 0.0
        self.sec_elps_prev  = 0.0
        self.sec_elps_diff  = 0.0
        self.sec_elps_loop  = 0.0 # exact esec 
        self.tick           = 0
        self.force_finish   = False
        self.delayed_flag   = False
        if self.verbose:
            print ("[%s] INITIALIZED WITH [%d]HZ. MAX_SEC:[%.1fsec]."
                % (self.name, self.Hz, self.max_sec))
    
    def start(self):
        """
        Start the timer by initializing the start time and resetting all time-related variables.

        Parameters:
            None
        """
        self.time_start     = datetime.now()
        self.sec_next       = 0.0
        self.sec_elps       = 0.0
        self.sec_elps_prev  = 0.0
        self.sec_elps_diff  = 0.0
        self.tick           = 0
        if self.verbose:
            print ("[%s] START ([%d]HZ. MAX_SEC:[%.1fsec])."
                % (self.name, self.Hz, self.max_sec))
    
    def finish(self):
        """
        Force finish the timer by setting the force_finish flag to True.
        This will cause the timer to consider itself finished immediately, regardless of elapsed time.

        Parameters:
            None
        """
        self.force_finish = True
    
    def is_finished(self):
        """
        Check if the timer has finished based on elapsed time or if force_finish is set.

        Parameters:
            None

        Returns:
            bool: True if the timer is finished (either by elapsed time exceeding max_sec or force_finish is True), False otherwise.
        """
        self.time_diff = datetime.now() - self.time_start
        self.sec_elps  = self.time_diff.total_seconds()
        if self.force_finish:
            return True
        if self.sec_elps > self.max_sec:
            return True
        else:
            return False
        
    def is_notfinished(self):
        """
        Check if the timer is still running, i.e., not finished.
        This method checks if the elapsed time is less than the maximum allowed time and if force_finish is not set.

        Parameters:
            None

        Returns:
            bool: True if the timer is still running (not finished), False if it has finished.
        """
        self.time_diff = datetime.now() - self.time_start
        self.sec_elps  = self.time_diff.total_seconds()
        if self.force_finish:
            return False
        if self.sec_elps > self.max_sec:
            return False
        else:
            return True
    
    def do_run(self):
        """
        Check if the timer should run based on the specified frequency (Hz).
        This method calculates the elapsed time since the last tick and determines if enough time has passed to execute the next tick.

        Parameters:
            None

        Returns:
            bool: True if the timer should run (i.e., enough time has passed since the last tick), False otherwise.
        """
        time.sleep(1e-6) # avoid busy waiting
        self.time_diff = datetime.now() - self.time_start
        self.sec_elps  = self.time_diff.total_seconds()
        if self.sec_elps > self.sec_next:
            self.sec_next = self.sec_next + self.sec_period
            self.tick     = self.tick + 1
            """ 
            Compute elapsed time since the last tick.
            """
            self.sec_elps_diff = self.sec_elps - self.sec_elps_prev
            self.sec_elps_prev = self.sec_elps
            """ 
            Check if the elapsed time since the last tick is significantly larger than the expected period.
            """
            if (self.sec_elps_diff > self.sec_period*1.5) and (self.Hz != 0):
                if self.verbose:
                    print ("[%s][%d][%.1fs] delayed! T:[%.1fms] But it took [%.1fms]. Acutally [%.1fms]"%
                           (self.name,self.tick, self.sec_elps, self.sec_period*1000.0, 
                            self.sec_elps_diff*1000.0,self.sec_elps_loop*1000.0))
                self.delayed_flag = True
            # Set the start time for the next run
            self.time_run_start = datetime.now()
            return True
        else:
            self.delayed_flag = False
            return False

    def end(self):
        """
        End the timer by calculating the elapsed time since the last tick and checking for delays.
        This method updates the elapsed time since the last tick and checks if it exceeds the expected period.
        """
        time_diff = datetime.now() - self.time_run_start
        self.sec_elps_loop = time_diff.total_seconds()
        if (self.sec_elps_loop > self.sec_period) and (self.Hz != 0):
            # Print all the time
            print ("[%s] is REALLY delayed! T:[%.1fms] BUT IT TOOK [%.1fms]"%
                   (self.name,self.sec_period*1000.0, self.sec_elps_loop*1000.0))

class RangeOfMotionTracker:
    """
    Tracks the range of motion (ROM) from arbitrary-dimensional motion data.

    This class accumulates data over time and computes the element-wise minimum,
    maximum, and range (max - min), which corresponds to the range of motion.

    Usage:
        rom = RangeOfMotionTracker()
        rom.update(data)  # Can be 1D, 2D, 3D, etc.
        result = rom.get()
        print(result['min'], result['max'], result['range'])

        # For multiple updates
        rom.update(data1)
        rom.update(data2)  # Accumulative update
        result = rom.get()

    Attributes:
        min_values (np.ndarray): Minimum observed values
        max_values (np.ndarray): Maximum observed values
        fitted (bool): Whether the tracker has been initialized
        shape (tuple): Shape of the input data
    """

    def __init__(self):
        """Initialize the RangeOfMotionTracker."""
        self.fitted = False
        self.min_values = None
        self.max_values = None
        self.shape = None

    def reset(self) -> None:
        """Reset the tracker to its initial, unfitted state."""
        self.fitted = False
        self.min_values = None
        self.max_values = None
        self.shape = None

    def update(self, data: np.ndarray) -> None:
        """
        Update the tracker with new motion data.

        Args:
            data (np.ndarray): Input data of any shape (1D, 2D, 3D, etc.)

        Raises:
            ValueError: If the input data has a different shape from previous inputs.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if not self.fitted:
            # Initialize with first data
            self.min_values = data.copy()
            self.max_values = data.copy()
            self.shape = data.shape
            self.fitted = True
        else:
            if data.shape != self.shape:
                raise ValueError(f"Data shape {data.shape} is inconsistent with the expected shape {self.shape}")
            
            self.min_values = np.minimum(self.min_values, data)
            self.max_values = np.maximum(self.max_values, data)

    def get(self) -> dict:
        """
        Retrieve the current range of motion data.

        Returns:
            dict: Dictionary containing:
                - 'min': Minimum values
                - 'max': Maximum values
                - 'range': Range of motion (max - min)

        Raises:
            ValueError: If the tracker has not received any data yet.
        """
        if not self.fitted:
            return None

        # Return 
        ret = {
            'min': self.min_values.copy(),
            'max': self.max_values.copy(),
            'center': (self.min_values+self.max_values)/2.0,
            'range': self.max_values-self.min_values,
        }
        return ret
    
def estimate_scalar_scale_vector_offset(X, Y):
    """
    Estimate a single scalar scale and a 3D vector offset to best align two sets of 3D points.

    This function fits the model: Y ≈ s * X + b, where:
        - s (float): scalar scale factor (applied equally to all dimensions)
        - b (np.ndarray of shape (3,)): vector offset added to each dimension

    Parameters:
        X (np.ndarray): Source point set, shape (N, 3)
        Y (np.ndarray): Target point set, shape (N, 3)

    Returns:
        s (float): Scalar scale factor
        b (np.ndarray): Offset vector of shape (3,)

    Example:
        >>> import numpy as np
        >>> X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        >>> Y = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        >>> s, b = estimate_scalar_scale_vector_offset(X, Y)
        >>> print(f"Scale: {s:.2f}, Offset: {b}")
        Scale: 2.00, Offset: [1. 2. 3.]

        >>> # Apply the transformation
        >>> Y_pred = s * X + b
        >>> print("Transformed X:")
        >>> print(Y_pred)
        [[1. 2. 3.]
         [3. 4. 5.]
         [5. 6. 7.]]
    """
    assert X.shape == Y.shape
    N = X.shape[0]

    # Step 1: solve for scalar scale using flattened data
    A = np.vstack([X.flatten(), np.ones(N * 3)]).T  # shape (3N, 2)
    y = Y.flatten()
    s, _ = np.linalg.lstsq(A, y, rcond=None)[0]

    # Step 2: compute offset as b = mean(Y - s * X) per dimension
    b = np.mean(Y - s * X, axis=0)

    return s, b

def get_ip_address():
    """
    Get the local IP address of the machine.
    """
    result = subprocess.run(['ipconfig','getifaddr','en0'],capture_output=True,text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return None
    
class VectorizedKalmanFilters:
    """
    VectorizedKalmanFilters is a class that implements a vectorized Kalman filter
    """
    def __init__(self, shape, dt=0.05, q=0.1, r=1.0):
        """
        Initializes the Kalman filter with the given parameters.

        Parameters:
            shape (tuple): The shape of the input data, e.g., (20, 2, 3) for 20 bones with 2 points each in 3D.
            dt (float): The time step for the Kalman filter.
            q (float): The process noise covariance.
            r (float): The measurement noise covariance.
        """
        self.shape = shape  # e.g., (20,2,3) → N = 120
        self.N = np.prod(shape)

        self.dt = dt
        self.F = np.array([[1, dt], [0, 1]])  # (2,2)
        self.H = np.array([[1, 0]])           # (1,2)
        self.Q = np.array([[q, 0], [0, q]])
        self.R = np.array([[r]])

        # Broadcast to (N, 2, 2), (N, 2, 1)
        self.Fs = np.broadcast_to(self.F, (self.N, 2, 2))
        self.Hs = np.broadcast_to(self.H, (self.N, 1, 2))
        self.Qs = np.broadcast_to(self.Q, (self.N, 2, 2))
        self.Rs = np.broadcast_to(self.R, (self.N, 1, 1))

        self.x_hat = np.zeros((self.N, 2, 1))  # initial states
        self.P_hat = np.tile(self.Q[None, :, :], (self.N, 1, 1))  # initial covariances

    def update(self, obs):  # obs: (20,2,3)
        """
        Updates the Kalman filter with the given observations.

        Parameters:
            obs (np.ndarray): The observations to update the filter with, shape (N, 2, 3).
        """
        obs_flat = obs.reshape(self.N)[:, None, None]  # (N, 1, 1)

        # Prediction
        x_hat_m = self.Fs @ self.x_hat                      # (N, 2, 1)
        P_hat_m = self.Fs @ self.P_hat @ self.Fs.transpose(0, 2, 1) + self.Qs  # (N, 2, 2)

        # Kalman Gain
        S = self.Hs @ P_hat_m @ self.Hs.transpose(0, 2, 1) + self.Rs  # (N, 1, 1)
        K = P_hat_m @ self.Hs.transpose(0, 2, 1) @ np.linalg.inv(S)   # (N, 2, 1)

        # Update
        y_diff = obs_flat - (self.Hs @ x_hat_m)  # (N,1,1)
        self.x_hat = x_hat_m + K @ y_diff
        self.P_hat = P_hat_m - K @ self.Hs @ P_hat_m

        return self.x_hat[:, 0, 0].reshape(self.shape)  # return estimated positions