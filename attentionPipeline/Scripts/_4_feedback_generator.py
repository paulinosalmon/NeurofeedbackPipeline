import os
import numpy as np
from PIL import Image, ImageTk, ImageOps
from queue import Queue
import random
import tkinter as tk
import time

from datetime import datetime

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def asymmetric_sigmoid_transfer(output, inflection_point=0.6, lower_bound=0.17, upper_bound=0.98):
    """
    Custom sigmoid function that maps the output to a specific range with an inflection point.
    
    Parameters:
    - output: Input value or array
    - inflection_point: The point at which the sigmoid curve has its midpoint
    - lower_bound: The lower bound of the output range
    - upper_bound: The upper bound of the output range
    
    Returns:
    - Sigmoid output mapped between lower_bound and upper_bound
    """
    sigmoid_range = upper_bound - lower_bound

    # Adjust the output to shift the sigmoid curve
    adjusted_output = output - inflection_point

    sigmoid_output = 1 / (1 + np.exp(-adjusted_output))
    return lower_bound + sigmoid_range * sigmoid_output

def adjust_pixel_range(image_array):
    # Adjust pixel values to have 10% in the ranges 0-46 and 210-255
    quantiles = np.quantile(image_array, [0.1, 0.9])
    low, high = quantiles[0], quantiles[1]
    image_array = np.where(image_array < low, np.interp(image_array, [0, low], [0, 46]), image_array)
    image_array = np.where(image_array > high, np.interp(image_array, [high, 255], [210, 255]), image_array)
    return image_array

def update_image(image_label, image_stimuli_path='../imageStimuli', alpha=0.5):
    # Select random categories for face and scene
    face_category = random.choice(['female', 'male'])
    scene_category = random.choice(['indoor', 'outdoor'])

    # Choose random images from the selected categories
    face_image_path = os.path.join(image_stimuli_path, face_category, random.choice(os.listdir(os.path.join(image_stimuli_path, face_category))))
    scene_image_path = os.path.join(image_stimuli_path, scene_category, random.choice(os.listdir(os.path.join(image_stimuli_path, scene_category))))

    # Load and convert images to grayscale
    image1 = Image.open(face_image_path).convert('L')
    image2 = Image.open(scene_image_path).convert('L')

    # Resize images and blend them using the alpha value
    image1 = ImageOps.fit(image1, (175, 175), Image.Resampling.LANCZOS)
    image2 = ImageOps.fit(image2, (175, 175), Image.Resampling.LANCZOS)
    mixed_image = Image.blend(image1, image2, alpha)

    # Adjust pixel range
    mixed_image_array = np.array(mixed_image)
    mixed_image_array = adjust_pixel_range(mixed_image_array)
    mixed_image = Image.fromarray(mixed_image_array.astype('uint8'))

    # Convert to a format compatible with Tkinter and update the image label
    tk_image = ImageTk.PhotoImage(mixed_image)
    image_label.config(image=tk_image)
    image_label.image = tk_image  # Keep a reference to the image

def realtime_graph(root, right_frame, queue_graph_update):
    # Create a figure for the plot
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    # Initialize lists to store the data
    x_data = []
    y_data = []

    # Function to update the graph
    def update_graph():
        nonlocal x_data, y_data
        # Check if new data is available to update the graph
        while not queue_graph_update.empty():
            # Get the latest classifier output and add it to the data lists
            classifier_output = queue_graph_update.get()
            x_data.append(len(x_data) + 1)  # Increment trial number
            y_data.append(classifier_output)  # Append the classifier output
            
            # Clear the previous plot and plot the updated data
            ax.clear()
            ax.plot(x_data, y_data, '-o', color='blue')
            ax.axhline(0, color='black', linewidth=1)  # Add a horizontal line at y=0
            ax.axvline(0, color='black', linewidth=1)  # Add a vertical line at x=0
            ax.set_title("Feedback Observation (Task-Relevant Category: Scenes)")
            ax.set_xlabel("Trial Number")
            ax.set_ylabel("Real-Time Category Decoding")
            ax.set_xlim(left=max(0, len(x_data) - 50), right=len(x_data) + 1)
            ax.set_ylim(min(y_data)-0.1, max(y_data)+0.1)

            # Redraw the canvas with the new data
            canvas.draw()

        # Schedule the function to run again after a short delay for continuous updating
        root.after(100, update_graph)

    # Add the plot to the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Call the update function to start the updating process
    update_graph()

def run_feedback_generator(queue_gui, queue_classifier, label_image, queue_graph_update):
    queue_gui.put(f"[{datetime.now()}] [Feedback Generator] Feedback Generation Initiated.")
    image_stimuli_path = '../imageStimuli'  # Path to the image stimuli

    while True:
        try:
            classifier_output = queue_classifier.get(block=True)
            alpha = asymmetric_sigmoid_transfer(classifier_output)
            queue_gui.put(f"[{datetime.now()}] [Feedback Generator] Received alpha from classifier stream: {alpha}")

            # Generate the image based on the received visibility score
            tk_image = update_image(label_image, image_stimuli_path, alpha=alpha)
            queue_gui.put(f"[{datetime.now()}] [Feedback Generator] Image Generated.")

            # Put the visibility score into the queue for the graph update
            queue_graph_update.put(alpha)

        except Exception as e:
            queue_gui.put(f"[{datetime.now()}] [Feedback Generator] Error: {e}")