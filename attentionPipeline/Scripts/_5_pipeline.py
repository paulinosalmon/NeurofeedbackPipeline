import tkinter as tk
import threading
import numpy as np
import os
import sys
import signal
from queue import Queue
from _1_data_preprocessing import run_data_preprocessing
from _2_artifact_rejection import run_artifact_rejection
from _3_classifier import run_classifier, calculate_mean_error_rate_cv
from _4_feedback_generator import run_feedback_generator, realtime_graph

def handle_keyboard_interrupt(signal, frame):
    print("Interrupt received, evaluating model before shutting down.")
    X_test = np.load(os.path.join("../data/", "X_test.npy"))
    y_test = np.load(os.path.join("../data/", "y_test.npy"))
    calculate_mean_error_rate_cv(X_test, y_test)
    print("Evaluation saved in ../reports/. Shutting down.")
    sys.exit(0)

def update_output(gui_queue, text_widget, image_label=None):
    while True:
        message = gui_queue.get()
        
        # Check if the message contains an image
        if isinstance(message, tuple) and isinstance(message[0], ImageTk.PhotoImage):
            # Unpack the tuple
            tk_image, alpha_log = message
            # Update the image label with the new image
            if image_label is not None:
                image_label.config(image=tk_image)
                image_label.image = tk_image  # Keep a reference
            # Add the alpha_log to the text widget
            text_widget.insert(tk.END, alpha_log + '\n')
            text_widget.see(tk.END)
        else:
            # Just a text message, update the text widget
            text_widget.insert(tk.END, message + '\n')
            text_widget.see(tk.END)

def setup_gui():
    root = tk.Tk()
    root.title("EEG Processing Pipeline")

    # Create a frame for text widgets on the left side
    left_frame = tk.Frame(root)
    left_frame.pack(side=tk.LEFT, fill=tk.Y)

    # Text widgets for different stages, placed inside the left frame
    text_data_preprocessing = tk.Text(left_frame, height=10, width=50)
    text_data_preprocessing.pack()
    text_artifact_rejection = tk.Text(left_frame, height=10, width=50)
    text_artifact_rejection.pack()
    text_classifier = tk.Text(left_frame, height=10, width=50)
    text_classifier.pack()
    text_feedback = tk.Text(left_frame, height=10, width=50)
    text_feedback.pack()

    # Right frame for image display
    right_frame = tk.Frame(root)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Label for the image within the right frame, centered
    label_image = tk.Label(right_frame)
    label_image.pack(anchor='center', expand=True)

    # Queues for GUI updates and data transfer
    queue_gui_preprocessing = Queue()
    queue_gui_artifact = Queue()
    queue_gui_classifier = Queue()
    queue_gui_feedback = Queue()
    queue_graph_update = Queue()  # Queue for graph updates

    queue_artifact = Queue()  # Artifact rejection to classifier bridge
    queue_classifier = Queue()  # Classifier to feedback bridge

    # realtime_graph(root, right_frame, queue_gui_feedback)

    # Threads for updating GUI based on different stages
    threading.Thread(target=update_output, args=(queue_gui_preprocessing, text_data_preprocessing), daemon=True).start()
    threading.Thread(target=update_output, args=(queue_gui_artifact, text_artifact_rejection), daemon=True).start()
    threading.Thread(target=update_output, args=(queue_gui_classifier, text_classifier), daemon=True).start()
    threading.Thread(target=update_output, args=(queue_gui_feedback, text_feedback, label_image), daemon=True).start()

    # Threads for different stages of the pipeline
    threading.Thread(target=run_data_preprocessing, args=(queue_gui_preprocessing, queue_artifact), daemon=True).start()
    threading.Thread(target=run_artifact_rejection, args=(queue_artifact, queue_gui_artifact), daemon=True).start()
    threading.Thread(target=run_classifier, args=(queue_gui_classifier, queue_artifact, queue_classifier), daemon=True).start()
    threading.Thread(target=run_feedback_generator, args=(queue_gui_feedback, queue_classifier, label_image, queue_graph_update), daemon=True).start()

    realtime_graph(root, right_frame, queue_graph_update)

    root.mainloop()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    setup_gui()