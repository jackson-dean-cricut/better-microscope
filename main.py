import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
import csv
from datetime import datetime
import os
from typing import Optional, Tuple, List

class MeasurementSystem:
    def __init__(self):
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("Microscope Measurement System")
        
        # Camera variables
        self.cap = None
        self.camera_index = 0
        self.frame = None
        
        # Measurement variables
        self.calibration_factor = None  # mm per pixel
        self.measuring = False
        self.calibrating = False
        self.start_point = None
        self.current_point = None
        self.measurements: List[dict] = []
        self.current_filename = ""
        

        # Setup GUI
        self.setup_gui()
        
        # Initialize camera
        self.initialize_camera()

        # Load calibration factor from file
        self.load_calibration()
        
        # CSV setup
        self.csv_file = "measurements.csv"
        self.ensure_csv_exists()
        
        # Additional variables
        self.original_width = 0
        self.original_height = 0
        
    def setup_gui(self):
        # Create frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5, padx=5, fill=tk.X)
        
        # Camera controls
        ttk.Label(control_frame, text="Camera:").pack(side=tk.LEFT, padx=5)
        self.camera_var = tk.StringVar(value="0")
        camera_entry = ttk.Entry(control_frame, textvariable=self.camera_var, width=5)
        camera_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Connect", command=self.initialize_camera).pack(side=tk.LEFT, padx=5)
        
        # Filename entry
        ttk.Label(control_frame, text="Machine:").pack(side=tk.LEFT, padx=5)
        self.machine_var = tk.StringVar()
        machine_entry = ttk.Entry(control_frame, textvariable=self.machine_var, width=20)
        machine_entry.pack(side=tk.LEFT, padx=5)
        
        # Dropdown for selecting option
        ttk.Label(control_frame, text="Select Option:").pack(side=tk.LEFT, padx=5)
        self.option_var = tk.StringVar(value="inkframe_cal")  # Default value
        options = ["inkframe_cal", "inkframe_test", "trad_cal", "trad_test"]
        option_combobox = ttk.Combobox(control_frame, textvariable=self.option_var, values=options, state='readonly')
        option_combobox.pack(side=tk.LEFT, padx=5)
        
        # Mode buttons
        ttk.Button(control_frame, text="Calibrate", command=self.start_calibration).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Measure", command=self.start_measurement).pack(side=tk.LEFT, padx=5)
                
        # Calibration display
        self.calibration_display_var = tk.StringVar(value="Calibration: Not set")
        ttk.Label(control_frame, textvariable=self.calibration_display_var).pack(side=tk.TOP, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(self.root, textvariable=self.status_var)
        status_label.pack(side=tk.BOTTOM, pady=5)
        
        # Canvas for camera feed
        self.canvas = tk.Canvas(self.root, width=1280, height=960)
        self.canvas.pack(pady=5)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
    def initialize_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.camera_index = int(self.camera_var.get())
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise ValueError("Could not open camera")
                
            self.status_var.set("Status: Camera connected")
            self.update_camera()
        except Exception as e:
            self.status_var.set(f"Status: Camera error - {str(e)}")
    
    def ensure_csv_exists(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['machine', 'measurement_number', 'length_in', 'option', 'timestamp'])
    
    def update_camera(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                # Convert frame to RGB for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Store original dimensions
                self.original_height, self.original_width = frame_rgb.shape[:2]
                # Draw current measurement line if exists
                if self.start_point is not None and self.current_point is not None:
                    # Scale points back to original dimensions
                    start_scaled = (
                        int(self.start_point[0] * self.original_width / self.canvas.winfo_width()),
                        int(self.start_point[1] * self.original_height / self.canvas.winfo_height())
                    )
                    current_scaled = (
                        int(self.current_point[0] * self.original_width / self.canvas.winfo_width()),
                        int(self.current_point[1] * self.original_height / self.canvas.winfo_height())
                    )
                    cv2.line(frame_rgb, start_scaled, current_scaled, (255, 0, 0), 2)
                
                # Convert to PhotoImage
                self.frame_rgb = cv2.resize(frame_rgb, (1280, 960))
                self.photo = tk.PhotoImage(data=cv2.imencode('.ppm', self.frame_rgb)[1].tobytes())
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
        # Schedule next update
        self.root.after(10, self.update_camera)
    
    def calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        # Scale points to original dimensions before calculating distance
        p1_scaled = (
            p1[0] * self.original_width / self.canvas.winfo_width(),
            p1[1] * self.original_height / self.canvas.winfo_height()
        )
        p2_scaled = (
            p2[0] * self.original_width / self.canvas.winfo_width(),
            p2[1] * self.original_height / self.canvas.winfo_height()
        )
        # Calculate distance
        distance = np.sqrt((p2_scaled[0] - p1_scaled[0])**2 + (p2_scaled[1] - p1_scaled[1])**2)
        
        # Determine sign based on direction
        x_dist = np.abs(p2_scaled[0] - p1_scaled[0])
        y_dist = np.abs(p2_scaled[1] - p1_scaled[1])
        if (x_dist > y_dist):
            if p2_scaled[0] < p1_scaled[0]:  # Right-to-left
                return -distance
        else:
            if p2_scaled[1] < p1_scaled[1]:
                return -distance
        return distance  # Positive if left-to-right and top-to-bottom
    
    def on_click(self, event):
        if self.calibrating or self.measuring:
            self.start_point = (event.x, event.y)
            self.current_point = self.start_point
    
    def on_drag(self, event):
        if (self.calibrating or self.measuring) and self.start_point:
            self.current_point = (event.x, event.y)
            # Update status with current measurement
            pixels = self.calculate_distance(self.start_point, self.current_point)
            if self.calibration_factor and self.measuring:
                inches = pixels * self.calibration_factor
                self.status_var.set(f"Current measurement: {inches:.4f} in")
            else:
                self.status_var.set(f"Current pixels: {pixels:.1f}")
    
    def on_release(self, event):
        if self.start_point and (self.calibrating or self.measuring):
            end_point = (event.x, event.y)
            pixels = self.calculate_distance(self.start_point, end_point)
            
            if self.calibrating:
                self.handle_calibration(pixels)
            elif self.measuring:
                self.handle_measurement(pixels)
    
    def handle_calibration(self, pixels: float):
        # Create calibration dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Calibration")
        
        ttk.Label(dialog, text="Enter known length (in):").pack(pady=5)
        length_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=length_var).pack(pady=5)
        
        def confirm():
            try:
                known_length = float(length_var.get())
                self.calibration_factor = known_length / pixels
                self.status_var.set(f"Status: Calibrated - {self.calibration_factor:.4f} in/pixel")
                self.calibration_display_var.set(f"Calibration: {self.calibration_factor:.4f} in/pixel")
                
                # Save calibration to file
                with open("calibration.txt", "w") as f:
                    f.write(str(self.calibration_factor))
                
                dialog.destroy()
                self.calibrating = False
            except ValueError:
                self.status_var.set("Status: Invalid calibration value")
        
        ttk.Button(dialog, text="Confirm", command=confirm).pack(pady=5)
    
    def handle_measurement(self, pixels: float):
        if not self.calibration_factor:
            self.status_var.set("Status: Please calibrate first")
            return
            
        if not self.machine_var.get():
            self.status_var.set("Status: Please enter a machine")
            return
            
        length_in = pixels * self.calibration_factor
        measurement_number = len([m for m in self.measurements if m['machine'] == self.machine_var.get() and m['option'] == self.option_var.get()]) + 1
        
        # Create measurement dict
        measurement = {
            'machine': self.machine_var.get(),
            'measurement_number': measurement_number,
            'length_in': length_in,
            'option': self.option_var.get(),  # Add selected option
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Confirmation dialog
        def confirm_save(event):
            # Save to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    measurement['machine'],
                    measurement['measurement_number'],
                    f"{measurement['length_in']:.3f}",
                    measurement['option'],  # Include option in CSV
                    measurement['timestamp']
                ])
            
            # Save the current frame as an image
            if self.frame is not None:
                # Create directory if it doesn't exist
                pictures_dir = os.path.join("pictures", self.machine_var.get(), self.option_var.get())
                os.makedirs(pictures_dir, exist_ok=True)
                
                # Save the image using frame_rgb
                image_filename = os.path.join(pictures_dir, f"{measurement_number}.png")
                cv2.imwrite(image_filename, self.frame_rgb)
                
                self.status_var.set(f"Status: Saved measurement {measurement_number}: {length_in:.3f} in and image saved as {image_filename}")
            self.measurements.append(measurement)
        
        def on_save():
            confirm_save(None)
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Confirm Save")
        dialog.geometry(f"+{self.root.winfo_pointerx()}+{self.root.winfo_pointery()}")
        
        ttk.Label(dialog, text="Confirm Save Measurement:").pack(pady=5)
        ttk.Label(dialog, text=str(measurement)).pack(pady=5)
        
        ttk.Button(dialog, text="Save", command=on_save).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(dialog, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5, pady=5)
    
    def start_calibration(self):
        self.calibrating = True
        self.measuring = False
        self.status_var.set("Status: Click and drag to calibrate")
    
    def start_measurement(self):
        if not self.calibration_factor:
            self.status_var.set("Status: Please calibrate first")
            return
        self.measuring = True
        self.calibrating = False
        self.status_var.set("Status: Click and drag to measure")
    
    def load_calibration(self):
        try:
            with open("calibration.txt", "r") as f:
                self.calibration_factor = float(f.read().strip())
                self.calibration_display_var.set(f"Calibration: {self.calibration_factor:.4f} in/pixel")
        except (FileNotFoundError, ValueError):
            self.calibration_factor = None  # Default to None if file not found or invalid
            self.calibration_display_var.set("Calibration: Not set")
    
    def run(self):
        self.root.mainloop()
        if self.cap is not None:
            self.cap.release()

if __name__ == "__main__":
    app = MeasurementSystem()
    app.run()