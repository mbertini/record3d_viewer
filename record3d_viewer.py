import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

# Attempt to import OpenEXR and Imath
try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False

class ImageDepthPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image (JPEG) and Depth (EXR) Player")
        self.root.geometry("1400x800")  # Increased window size
        self.root.minsize(1200, 700)  # Set minimum size

        if not OPENEXR_AVAILABLE:
            messagebox.showerror("Dependency Error",
                                 "OpenEXR library not found. This library is required to read EXR files.\n"
                                 "Please install it (e.g., 'pip install OpenEXR') and ensure its dependencies are met.\n"
                                 "On Linux: sudo apt-get install libopenexr-dev\n"
                                 "On Windows: Use precompiled binaries from https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr\n"
                                 "On macOS: brew install openexr")
            # master.destroy() # Or disable functionality
            # return # Early exit if critical dependency is missing

        # State variables
        self.folder_path = None
        self.image_folder = None
        self.depth_folder = None
        self.jpg_files = []
        self.exr_files = []
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.play_speed = 33  # ms between frames (~30 fps by default)
        self.depth_contrast = 1.0  # Contrast multiplier for depth visualization
        self.current_depth_data = None  # Cache current depth data

        # Create the main UI
        self.create_ui()

    def create_ui(self):
        # Top frame for folder selection
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Button(top_frame, text="Select Folder", command=self.select_folder).pack(side="left", padx=5)
        self.folder_label = ttk.Label(top_frame, text="No folder selected")
        self.folder_label.pack(side="left", padx=5, fill="x", expand=True)

        # Frame display area
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left side: Depth view with side contrast control
        depth_container = ttk.Frame(self.display_frame)
        depth_container.pack(side="left", fill="both", expand=True, padx=(0, 5))

        depth_main_frame = ttk.Frame(depth_container)
        depth_main_frame.pack(fill="both", expand=True)

        # Depth canvas (same size as image canvas)
        self.depth_canvas = tk.Canvas(depth_main_frame, bg="black", width=576, height=576)
        self.depth_canvas.pack(side="left", fill="both", expand=True)

        # Vertical contrast control on the right side of depth view
        contrast_frame = ttk.Frame(depth_main_frame, width=80)
        contrast_frame.pack(side="right", fill="y", padx=(5, 0))
        contrast_frame.pack_propagate(False)  # Maintain fixed width

        ttk.Label(contrast_frame, text="Contrast", font=("Arial", 8)).pack(pady=(0, 5))

        self.contrast_var = tk.DoubleVar(value=1.0)

        # Create a frame for the slider to better control its behavior
        slider_frame = ttk.Frame(contrast_frame)
        slider_frame.pack(fill="both", expand=True, pady=(0, 5))

        self.contrast_slider = ttk.Scale(slider_frame, from_=0.1, to=5.0,
                                         orient="vertical", variable=self.contrast_var,
                                         command=self.update_depth_contrast)
        self.contrast_slider.pack(fill="both", expand=True)

        self.contrast_label = ttk.Label(contrast_frame, text="1.0x", font=("Arial", 8))
        self.contrast_label.pack(pady=(0, 5))

        # Histogram equalization checkbox
        self.hist_eq_var = tk.BooleanVar(value=False)
        self.hist_eq_check = ttk.Checkbutton(contrast_frame, text="Hist EQ",
                                             variable=self.hist_eq_var,
                                             command=self.update_depth_display)
        self.hist_eq_check.pack()

        # Right side: Image view
        self.image_canvas = tk.Canvas(self.display_frame, bg="black", width=576, height=576)
        self.image_canvas.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # Controls frame
        controls_frame = ttk.Frame(self.root, padding=10)
        controls_frame.pack(fill="x", side="bottom")

        # Frame slider
        self.slider_label = ttk.Label(controls_frame, text="Frame: 0/0")
        self.slider_label.pack(side="top", pady=5)

        slider_frame = ttk.Frame(controls_frame)
        slider_frame.pack(fill="x", pady=5)

        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=0, orient="horizontal",
                                      command=self.slider_changed)
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=5)

        # Navigation buttons
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(fill="x", pady=5)

        self.prev_button = ttk.Button(nav_frame, text="⏮", command=self.prev_frame, width=5)
        self.prev_button.pack(side="left", padx=5)

        self.play_button = ttk.Button(nav_frame, text="▶", command=self.toggle_play, width=10)
        self.play_button.pack(side="left", padx=5)

        self.next_button = ttk.Button(nav_frame, text="⏭", command=self.next_frame, width=5)
        self.next_button.pack(side="left", padx=5)

        # Speed controls
        speed_frame = ttk.Frame(controls_frame)
        speed_frame.pack(fill="x", pady=5)

        ttk.Label(speed_frame, text="Speed:").pack(side="left", padx=5)
        speed_values = ["0.5x", "1x", "2x", "5x"]
        self.speed_combo = ttk.Combobox(speed_frame, values=speed_values, width=5, state="readonly")
        self.speed_combo.current(1)  # Default to 1x
        self.speed_combo.pack(side="left", padx=5)
        self.speed_combo.bind("<<ComboboxSelected>>", self.change_speed)

        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<space>", lambda e: self.toggle_play())

        # Initialize disabled controls
        self.update_controls_state(False)

    def update_depth_contrast(self, value=None):
        """Update depth contrast and refresh display."""
        new_contrast = self.contrast_var.get()

        # check value is still within limits
        new_contrast = max(0.1, min(5.0, new_contrast))

        # update only if value changed
        if self.depth_contrast != new_contrast:
            self.depth_contrast = new_contrast
            self.contrast_label.config(text=f"{self.depth_contrast:.1f}x")

            # if we have fresh depth data, update the display
            if self.current_depth_data is not None:
                self.update_depth_display()

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Select folder containing image and depth subfolders")
        if not folder_path:
            return

        self.folder_path = folder_path
        self.folder_label.config(text=f"Selected: {folder_path}")

        # Check subdirectories
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        if len(subdirs) < 2:
            self.show_error("The selected folder should contain at least 2 subdirectories")
            return

        # Look for image and depth folders
        jpg_files = []
        exr_files = []

        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir)
            files = os.listdir(subdir_path)

            jpg_count = len([f for f in files if f.endswith('.jpeg') or f.endswith('.jpg')])
            exr_count = len([f for f in files if f.endswith('.exr')])

            if jpg_count > exr_count:
                self.image_folder = subdir_path
                jpg_files = sorted([f for f in files if f.endswith('.jpeg') or f.endswith('.jpg')],
                                   key=lambda x: int(os.path.splitext(x)[0]))
            elif exr_count > 0:
                self.depth_folder = subdir_path
                exr_files = sorted([f for f in files if f.endswith('.exr')],
                                   key=lambda x: int(os.path.splitext(x)[0]))

        if not self.image_folder or not self.depth_folder:
            self.show_error(
                "Could not find image and depth folders. Make sure one folder contains JPEG files and another contains EXR files.")
            return

        # Match files by number
        self.jpg_files = jpg_files
        self.exr_files = exr_files

        # Determine total frames (minimum of both file types)
        self.total_frames = min(len(self.jpg_files), len(self.exr_files))

        if self.total_frames == 0:
            self.show_error("No matching image-depth pairs found")
            return

        # Update slider
        self.frame_slider.config(to=self.total_frames - 1)
        self.update_slider_label()

        # Enable controls
        self.update_controls_state(True)

        # Load first frame
        self.current_frame = 0
        self.show_current_frame()

    def update_controls_state(self, enabled):
        state = "normal" if enabled else "disabled"
        self.frame_slider.config(state=state)
        self.prev_button.config(state=state)
        self.play_button.config(state=state)
        self.next_button.config(state=state)
        self.speed_combo.config(state="readonly" if enabled else "disabled")
        self.contrast_slider.config(state=state)
        self.hist_eq_check.config(state=state)

    def show_error(self, message):
        tk.messagebox.showerror("Error", message)

    def update_slider_label(self):
        self.slider_label.config(text=f"Frame: {self.current_frame + 1}/{self.total_frames}")

    def slider_changed(self, value):
        # Convert string to int and handle potential floating point
        frame = int(float(value))
        if frame != self.current_frame:
            self.current_frame = frame
            self.show_current_frame()
            self.update_slider_label()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.set(self.current_frame)
            self.show_current_frame()
            self.update_slider_label()

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.set(self.current_frame)
            self.show_current_frame()
            self.update_slider_label()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_button.config(text="⏸" if self.is_playing else "▶")

        if self.is_playing:
            self.play_frames()

    def play_frames(self):
        if not self.is_playing:
            return

        # Check if we've reached the end
        if self.current_frame >= self.total_frames - 1:
            self.current_frame = 0

        # Move to next frame
        self.next_frame()

        # Schedule next frame
        self.root.after(self.play_speed, self.play_frames)

    def change_speed(self, event=None):
        speed = self.speed_combo.get()
        if speed == "0.5x":
            self.play_speed = 66  # 60 fps
        elif speed == "1x":
            self.play_speed = 33   # ~30 fps - Record3D default
        elif speed == "2x":
            self.play_speed = 16  # 15 fps
        elif speed == "5x":
            self.play_speed = 7  # 6 fps

    def show_current_frame(self):
        if self.current_frame >= self.total_frames:
            return

        # Load JPEG image
        jpg_path = os.path.join(self.image_folder, self.jpg_files[self.current_frame])
        image = cv2.imread(jpg_path)
        if image is None:
            self.show_error(f"Could not load image: {jpg_path}")
            return

        # Convert from BGR to RGB for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load EXR depth map
        exr_path = os.path.join(self.depth_folder, self.exr_files[self.current_frame])
        depth_map = self.load_exr(exr_path)
        if depth_map is None:
            self.show_error(f"Could not load depth map: {exr_path}")
            return

        # Cache the raw depth data
        self.current_depth_data = depth_map

        # Update both displays
        self.update_image_display(image)
        self.update_depth_display()

    def update_image_display(self, image):
        """Update the image canvas with the given image."""
        # Resize to fit canvas while maintaining aspect ratio
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:  # Skip if canvas not yet realized
            # Resize image
            h, w = image.shape[:2]
            aspect = w / h

            if canvas_width / canvas_height > aspect:
                new_h = canvas_height
                new_w = int(new_h * aspect)
            else:
                new_w = canvas_width
                new_h = int(new_w / aspect)

            image = cv2.resize(image, (new_w, new_h))

        # Convert to PhotoImage
        self.image_photo = ImageTk.PhotoImage(image=Image.fromarray(image))

        # Update canvas
        self.image_canvas.delete("all")

        # Center image in canvas
        img_x = (self.image_canvas.winfo_width() - self.image_photo.width()) // 2
        img_y = (self.image_canvas.winfo_height() - self.image_photo.height()) // 2

        self.image_canvas.create_image(max(0, img_x), max(0, img_y), anchor="nw", image=self.image_photo)

    def update_depth_display(self):
        """Update the depth canvas with current depth data and contrast setting."""
        if self.current_depth_data is None:
            return

        # Apply contrast and histogram equalization to normalize depth map for display
        depth_map = self.normalize_depth_map(
            self.current_depth_data,
            self.depth_contrast,
            self.hist_eq_var.get()
        )

        # Convert to grayscale image
        depth_img = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)

        # Resize to fit canvas while maintaining aspect ratio
        canvas_width = self.depth_canvas.winfo_width()
        canvas_height = self.depth_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:  # Skip if canvas not yet realized
            h, w = depth_img.shape[:2]
            aspect = w / h

            if canvas_width / canvas_height > aspect:
                new_h = canvas_height
                new_w = int(new_h * aspect)
            else:
                new_w = canvas_width
                new_h = int(new_w / aspect)

            depth_img = cv2.resize(depth_img, (new_w, new_h))

        # Convert to PhotoImage
        self.depth_photo = ImageTk.PhotoImage(image=Image.fromarray(depth_img))

        # Update canvas
        self.depth_canvas.delete("all")

        # Center image in canvas
        depth_x = (self.depth_canvas.winfo_width() - self.depth_photo.width()) // 2
        depth_y = (self.depth_canvas.winfo_height() - self.depth_photo.height()) // 2

        self.depth_canvas.create_image(max(0, depth_x), max(0, depth_y), anchor="nw", image=self.depth_photo)

    def normalize_depth_map(self, depth_map, contrast=1.0, use_hist_eq=False):
        """Normalize depth map to 0-255 range with contrast adjustment and optional histogram equalization."""
        if depth_map is None:
            return np.zeros((100, 100), dtype=np.uint8)

        # Handle NaN and infinity values
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

        # Remove any remaining invalid values
        depth_map = np.where(np.isfinite(depth_map), depth_map, 0)

        # Get valid (non-zero) values for normalization
        valid_mask = depth_map > 0
        if not np.any(valid_mask):
            # If all values are zero or invalid, return zeros
            return np.zeros_like(depth_map, dtype=np.uint8)

        valid_values = depth_map[valid_mask]
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)

        if max_val <= min_val:
            # If all valid values are the same, return a constant image
            return np.full_like(depth_map, 128, dtype=np.uint8)

        # Normalize to 0-1 range first
        normalized = np.zeros_like(depth_map, dtype=np.float32)
        normalized[valid_mask] = (valid_values - min_val) / (max_val - min_val)

        # Apply contrast adjustment
        # Contrast > 1 increases contrast, < 1 decreases it
        # Formula: output = ((input - 0.5) * contrast) + 0.5
        contrast_adjusted = ((normalized - 0.5) * contrast) + 0.5

        # Clamp to valid range
        contrast_adjusted = np.clip(contrast_adjusted, 0.0, 1.0)

        # Convert to 0-255 range
        result = (contrast_adjusted * 255.0).astype(np.uint8)

        # Apply histogram equalization if requested
        if use_hist_eq:
            # Only apply histogram equalization to non-zero areas
            if np.any(valid_mask):
                # Create a mask for histogram equalization
                eq_result = cv2.equalizeHist(result)
                # Only apply equalization where we have valid depth data
                result = np.where(valid_mask, eq_result, result)

        return result

    def load_exr(self, path):
        """Load an EXR file and return it as a numpy array."""
        try:
            # Open the input file
            exr_file = OpenEXR.InputFile(path)

            # Get the header and extract relevant information
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Print available channels for debugging
            available_channels = list(header['channels'].keys())
            print(f"Loading {path}, available channels: {available_channels}")

            # Try different channel combinations for depth data
            # Common depth channel names
            depth_channels = ["Z", "depth", "Depth", "DEPTH", "R", "G", "B", "Y", "A"]
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

            depth_data = None
            used_channel = None

            # First, try specific depth channels
            for channel in depth_channels:
                if channel in available_channels:
                    try:
                        str_data = exr_file.channel(channel, pixel_type)
                        depth_array = np.frombuffer(str_data, dtype=np.float32)
                        depth_data = depth_array.reshape(height, width)
                        used_channel = channel
                        break
                    except Exception as e:
                        print(f"Failed to read channel {channel}: {e}")
                        continue

            # If no specific channel worked, try the first available channel
            if depth_data is None and len(available_channels) > 0:
                channel = available_channels[0]
                try:
                    str_data = exr_file.channel(channel, pixel_type)
                    depth_array = np.frombuffer(str_data, dtype=np.float32)
                    depth_data = depth_array.reshape(height, width)
                    used_channel = channel
                except Exception as e:
                    print(f"Failed to read first channel {channel}: {e}")

            if depth_data is not None:
                print(f"Successfully loaded channel '{used_channel}', shape: {depth_data.shape}, "
                      f"range: [{np.min(depth_data):.3f}, {np.max(depth_data):.3f}], "
                      f"valid values: {np.sum(np.isfinite(depth_data))}/{depth_data.size}")

            return depth_data

        except Exception as e:
            print(f"Error loading EXR file {path}: {e}")
            return None


def main():
    root = tk.Tk()
    app = ImageDepthPlayer(root)
    root.mainloop()


if __name__ == "__main__":
    main()