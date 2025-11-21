import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageDraw, ImageCms, ImageGrab
import os
import tempfile
import gc
import threading
import math
import time
import sys
import platform
import io
import copy
import queue
import itertools
from collections import OrderedDict, namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

# --- Third-Party and OS-Specific Imports with Graceful Fallbacks ---

# For drag-and-drop functionality
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_ENABLED = True
except ImportError:
    DND_ENABLED = False
    print("WARNING: tkinterdnd2 not found. Drag-and-drop will be disabled.")

# For Windows-specific features like printing and advanced device info
WIN32_AVAILABLE = False
if platform.system() == "Windows":
    try:
        import win32api
        import win32print
        import win32con
        import win32ui
        import win32gui
        import pywintypes
        import ctypes
        from ctypes import wintypes
        import psutil # psutil is cross-platform but used here with Windows features
        WIN32_AVAILABLE = True
        print("INFO: Windows-specific features (printing, resource monitoring) enabled.")
    except ImportError:
        print("WARNING: pywin32 or psutil not found. Windows-specific features will be disabled.")
else:
    print("INFO: Running on a non-Windows OS. Printing and advanced features are disabled.")


# For optional GPU acceleration and Gamut Warning
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("INFO: NumPy found. Gamut warning feature is enabled.")
except ImportError:
    print("INFO: numpy not found. Gamut warning feature is disabled.")


# For optional GPU acceleration via OpenCL
GPU_ACCELERATION = False
if WIN32_AVAILABLE and NUMPY_AVAILABLE:
    try:
        import pyopencl as cl
        GPU_ACCELERATION = True
        print("INFO: OpenCL found. GPU acceleration is available.")
    except ImportError:
        print("INFO: pyopencl not found. GPU acceleration is disabled.")

# --- Constants ---
# UPDATE: Reduced default cache sizes to lower the baseline memory footprint,
# addressing concerns about holding too many image variants in memory.
MASTER_CACHE_SIZE = 20
PROCESSED_CACHE_SIZE = 40 # Cache for color-corrected images before final resize
THUMBNAIL_CACHE_SIZE = 200
DEFAULT_PHOTO_WIDTH_MM = 35
DEFAULT_PHOTO_HEIGHT_MM = 45
DEFAULT_PHOTO_COPIES = 1
DEFAULT_BRIGHTNESS = 0
DEFAULT_RESIZE_MODE = "crop"
DEFAULT_MARGIN_MM = 10
DEFAULT_INTERVAL_MM = 5
RESIZE_HANDLE_SIZE = 8 # Size of the resize handles in pixels
GAMUT_WARNING_COLOR = (255, 0, 128) # Bright magenta for gamut warning overlay

# Configuration for handling large images and memory
MAX_IMAGE_FILE_SIZE_MB = 100
MAX_IMAGE_DIMENSION_PX = 12000
PREVIEW_DOWNSCALE_DIMENSION_PX = 3000
MEMORY_WARNING_THRESHOLD_MB = 1024 # 1 GB

# UPDATE: Defined structured cache keys to replace unwieldy nested tuples.
# This improves readability and maintainability, resolving the "Cache Key Complexity" issue.
ProcessedCacheKey = namedtuple('ProcessedCacheKey', ['path', 'brightness', 'profile_id', 'intent'])
ThumbnailCacheKey = namedtuple('ThumbnailCacheKey', ['path', 'brightness', 'profile_id', 'intent', 'width', 'height', 'gamut_warning'])


# --- Helper Functions and Classes ---

def resource_path(relative_path):
    """ Get absolute path to a resource, for PyInstaller compatibility. """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class Tooltip:
    """ Simple tooltip class to provide hints for widgets. """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        """Create and show the tooltip window near the mouse cursor."""
        if self.tooltip_window:
            return
        x = event.x_root + 20
        y = event.y_root + 10

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        """Destroy the tooltip window."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

class Photo:
    """ Encapsulates all data and settings for a single photo. """
    _id_counter = 0
    def __init__(self, path, width, height, copies, brightness, resize_mode, original_width, original_height, icc_profile_data):
        self.id = Photo._id_counter
        Photo._id_counter += 1
        self.path = path
        self.width = width
        self.height = height
        self.copies = copies
        self.brightness = brightness
        self.resize_mode = resize_mode
        self.original_width = original_width
        self.original_height = original_height
        self.icc_profile_data = icc_profile_data

    def __repr__(self):
        return f"Photo(id={self.id}, path='{os.path.basename(self.path)}')"

    def get_settings(self):
        """ Returns a dictionary of the photo's current settings. """
        return {
            "width": self.width,
            "height": self.height,
            "copies": self.copies,
            "brightness": self.brightness,
            "resize_mode": self.resize_mode
        }

    def update_settings(self, settings_dict):
        """ Updates photo attributes from a settings dictionary. """
        for key, value in settings_dict.items():
            setattr(self, key, value)

class EditPhotoDialog(tk.Toplevel):
    """ A modal dialog for cropping a photo, with move and resize functionality. """
    def __init__(self, parent, photo_path):
        super().__init__(parent)
        self.withdraw()
        self.transient(parent)
        self.grab_set()
        self.title("Edit Photo")
        self.result = None
        self.parent = parent
        self.aspect_ratio = None

        try:
            self.iconbitmap(resource_path("neka_icon.ico"))
        except tk.TclError:
            print("INFO: Icon 'neka_icon.ico' not found for edit dialog.")

        try:
            with Image.open(photo_path) as img:
                img.thumbnail((4000, 4000), Image.LANCZOS)
                self.original_image = ImageOps.exif_transpose(img)
                self.original_icc = img.info.get('icc_profile')
                self.display_image = self.original_image.copy().convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image for editing: {e}", parent=self)
            self.destroy()
            return

        self._setup_ui()
        self._setup_bindings_and_state()

        self.update_idletasks()
        parent_w, parent_h = parent.winfo_width(), parent.winfo_height()
        parent_x, parent_y = parent.winfo_rootx(), parent.winfo_rooty()
        initial_w, initial_h = int(parent_w * 0.85), int(parent_h * 0.85)
        x = parent_x + (parent_w - initial_w) // 2
        y = parent_y + (parent_h - initial_h) // 2
        self.geometry(f'{initial_w}x{initial_h}+{x}+{y}')
        self.minsize(600, 400)

        self.deiconify()
        self.redraw_image()
        self.animate_ants()

    def _setup_ui(self):
        paned_window = ttk.PanedWindow(self, orient='horizontal')
        paned_window.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(paned_window, cursor="cross", bg="#333333", highlightthickness=0)
        paned_window.add(self.canvas, weight=4)

        template_panel = self._create_template_sizes_panel(paned_window)
        paned_window.add(template_panel, weight=1)
        
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(btn_frame, text="Apply Crop", command=self.on_apply).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="OK", command=self.on_ok, style="Accent.TButton").pack(side="right", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side="right", padx=5)

    def _setup_bindings_and_state(self):
        self.crop_rect_id = None
        self.crop_coords = None
        self.resize_handles = {}
        self.ant_offset = 0
        self.action_state = None
        self.start_x = self.start_y = 0
        self.drag_start_x = self.drag_start_y = 0
        self.active_handle = None

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.bind("<Configure>", self.redraw_image)

    def redraw_image(self, event=None):
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: return

        img_w, img_h = self.display_image.size
        self.scale = min(canvas_width / img_w, canvas_height / img_h)
        new_w, new_h = int(img_w * self.scale), int(img_h * self.scale)

        resized_img = self.display_image.resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_img)
        self.canvas.delete("all")

        self.offset_x = (canvas_width - new_w) / 2
        self.offset_y = (canvas_height - new_h) / 2
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.tk_image, anchor="nw")
        
        self.draw_crop_rectangle()

    def on_mouse_press(self, event):
        self.drag_start_x, self.drag_start_y = event.x, event.y
        
        if self.crop_rect_id:
            handle = self.get_handle_at(event.x, event.y)
            if handle:
                self.action_state = 'resizing'
                self.active_handle = handle
                return

            if self.is_within_crop_box(event.x, event.y):
                self.action_state = 'moving'
                return

        self.action_state = 'drawing'
        self.start_x, self.start_y = event.x, event.y
        if self.crop_rect_id: self.clear_crop_rectangle()
        self.crop_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="white", dash=(4, 4), width=2
        )

    def on_mouse_drag(self, event):
        if self.action_state == 'drawing':
            self._handle_drawing(event)
        elif self.action_state == 'moving':
            self._handle_moving(event)
        elif self.action_state == 'resizing':
            self._handle_resizing(event)
        
        self.draw_crop_rectangle()

    def on_mouse_release(self, event):
        self.action_state = None
        self.active_handle = None
        self.update_crop_coords_from_canvas()

    def on_mouse_move(self, event):
        if not self.crop_rect_id or self.action_state:
            return
        
        handle = self.get_handle_at(event.x, event.y)
        if handle:
            if handle in ['nw', 'se']:
                self.canvas.config(cursor="size_nw_se")
            else:
                self.canvas.config(cursor="size_ne_sw")
        elif self.is_within_crop_box(event.x, event.y):
            self.canvas.config(cursor="fleur")
        else:
            self.canvas.config(cursor="cross")

    def _handle_drawing(self, event):
        if not self.crop_rect_id: return
        end_x, end_y = event.x, event.y
        if self.aspect_ratio:
            width = end_x - self.start_x
            height = width / self.aspect_ratio if self.aspect_ratio else 0
            end_y = self.start_y + height
        self.canvas.coords(self.crop_rect_id, self.start_x, self.start_y, end_x, end_y)

    def _handle_moving(self, event):
        if not self.crop_rect_id: return
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.canvas.move(self.crop_rect_id, dx, dy)
        for handle_id in self.resize_handles.values():
            self.canvas.move(handle_id, dx, dy)
        self.drag_start_x, self.drag_start_y = event.x, event.y
        
    def _handle_resizing(self, event):
        if not self.crop_rect_id or not self.active_handle: return
        x1, y1, x2, y2 = self.canvas.coords(self.crop_rect_id)
        
        fixed_x = x2 if 'w' in self.active_handle else x1
        fixed_y = y2 if 'n' in self.active_handle else y1
        
        new_x, new_y = event.x, event.y

        if self.aspect_ratio:
            delta_x = new_x - fixed_x
            delta_y_from_x = delta_x / self.aspect_ratio if 's' in self.active_handle else -delta_x / self.aspect_ratio
            
            delta_y = new_y - fixed_y
            delta_x_from_y = delta_y * self.aspect_ratio if 'e' in self.active_handle else -delta_y * self.aspect_ratio
            
            if abs(delta_x) > abs(delta_y):
                new_y = fixed_y + delta_y_from_x
            else:
                new_x = fixed_x + delta_x_from_y
        
        if 'n' in self.active_handle: y1 = new_y
        if 's' in self.active_handle: y2 = new_y
        if 'w' in self.active_handle: x1 = new_x
        if 'e' in self.active_handle: x2 = new_x
        
        self.canvas.coords(self.crop_rect_id, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def animate_ants(self):
        if self.crop_rect_id:
            self.ant_offset = (self.ant_offset + 1) % 8
            self.canvas.itemconfig(self.crop_rect_id, dashoffset=self.ant_offset)
        self.after(100, self.animate_ants)

    def apply_crop(self):
        self.update_crop_coords_from_canvas()
        if self.crop_coords and (self.crop_coords[2] > self.crop_coords[0]) and (self.crop_coords[3] > self.crop_coords[1]):
            self.original_image = self.original_image.crop(self.crop_coords)
            self.display_image = self.original_image.copy().convert("RGB")
            self.clear_crop_rectangle()
            self.redraw_image()
            return True
        return False

    def on_apply(self):
        self.apply_crop()

    def on_ok(self):
        self.apply_crop()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="cropped_") as tmp_file:
                self.original_image.save(tmp_file.name, "PNG", icc_profile=self.original_icc)
                self.result = tmp_file.name
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Could not save cropped image: {e}", parent=self)

    def on_cancel(self):
        self.destroy()

    def _create_template_sizes_panel(self, parent):
        """
        Creates the UI panel with a treeview of predefined crop templates.
        NOTE: Lazy-loading is not implemented here as the number of templates
        is small and static, making the performance gain negligible.
        """
        frame = ttk.LabelFrame(parent, text="Crop Templates")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.template_tree = ttk.Treeview(frame, columns=("size",), show="tree", selectmode="browse")
        self.template_tree.grid(row=0, column=0, sticky="nsew")
        
        template_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.template_tree.yview)
        template_scroll.grid(row=0, column=1, sticky="ns")
        self.template_tree.configure(yscrollcommand=template_scroll.set)
        
        self.template_tree.column("#0", width=0, stretch=False)
        self.template_tree.column("size", anchor="w", width=150)
        
        template_sizes = [
            ("2x3 cm", 20, 30), ("3x4 cm", 30, 40), ("4x6 cm", 40, 60),
            ("Passport (3.5x4.5 cm)", 35, 45),
            ("Square (1x1)", 1, 1),
            ("16:9 Widescreen", 16, 9),
            ("4:3 Standard", 4, 3),
        ]
        
        self.template_sizes_map = {}
        for name, width, height in template_sizes:
            item_id = self.template_tree.insert("", "end", values=(name,))
            self.template_sizes_map[item_id] = (width, height)
        
        self.template_tree.bind("<<TreeviewSelect>>", self.on_template_selected)
        Tooltip(self.template_tree, "Select a template to lock the crop aspect ratio.")
        
        clear_btn = ttk.Button(frame, text="Unlock Aspect Ratio", command=self.clear_aspect_ratio)
        clear_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5,0), padx=2)
        Tooltip(clear_btn, "Remove the aspect ratio constraint from cropping.")

        return frame

    def on_template_selected(self, event):
        selected_items = self.template_tree.selection()
        if not selected_items: return
        item_id = selected_items[0]
        if item_id in self.template_sizes_map:
            width, height = self.template_sizes_map[item_id]
            if height == 0: return
            self.aspect_ratio = width / height
            
            if self.crop_rect_id:
                x1, y1, x2, y2 = self.canvas.coords(self.crop_rect_id)
                w = x2 - x1
                h = w / self.aspect_ratio
                self.canvas.coords(self.crop_rect_id, x1, y1, x1 + w, y1 + h)
                self.draw_crop_rectangle()

    def clear_aspect_ratio(self):
        self.aspect_ratio = None
        selection = self.template_tree.selection()
        if selection:
            self.template_tree.selection_remove(selection)

    def draw_crop_rectangle(self):
        if not self.crop_rect_id: return
        
        for handle_id in self.resize_handles.values():
            self.canvas.delete(handle_id)
        self.resize_handles.clear()

        x1, y1, x2, y2 = self.canvas.coords(self.crop_rect_id)
        s = RESIZE_HANDLE_SIZE / 2
        
        handle_positions = {
            'nw': (x1-s, y1-s, x1+s, y1+s), 'ne': (x2-s, y1-s, x2+s, y1+s),
            'sw': (x1-s, y2-s, x1+s, y2+s), 'se': (x2-s, y2-s, x2+s, y2+s)
        }
        
        for name, coords in handle_positions.items():
            handle_id = self.canvas.create_rectangle(coords, fill="white", outline="black")
            self.resize_handles[name] = handle_id

    def clear_crop_rectangle(self):
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None
        for handle_id in self.resize_handles.values():
            self.canvas.delete(handle_id)
        self.resize_handles.clear()
        self.crop_coords = None

    def get_handle_at(self, x, y):
        for name, handle_id in self.resize_handles.items():
            if self.canvas.find_withtag(handle_id):
                hx1, hy1, hx2, hy2 = self.canvas.coords(handle_id)
                if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                    return name
        return None

    def is_within_crop_box(self, x, y):
        if not self.crop_rect_id: return False
        x1, y1, x2, y2 = self.canvas.coords(self.crop_rect_id)
        return x1 < x < x2 and y1 < y < y2

    def update_crop_coords_from_canvas(self):
        if not self.crop_rect_id:
            self.crop_coords = None
            return
            
        x1_c, y1_c, x2_c, y2_c = self.canvas.coords(self.crop_rect_id)
        
        x1 = (x1_c - self.offset_x) / self.scale
        y1 = (y1_c - self.offset_y) / self.scale
        x2 = (x2_c - self.offset_x) / self.scale
        y2 = (y2_c - self.offset_y) / self.scale

        img_w, img_h = self.display_image.size
        x1, x2 = sorted([max(0, min(img_w, x1)), max(0, min(img_w, x2))])
        y1, y2 = sorted([max(0, min(img_h, y1)), max(0, min(img_h, y2))])
        self.crop_coords = (int(x1), int(y1), int(x2), int(y2))

class ResourceMonitor:
    """ Threaded resource monitor for CPU and Memory usage. """
    def __init__(self):
        self.cpu_usage = 0
        self.memory_usage = 0
        if not WIN32_AVAILABLE:
            self.is_supported = False
            return
        self.is_supported = True
        self.update_interval = 2
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def _monitor_loop(self):
        process = psutil.Process(os.getpid())
        while self.running:
            try:
                self.cpu_usage = psutil.cpu_percent(interval=0.5)
                self.memory_usage = process.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.running = False
            except Exception as e:
                print(f"ERROR: Resource monitoring failed: {e}")
            time.sleep(self.update_interval)

    def stop(self):
        self.running = False

class GPUManager:
    """ Manages OpenCL context and kernels for GPU-accelerated operations. """
    def __init__(self):
        self.ctx = self.queue = self.program = self.resize_kernel = self.brightness_kernel = None
        if GPU_ACCELERATION:
            try:
                platforms = cl.get_platforms()
                if not platforms:
                    raise RuntimeError("No OpenCL platforms found.")
                gpu_devices = [dev for p in platforms for dev in p.get_devices(device_type=cl.device_type.GPU)]
                if gpu_devices:
                    self.ctx = cl.Context(devices=[gpu_devices[0]])
                    self.queue = cl.CommandQueue(self.ctx)
                    print(f"INFO: Using GPU for acceleration: {gpu_devices[0].name}")
                    self._build_program()
                else:
                    print("INFO: No OpenCL GPU devices found, falling back to CPU.")
            except Exception as e:
                print(f"WARNING: GPU initialization failed: {e}")
                self.resize_kernel = self.brightness_kernel = None

    def _build_program(self):
        if not self.ctx: return
        try:
            source = """
            __kernel void resize_bilinear(__global const uchar *input, __global uchar *output,
                const int width, const int height, const int channels,
                const int new_width, const int new_height) {
                int x = get_global_id(0); int y = get_global_id(1);
                if (x >= new_width || y >= new_height) return;

                float gx = ((float)x + 0.5f) * ((float)width / new_width) - 0.5f;
                float gy = ((float)y + 0.5f) * ((float)height / new_height) - 0.5f;

                int gxi0 = (int)floor(gx); int gyi0 = (int)floor(gy);
                int gxi1 = min(gxi0 + 1, width - 1); int gyi1 = min(gyi0 + 1, height - 1);

                float dx = gx - gxi0; float dy = gy - gyi0;

                for (int c = 0; c < channels; c++) {
                    float p00 = input[(gyi0 * width + gxi0) * channels + c];
                    float p01 = input[(gyi0 * width + gxi1) * channels + c];
                    float p10 = input[(gyi1 * width + gxi0) * channels + c];
                    float p11 = input[(gyi1 * width + gxi1) * channels + c];

                    float p0 = p00 * (1.0f - dx) + p01 * dx;
                    float p1 = p10 * (1.0f - dx) + p11 * dx;
                    float p = p0 * (1.0f - dy) + p1 * dy;

                    output[(y * new_width + x) * channels + c] = (uchar)clamp(p, 0.0f, 255.0f);
                }
            }
            
            __kernel void adjust_brightness(__global uchar* image,
                                            const int num_pixels,
                                            const float factor) {
                int i = get_global_id(0);
                if (i >= num_pixels) return;

                int base_idx = i * 3; // Assuming RGB input
                float r = image[base_idx + 0] * factor;
                float g = image[base_idx + 1] * factor;
                float b = image[base_idx + 2] * factor;

                image[base_idx + 0] = (uchar)clamp(r, 0.0f, 255.0f);
                image[base_idx + 1] = (uchar)clamp(g, 0.0f, 255.0f);
                image[base_idx + 2] = (uchar)clamp(b, 0.0f, 255.0f);
            }
            """
            self.program = cl.Program(self.ctx, source).build()
            self.resize_kernel = self.program.resize_bilinear
            self.brightness_kernel = self.program.adjust_brightness
        except Exception as e:
            print(f"ERROR: Failed to build OpenCL program: {e}")
            self.resize_kernel = self.brightness_kernel = None

    def gpu_resize(self, img, target_width, target_height):
        if not self.resize_kernel:
            return img.resize((target_width, target_height), Image.LANCZOS)

        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            np_img = np.array(img, dtype=np.uint8)
            h, w, channels = np_img.shape
            mf = cl.mem_flags

            input_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_img)
            output_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=target_height * target_width * channels)

            self.resize_kernel.set_args(input_buf, output_buf, np.int32(w), np.int32(h), np.int32(channels), np.int32(target_width), np.int32(target_height))
            cl.enqueue_nd_range_kernel(self.queue, self.resize_kernel, (target_width, target_height), None)

            output_np = np.empty((target_height, target_width, channels), dtype=np.uint8)
            cl.enqueue_copy(self.queue, output_np, output_buf).wait()
            return Image.fromarray(output_np)
        except Exception as e:
            print(f"WARNING: GPU resize failed, falling back to CPU. Error: {e}")
            return img.resize((target_width, target_height), Image.LANCZOS)

    def gpu_adjust_brightness(self, img, factor):
        if not self.brightness_kernel or factor == 1.0:
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(factor)

        try:
            img_copy = img.copy()
            if img_copy.mode != 'RGB':
                img_copy = img_copy.convert('RGB')

            np_img = np.array(img_copy, dtype=np.uint8)
            h, w, channels = np_img.shape
            if channels != 3: # Kernel is written for RGB
                enhancer = ImageEnhance.Brightness(img)
                return enhancer.enhance(factor)

            mf = cl.mem_flags
            buffer = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_img)
            
            self.brightness_kernel.set_args(buffer, np.int32(w * h), np.float32(factor))
            cl.enqueue_nd_range_kernel(self.queue, self.brightness_kernel, (w * h,), None)
            
            output_np = np.empty_like(np_img)
            cl.enqueue_copy(self.queue, output_np, buffer).wait()
            return Image.fromarray(output_np)
        except Exception as e:
            print(f"WARNING: GPU brightness adjustment failed, falling back to CPU. Error: {e}")
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(factor)

class CustomPaperDialog(tk.Toplevel):
    """ A simple dialog to get custom paper dimensions from the user. """
    def __init__(self, parent, current_size_mm):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Custom Paper Size")
        self.resizable(False, False)
        self.result = None

        try:
            self.iconbitmap(resource_path("neka_icon.ico"))
        except tk.TclError:
            print("INFO: Icon 'neka_icon.ico' not found for custom paper dialog.")

        frame = ttk.Frame(self, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Width (mm):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.width_var = tk.IntVar(value=current_size_mm[0])
        width_entry = ttk.Entry(frame, textvariable=self.width_var, width=10)
        width_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        width_entry.focus()

        ttk.Label(frame, text="Height (mm):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.height_var = tk.IntVar(value=current_size_mm[1])
        ttk.Entry(frame, textvariable=self.height_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(15, 0))
        ttk.Button(btn_frame, text="OK", command=self.on_ok, style="Accent.TButton").pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="left", padx=5)

        self.bind("<Return>", lambda e: self.on_ok())
        self.update_idletasks()
        self.geometry(f"+{parent.winfo_rootx()+50}+{parent.winfo_rooty()+50}")

    def on_ok(self):
        try:
            width, height = self.width_var.get(), self.height_var.get()
            if width <= 0 or height <= 0:
                messagebox.showerror("Invalid Size", "Width and height must be positive numbers.", parent=self)
                return
            self.result = (width, height)
            self.destroy()
        except tk.TclError:
            messagebox.showerror("Invalid Input", "Please enter valid whole numbers for width and height.", parent=self)

# =============================================================================
# --- Main Application ---
# =============================================================================
class PhotoOrganizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neka Photocopy")
        self.root.geometry("1280x800")
        self.root.minsize(900, 700)

        try:
            self.root.iconbitmap(resource_path("neka_icon.ico"))
        except tk.TclError:
            print("INFO: Main application icon 'neka_icon.ico' not found.")

        self.resource_monitor = None
        self.gpu_manager = None
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        
        # UPDATE: Replaced standard queue with a PriorityQueue to allow critical UI updates
        # (e.g., dialogs, final renders) to be processed before less important ones.
        self.ui_queue = queue.PriorityQueue()
        self.task_counter = itertools.count() # Used for stable sorting in the priority queue

        self.photos = []
        self.current_photo = None
        self.layouts = []
        self.current_page_index = 0
        self.is_rendering = False
        self.programmatically_setting = False
        self.render_after_id = None
        
        self.master_image_cache = OrderedDict()
        self.processed_image_cache = OrderedDict()
        self.thumbnail_cache = OrderedDict()
        self.layout_cache = None
        self.layout_cache_key = None
        self.list_photo_thumbnails = {}
        self.item_id_to_photo_id = {}
        
        self._internal_copy_photo_object = None

        self.printers = []
        self.printer_settings = None
        self.printer_icc_profile_cache = {}
        self.monitor_icc_profile = None
        self.srgb_profile = ImageCms.createProfile("sRGB")

        self._setup_style()
        self._init_vars()
        self._setup_ui()

        self.root.after(500, self._initialize_background_services)
        self.root.after(100, self.process_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _initialize_background_services(self):
        self.status_var.set("Initializing background services...")
        self.root.update_idletasks()

        self.gpu_manager = GPUManager()
        self.resource_monitor = ResourceMonitor()

        if self.resource_monitor and self.resource_monitor.is_supported:
            self.update_resource_display()

        if WIN32_AVAILABLE:
            self.load_printers_background()
            self.monitor_icc_profile = self._get_monitor_icc_profile()
        else:
            self.printer_var.set("Printing not supported on this OS")
            self.status_var.set("Ready")

    def _setup_style(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            print("WARNING: 'clam' theme not available, using default.")
        
        self.style.configure("TLabel", padding=5)
        self.style.configure("TButton", padding=5)
        self.style.configure("Treeview", rowheight=45)
        self.style.configure("Accent.TButton", foreground="white", background="#0078D7")
        self.style.map("Accent.TButton", background=[('active', '#005A9E')])
        self.style.configure("Warning.TLabel", background="orange", foreground="black", anchor='w')

    def _init_vars(self):
        self.photo_width = tk.IntVar(value=DEFAULT_PHOTO_WIDTH_MM)
        self.photo_height = tk.IntVar(value=DEFAULT_PHOTO_HEIGHT_MM)
        self.photo_copies = tk.IntVar(value=DEFAULT_PHOTO_COPIES)
        self.photo_brightness = tk.IntVar(value=DEFAULT_BRIGHTNESS)
        self.photo_resize_mode = tk.StringVar(value=DEFAULT_RESIZE_MODE)

        self.paper_sizes = [
            ("A4 (210x297 mm)", 210, 297),
            ("A5 (148x210 mm)", 148, 210),
            ("A6 (105x148 mm)", 105, 148),
            ("B5 (176x250 mm)", 176, 250),
            ("B6 (125x176 mm)", 125, 176),
            ("F4 (216x330 mm)", 216, 330),
            ("Letter (216x279 mm)", 216, 279),
            ("Legal (216x356 mm)", 216, 356),
            ("User Defined", 0, 0)
        ]
        self.user_defined_size = (210, 297)
        self.paper_size_var = tk.StringVar(value=self.paper_sizes[0][0])
        self.margin = tk.IntVar(value=DEFAULT_MARGIN_MM)
        self.interval = tk.IntVar(value=DEFAULT_INTERVAL_MM)
        self.outline = tk.BooleanVar(value=True)
        
        # NEW: Replace rotation_allowed with symmetrical_orientation for symmetrical mode
        self.symmetrical_orientation = tk.StringVar(value="Portrait")
        self.rotation_allowed = tk.BooleanVar(value=True)  # For efficient mode
        self.layout_mode = tk.StringVar(value="Efficient")

        self.printer_var = tk.StringVar()
        self.print_range_var = tk.StringVar(value="all")
        self.status_var = tk.StringVar(value="Ready")
        self.resource_var = tk.StringVar(value="CPU: - | Mem: -")

        self.rendering_intent_var = tk.StringVar(value="Relative Colorimetric")
        self.gamut_warning_var = tk.BooleanVar(value=False)
        
        self.gamut_warning_var.trace_add("write", lambda *a: self.schedule_render(force_relayout=False))
        self.rendering_intent_var.trace_add("write", lambda *a: self.schedule_render(force_relayout=False))
        self.layout_mode.trace_add("write", lambda *a: self.on_layout_mode_changed())

    def on_layout_mode_changed(self):
        """Update UI when layout mode changes between Efficient and Symmetrical"""
        mode = self.layout_mode.get()
        
        # Show/hide orientation controls based on mode
        if hasattr(self, 'rotation_frame') and hasattr(self, 'symmetrical_orientation_frame'):
            if mode == "Symmetrical":
                self.rotation_frame.pack_forget()
                self.symmetrical_orientation_frame.pack(side='left', padx=(10, 0))
                # Disable "Apply to Selected" in symmetrical mode
                if hasattr(self, 'apply_btn'):
                    self.apply_btn.config(state='disabled')
            else:  # Efficient mode
                self.symmetrical_orientation_frame.pack_forget()
                self.rotation_frame.pack(side='left', padx=(10, 0))
                # Enable "Apply to Selected" in efficient mode
                if hasattr(self, 'apply_btn'):
                    self.apply_btn.config(state='normal')
        
        self.schedule_render(force_relayout=True)

    def _setup_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)

        main_paned_window = ttk.PanedWindow(self.root, orient='horizontal')
        main_paned_window.grid(row=0, column=0, sticky='nsew', padx=10, pady=(10, 5))

        left_panel = self._create_left_panel(main_paned_window)
        main_paned_window.add(left_panel, weight=1)

        center_panel = self._create_center_panel(main_paned_window)
        main_paned_window.add(center_panel, weight=4)

        right_panel = self._create_right_panel(main_paned_window)
        main_paned_window.add(right_panel, weight=1)

        bottom_panel = self._create_bottom_panel(self.root)
        bottom_panel.grid(row=1, column=0, sticky='ew', padx=10, pady=5)
        
        status_bar = ttk.Frame(self.root, style="TFrame")
        status_bar.grid(row=2, column=0, sticky='ew')
        self.status_label = ttk.Label(status_bar, textvariable=self.status_var, relief='sunken', anchor='w')
        self.status_label.pack(side='left', fill='x', expand=True)
        ttk.Label(status_bar, textvariable=self.resource_var, relief='sunken', anchor='e').pack(side='right')

        self.canvas.bind("<Configure>", lambda e: self.schedule_render(force_relayout=False))
        self.root.bind_all("<Control-c>", self.copy_photo)
        self.root.bind_all("<Control-v>", self.paste_photo)

    def _create_left_panel(self, parent):
        panel = ttk.Frame(parent, width=250)
        panel.pack(fill='y', side='left')

        list_frame = ttk.LabelFrame(panel, text="Photo List")
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.photo_tree = ttk.Treeview(list_frame, columns=("name",), show="tree headings", selectmode="browse")
        self.photo_tree.grid(row=0, column=0, sticky="nsew")
        self.photo_tree.heading("#0", text="Preview", anchor="w")
        self.photo_tree.column("#0", width=60, anchor="center", stretch=False)
        self.photo_tree.heading("name", text="Filename", anchor="w")
        self.photo_tree.column("name", width=180, anchor="w", stretch=True)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.photo_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.photo_tree.config(yscrollcommand=scrollbar.set)
        
        self.photo_tree.bind("<<TreeviewSelect>>", self.on_photo_select)
        self.photo_tree.bind("<Delete>", lambda e: self.remove_selected_photo())

        if DND_ENABLED:
            self.photo_tree.drop_target_register(DND_FILES)
            self.photo_tree.dnd_bind('<<Drop>>', self.on_drop)
            Tooltip(list_frame, "Drag and drop image files here.")

        btn_frame = ttk.Frame(list_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        btn_frame.columnconfigure((0, 1), weight=1)
        
        add_btn = ttk.Button(btn_frame, text="Add Photos...", command=self.add_photos)
        add_btn.grid(row=0, column=0, columnspan=2, padx=2, pady=2, sticky="ew")
        Tooltip(add_btn, "Open a dialog to select image files (JPG, PNG, BMP).")
        
        remove_btn = ttk.Button(btn_frame, text="Remove", command=self.remove_selected_photo)
        remove_btn.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        Tooltip(remove_btn, "Remove the currently selected photo (Delete key).")
        
        remove_all_btn = ttk.Button(btn_frame, text="Remove All", command=self.remove_all_photos)
        remove_all_btn.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        Tooltip(remove_all_btn, "Clear all photos from the list.")

        return panel

    def _create_center_panel(self, parent):
        panel = ttk.Frame(parent)
        panel.pack(fill='both', expand=True, side='left')
        panel.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(panel, text="Layout Preview (Soft Proofing Enabled)")
        preview_frame.pack(fill='both', expand=True, padx=5, pady=5)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(preview_frame, bg="gray75", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        proofing_panel = self._create_proofing_panel(preview_frame)
        proofing_panel.grid(row=1, column=0, sticky="ew", pady=(5,0))
        
        nav_frame = ttk.Frame(preview_frame)
        nav_frame.grid(row=2, column=0, sticky="ew", pady=(5,0))
        nav_frame.columnconfigure(1, weight=1)
        self.prev_button = ttk.Button(nav_frame, text="< Prev", command=self.prev_page, state="disabled")
        self.prev_button.grid(row=0, column=0, padx=5)
        self.page_label = ttk.Label(nav_frame, text="Page 0 / 0", anchor="center")
        self.page_label.grid(row=0, column=1, sticky="ew")
        self.next_button = ttk.Button(nav_frame, text="Next >", command=self.next_page, state="disabled")
        self.next_button.grid(row=0, column=2, padx=5)

        return panel

    def _create_proofing_panel(self, parent):
        """Creates the UI panel for soft proofing controls."""
        panel = ttk.Frame(parent)
        
        ttk.Label(panel, text="Intent:").pack(side='left', padx=(5, 2))
        intent_combo = ttk.Combobox(panel, textvariable=self.rendering_intent_var,
                                    values=["Perceptual", "Relative Colorimetric"],
                                    state="readonly", width=20)
        intent_combo.pack(side='left')
        Tooltip(intent_combo, "Perceptual: Good for saturated images, preserves color relationships.\nRelative Colorimetric: Accurate for in-gamut colors, clips others.")

        gamut_warn_cb = ttk.Checkbutton(panel, text="Gamut Warning", variable=self.gamut_warning_var)
        gamut_warn_cb.pack(side='left', padx=(10, 0))
        Tooltip(gamut_warn_cb, "Highlights colors that are outside the printer's color range.")
        if not NUMPY_AVAILABLE:
            gamut_warn_cb.config(state='disabled')
            Tooltip(gamut_warn_cb, "This feature requires the 'numpy' library to be installed.")

        return panel

    def _create_right_panel(self, parent):
        panel = ttk.Frame(parent, width=300)
        panel.pack(fill='y', side='left')
        panel.rowconfigure(1, weight=1)

        settings_frame = ttk.LabelFrame(panel, text="Photo Settings")
        settings_frame.pack(fill='x', padx=5, pady=5)
        settings_frame.columnconfigure(1, weight=1)
        
        vcmd = (panel.register(self._validate_numeric_input), '%P')

        ttk.Label(settings_frame, text="Width (mm):").grid(row=0, column=0, padx=5, pady=3, sticky="w")
        w_entry = ttk.Entry(settings_frame, textvariable=self.photo_width, width=8, validate='key', validatecommand=vcmd)
        w_entry.grid(row=0, column=1, padx=5, pady=3, sticky="ew")

        ttk.Label(settings_frame, text="Height (mm):").grid(row=1, column=0, padx=5, pady=3, sticky="w")
        h_entry = ttk.Entry(settings_frame, textvariable=self.photo_height, width=8, validate='key', validatecommand=vcmd)
        h_entry.grid(row=1, column=1, padx=5, pady=3, sticky="ew")

        ttk.Label(settings_frame, text="Copies:").grid(row=2, column=0, padx=5, pady=3, sticky="w")
        c_entry = ttk.Entry(settings_frame, textvariable=self.photo_copies, width=8, validate='key', validatecommand=vcmd)
        c_entry.grid(row=2, column=1, padx=5, pady=3, sticky="ew")

        ttk.Label(settings_frame, text="Brightness:").grid(row=3, column=0, padx=5, pady=3, sticky="w")
        brightness_frame = ttk.Frame(settings_frame)
        brightness_frame.grid(row=3, column=1, sticky="ew", padx=5, pady=3)
        brightness_scale = ttk.Scale(brightness_frame, from_=-100, to=100, orient="horizontal", variable=self.photo_brightness, command=self.on_brightness_changed)
        brightness_scale.pack(side="left", fill="x", expand=True)
        self.brightness_label = ttk.Label(brightness_frame, text="0", width=4)
        self.brightness_label.pack(side="right", padx=(5, 0))
        self.photo_brightness.trace_add("write", self.update_brightness_label)
        Tooltip(brightness_scale, "Adjust photo brightness (-100 to 100).")

        ttk.Label(settings_frame, text="Resize Mode:").grid(row=4, column=0, padx=5, pady=3, sticky="w")
        resize_combo = ttk.Combobox(settings_frame, textvariable=self.photo_resize_mode, values=["stretch", "crop", "fit"], state="readonly", width=7)
        resize_combo.grid(row=4, column=1, padx=5, pady=3, sticky="ew")
        Tooltip(resize_combo, "'stretch': Fills the dimensions.\n'crop': Resizes and crops to fit.\n'fit': Resizes to fit inside, may add borders.")

        btn_frame = ttk.Frame(settings_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(10, 5))
        btn_frame.columnconfigure((0, 1), weight=1)

        edit_btn = ttk.Button(btn_frame, text="Edit/Crop...", command=self.edit_photo)
        edit_btn.grid(row=0, column=0, columnspan=2, padx=2, pady=2, sticky="ew")
        Tooltip(edit_btn, "Open an editor to crop the selected photo.")
        
        self.apply_btn = ttk.Button(btn_frame, text="Apply to Selected", command=self.apply_photo_settings)
        self.apply_btn.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        Tooltip(self.apply_btn, "Apply the current settings to the selected photo.")

        apply_all_btn = ttk.Button(btn_frame, text="Apply to All", command=self.apply_settings_to_all)
        apply_all_btn.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        Tooltip(apply_all_btn, "Apply the current settings to ALL photos in the list.")

        template_frame = self._create_template_sizes_panel_right(panel)
        template_frame.pack(fill='both', expand=True, padx=5, pady=5)

        return panel

    def _create_template_sizes_panel_right(self, parent):
        """
        Creates the UI panel with a treeview of predefined photo sizes.
        NOTE: Lazy-loading is not implemented here as the number of templates
        is small and static, making the performance gain negligible.
        """
        frame = ttk.LabelFrame(parent, text="Template Sizes")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.template_tree = ttk.Treeview(frame, columns=("size",), show="tree", selectmode="browse")
        self.template_tree.grid(row=0, column=0, sticky="nsew")
        
        template_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.template_tree.yview)
        template_scroll.grid(row=0, column=1, sticky="ns")
        self.template_tree.configure(yscrollcommand=template_scroll.set)
        
        self.template_tree.column("#0", width=0, stretch=False)
        self.template_tree.column("size", anchor="w", width=150)
        
        template_sizes = [
            ("2x3 cm", 20, 30), ("3x4 cm", 30, 40), ("4x6 cm", 40, 60),
            ("2R (6x9 cm)", 60, 90), ("3R (8.9x12.7 cm)", 89, 127),
            ("4R (10.2x15.2 cm)", 102, 152), ("5R (12.7x17.8 cm)", 127, 178)
        ]
        
        self.template_sizes_map = {}
        for name, width, height in template_sizes:
            item_id = self.template_tree.insert("", "end", values=(name,))
            self.template_sizes_map[item_id] = (width, height)
        
        self.template_tree.bind("<<TreeviewSelect>>", self.on_template_selected)
        Tooltip(self.template_tree, "Select a standard size to apply it to the settings above.")
        return frame

    def _create_bottom_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Layout and Print Settings")
        
        paper_frame = ttk.LabelFrame(panel, text="Paper")
        paper_frame.pack(side="left", fill="both", padx=10, pady=5, expand=True)
        paper_frame.columnconfigure(1, weight=1)

        ttk.Label(paper_frame, text="Paper Size:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.paper_menu = ttk.Combobox(paper_frame, textvariable=self.paper_size_var, values=[x[0] for x in self.paper_sizes], state="readonly")
        self.paper_menu.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.paper_menu.bind("<<ComboboxSelected>>", self.on_paper_selected)
        Tooltip(self.paper_menu, "Select the physical paper size.")

        ttk.Label(paper_frame, text="Margin (mm):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(paper_frame, textvariable=self.margin).grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(paper_frame, text="Spacing (mm):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(paper_frame, textvariable=self.interval).grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        Tooltip(paper_frame.grid_slaves(row=2, column=1)[0], "The gap between photos on the page.")

        ttk.Label(paper_frame, text="Layout Mode:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        layout_combo = ttk.Combobox(paper_frame, textvariable=self.layout_mode, values=["Efficient", "Symmetrical"], state="readonly")
        layout_combo.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        Tooltip(layout_combo, "'Efficient': Packs photos tightly to save space.\n'Symmetrical': Arranges photos in neat rows.")

        # NEW: Replace rotation checkbox with orientation controls for symmetrical mode
        cb_frame = ttk.Frame(paper_frame)
        cb_frame.grid(row=4, column=0, columnspan=2, sticky='w', pady=5)
        
        ttk.Checkbutton(cb_frame, text="Draw Outlines", variable=self.outline, 
                       command=lambda: self.schedule_render(force_relayout=False)).pack(side='left')
        
        # Rotation frame for Efficient mode
        self.rotation_frame = ttk.Frame(cb_frame)
        ttk.Checkbutton(self.rotation_frame, text="Allow Rotation", variable=self.rotation_allowed,
                       command=lambda: self.schedule_render(force_relayout=True)).pack(side='left')
        Tooltip(self.rotation_frame, "Allow photos to be rotated 90 degrees to fit better.")
        
        # Symmetrical orientation frame for Symmetrical mode
        self.symmetrical_orientation_frame = ttk.Frame(cb_frame)
        ttk.Label(self.symmetrical_orientation_frame, text="Orientation:").pack(side='left')
        ttk.Radiobutton(self.symmetrical_orientation_frame, text="Portrait", 
                       variable=self.symmetrical_orientation, value="Portrait",
                       command=lambda: self.schedule_render(force_relayout=True)).pack(side='left', padx=(5,0))
        ttk.Radiobutton(self.symmetrical_orientation_frame, text="Landscape", 
                       variable=self.symmetrical_orientation, value="Landscape",
                       command=lambda: self.schedule_render(force_relayout=True)).pack(side='left', padx=(5,0))
        
        # Initially show the correct frame based on current mode
        if self.layout_mode.get() == "Symmetrical":
            self.symmetrical_orientation_frame.pack(side='left', padx=(10, 0))
        else:
            self.rotation_frame.pack(side='left', padx=(10, 0))

        refresh_btn = ttk.Button(paper_frame, text="Refresh Preview", command=lambda: self.schedule_render(force_relayout=True))
        refresh_btn.grid(row=5, column=0, columnspan=2, pady=5, sticky="ew")
        Tooltip(refresh_btn, "Force the layout to be recalculated and the preview redrawn.")

        printer_frame = ttk.LabelFrame(panel, text="Printer")
        printer_frame.pack(side="right", fill="both", padx=10, pady=5, expand=True)

        ttk.Label(printer_frame, text="Select Printer:").pack(anchor="w", padx=5, pady=2)
        self.printer_combo = ttk.Combobox(printer_frame, textvariable=self.printer_var, state="readonly")
        self.printer_combo.pack(fill="x", padx=5, pady=2)
        if WIN32_AVAILABLE:
            self.printer_combo.bind("<<ComboboxSelected>>", self.on_printer_selected)
        else:
            self.printer_combo.config(state="disabled")

        props_button = ttk.Button(printer_frame, text="Printer Properties...", command=self.show_printer_properties)
        props_button.pack(fill="x", padx=5, pady=(5,0))
        if not WIN32_AVAILABLE: props_button.config(state="disabled")
        
        print_controls_frame = ttk.Frame(printer_frame)
        print_controls_frame.pack(fill='x', padx=5, pady=5)
        print_controls_frame.columnconfigure(0, weight=1)

        radio_frame = ttk.Frame(print_controls_frame)
        radio_frame.grid(row=0, column=0, sticky='ew')
        
        print_button_frame = ttk.Frame(print_controls_frame)
        print_button_frame.grid(row=0, column=1, sticky='ns', padx=(10,0))

        def toggle_specific_entry():
            state = "normal" if self.print_range_var.get() == "specific" else "disabled"
            self.specific_pages_entry.config(state=state)
            if state == "normal": self.specific_pages_entry.focus()

        ttk.Radiobutton(radio_frame, text="All Pages", variable=self.print_range_var, value="all", command=toggle_specific_entry).pack(anchor='w')
        ttk.Radiobutton(radio_frame, text="Current Page", variable=self.print_range_var, value="current", command=toggle_specific_entry).pack(anchor='w')
        
        specific_frame = ttk.Frame(radio_frame)
        specific_frame.pack(anchor='w', fill='x')
        ttk.Radiobutton(specific_frame, text="Pages:", variable=self.print_range_var, value="specific", command=toggle_specific_entry).pack(side="left")
        self.specific_pages_entry = ttk.Entry(specific_frame, width=12, state="disabled")
        self.specific_pages_entry.pack(side="left", padx=2, expand=True, fill='x')
        Tooltip(self.specific_pages_entry, "e.g., 1, 3, 5-8")

        print_btn = ttk.Button(print_button_frame, text="Print", command=self.print_layout, style="Accent.TButton")
        print_btn.pack(expand=True, fill='both')
        if not WIN32_AVAILABLE: print_btn.config(state="disabled")

        return panel

    def process_ui_queue(self):
        """
        Process UI updates from background threads safely via the priority queue.
        This function is run periodically on the main Tkinter thread.
        """
        try:
            while not self.ui_queue.empty():
                # Get item from priority queue: (priority, count, task_tuple)
                _priority, _count, task = self.ui_queue.get_nowait()
                callback, args, kwargs = task
                callback(*args, **kwargs)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"ERROR: Unhandled exception in UI queue processor: {e}")
        finally:
            self.root.after(100, self.process_ui_queue)

    def queue_ui_update(self, callback, *args, priority=10, **kwargs):
        """
        A thread-safe way to schedule a UI update using a priority queue.
        Lower priority numbers are processed first.
        - Priority 1-5: Critical UI updates (e.g., dialogs, final renders).
        - Priority 10: Standard updates.
        - Priority 20+: Background/lazy-loading tasks.
        """
        task_tuple = (callback, args, kwargs)
        # The counter from itertools ensures that if priorities are equal,
        # tasks are processed in the order they were added (FIFO).
        self.ui_queue.put((priority, next(self.task_counter), task_tuple))

    @contextmanager
    def busy_cursor(self):
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            yield
        finally:
            self.root.config(cursor="")

    def _validate_numeric_input(self, P):
        return P.isdigit() or P == ""

    def on_close(self):
        self.status_var.set("Shutting down...")
        if self.resource_monitor:
            self.resource_monitor.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()
        gc.collect()

    def on_brightness_changed(self, value):
        if self.current_photo and not self.programmatically_setting:
            self.apply_photo_settings()

    def update_brightness_label(self, *args):
        try:
            self.brightness_label.config(text=str(int(float(self.photo_brightness.get()))))
        except (tk.TclError, ValueError):
            pass

    def update_resource_display(self):
        if self.resource_monitor and self.resource_monitor.is_supported:
            mem_usage = self.resource_monitor.memory_usage
            self.resource_var.set(f"CPU: {self.resource_monitor.cpu_usage:.1f}% | Mem: {mem_usage:.1f} MB")
            
            # UPDATE: Proactively clear caches on high memory usage to mitigate thrashing.
            if mem_usage > MEMORY_WARNING_THRESHOLD_MB:
                # Check if a warning is already displayed to avoid repeated clearing
                if self.status_label.cget("style") != "Warning.TLabel":
                    self.status_var.set(f"WARNING: High memory usage ({mem_usage:.0f} MB). Clearing image caches...")
                    self.status_label.config(style="Warning.TLabel")
                    self.root.update_idletasks() # Ensure message is displayed before potential lag from gc
                    self.clear_caches(aggressive=False) # Clears thumbnail and processed caches
            else:
                if self.status_label.cget("style") == "Warning.TLabel":
                    self.status_var.set("Ready")
                    self.status_label.config(style="TLabel")
        
        self.root.after(2000, self.update_resource_display)

    def on_paper_selected(self, event):
        selected_paper = self.paper_size_var.get()
        if selected_paper == "User Defined":
            self.show_custom_paper_dialog()
        else:
            if WIN32_AVAILABLE:
                self.update_printer_settings_from_ui(self.printer_var.get())
            self.schedule_render(force_relayout=True)

    def on_printer_selected(self, event=None):
        printer_name = self.printer_var.get()
        self.printer_settings = None
        self.status_var.set(f"Printer changed to {printer_name}. Syncing settings...")
        print(f"INFO: Printer settings reset due to new selection: {printer_name}")
        
        if WIN32_AVAILABLE:
            self.update_printer_settings_from_ui(printer_name)

        self.thumbnail_cache.clear()
        self.processed_image_cache.clear()
        self.schedule_render(force_relayout=True)

    def show_custom_paper_dialog(self):
        dialog = CustomPaperDialog(self.root, self.user_defined_size)
        self.root.wait_window(dialog)
        if dialog.result:
            self.user_defined_size = dialog.result
            if WIN32_AVAILABLE:
                self.update_printer_settings_from_ui(self.printer_var.get())
            self.schedule_render(force_relayout=True)

    def on_drop(self, event):
        try:
            files = self.root.tk.splitlist(event.data)
            self.add_photos(files)
        except Exception as e:
            messagebox.showerror("Drop Error", f"Failed to process dropped files: {e}")

    def on_template_selected(self, event):
        selected_items = self.template_tree.selection()
        if not selected_items: return
        item_id = selected_items[0]
        if item_id in self.template_sizes_map:
            width, height = self.template_sizes_map[item_id]
            self.photo_width.set(width)
            self.photo_height.set(height)
            self.status_var.set(f"Selected template: {self.template_tree.item(item_id, 'values')[0]}")
            if self.current_photo:
                self.apply_photo_settings()

    def prev_page(self):
        if self.current_page_index > 0:
            self.current_page_index -= 1
            self.render_preview()
            self.update_page_controls()

    def next_page(self):
        if self.current_page_index < len(self.layouts) - 1:
            self.current_page_index += 1
            self.render_preview()
            self.update_page_controls()

    def update_page_controls(self):
        num_pages = len(self.layouts)
        if num_pages > 0:
            self.page_label.config(text=f"Page {self.current_page_index + 1} / {num_pages}")
            self.prev_button.config(state="normal" if self.current_page_index > 0 else "disabled")
            self.next_button.config(state="normal" if self.current_page_index < num_pages - 1 else "disabled")
        else:
            self.page_label.config(text="Page 0 / 0")
            self.prev_button.config(state="disabled")
            self.next_button.config(state="disabled")

    def add_photos(self, file_paths=None):
        if not file_paths:
            file_paths = filedialog.askopenfilenames(
                title="Select Photos",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
            )
        if not file_paths: return 0

        self.status_var.set(f"Adding {len(file_paths)} photos...")
        added_count = 0
        skipped_files = []
        for f in file_paths:
            self.status_var.set(f"Validating {os.path.basename(f)}...")
            self.root.update_idletasks()
            
            try:
                if not os.path.isfile(f):
                    print(f"WARNING: Skipping non-file path: {f}")
                    continue

                file_size_mb = os.path.getsize(f) / (1024 * 1024)
                if file_size_mb > MAX_IMAGE_FILE_SIZE_MB:
                    skipped_files.append(f"{os.path.basename(f)} (File too large: {file_size_mb:.1f} MB)")
                    continue

                with Image.open(f) as img:
                    original_width, original_height = img.size
                    icc_profile_data = img.info.get('icc_profile')
                
                if original_width > MAX_IMAGE_DIMENSION_PX or original_height > MAX_IMAGE_DIMENSION_PX:
                    skipped_files.append(f"{os.path.basename(f)} (Dimensions too large: {original_width}x{original_height})")
                    continue

                photo = Photo(f, self.photo_width.get(), self.photo_height.get(),
                              self.photo_copies.get(), self.photo_brightness.get(),
                              self.photo_resize_mode.get(), original_width, original_height, icc_profile_data)
                self.photos.append(photo)
                
                placeholder = ImageTk.PhotoImage(Image.new("RGB", (40, 40), "gray"))
                item_id = self.photo_tree.insert("", "end", text="", image=placeholder,
                                                 values=(os.path.basename(f),))
                self.list_photo_thumbnails[item_id] = placeholder
                self.item_id_to_photo_id[item_id] = photo.id
                self.executor.submit(self.load_and_update_list_thumbnail, item_id, photo)
                added_count += 1
            except Exception as e:
                skipped_files.append(f"{os.path.basename(f)} (Error: {e})")
        
        if skipped_files:
            messagebox.showwarning("Some Files Skipped",
                                   "The following files were skipped due to size, dimensions, or errors:\n\n" +
                                   "\n".join(skipped_files))
        
        if added_count > 0:
            self.status_var.set(f"Added {added_count} photo(s).")
            self.schedule_render(force_relayout=True)
        else:
            self.status_var.set("Ready")
        return added_count

    def load_and_update_list_thumbnail(self, item_id, photo):
        try:
            working_img = self.get_working_image(photo.path)
            source_profile = self._get_cms_profile(photo.icc_profile_data)
            
            display_img = ImageCms.profileToProfile(working_img, source_profile, self.srgb_profile,
                                                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                                    outputMode='RGB')

            display_img.thumbnail((40, 40), Image.LANCZOS)
            thumbnail = ImageTk.PhotoImage(display_img)
            # Low priority update for background-loaded thumbnails
            self.queue_ui_update(self.update_list_thumbnail, item_id, thumbnail, priority=20)
        except Exception as e:
            print(f"ERROR: Failed to create thumbnail for {photo.path}: {e}")
            error_thumb = ImageTk.PhotoImage(Image.new("RGB", (40,40), "red"))
            self.queue_ui_update(self.update_list_thumbnail, item_id, error_thumb, priority=20)

    def update_list_thumbnail(self, item_id, thumbnail):
        if self.photo_tree.exists(item_id):
            self.photo_tree.item(item_id, image=thumbnail)
            self.list_photo_thumbnails[item_id] = thumbnail

    def on_photo_select(self, event):
        selection = self.photo_tree.selection()
        if not selection: return
        
        photo = self.get_photo_by_item_id(selection[0])
        if not photo: return
        
        self.current_photo = photo
        self.programmatically_setting = True
        try:
            for key, value in photo.get_settings().items():
                var = getattr(self, f"photo_{key}")
                var.set(value)
        finally:
            self.programmatically_setting = False

    def apply_photo_settings(self, event=None):
        if not self.current_photo:
            messagebox.showinfo("No Selection", "Please select a photo to apply settings.")
            return
        
        try:
            settings = {
                "width": self.photo_width.get(), "height": self.photo_height.get(),
                "copies": self.photo_copies.get(), "brightness": self.photo_brightness.get(),
                "resize_mode": self.photo_resize_mode.get()
            }
            self.current_photo.update_settings(settings)
            self.schedule_render(force_relayout=True)
            self.status_var.set(f"Settings applied to '{os.path.basename(self.current_photo.path)}'.")
        except tk.TclError:
            messagebox.showerror("Invalid Input", "Please ensure all settings have valid numeric values.")

    def apply_settings_to_all(self, event=None):
        if not self.photos:
            messagebox.showinfo("No Photos", "There are no photos to apply settings to.")
            return
        
        if not messagebox.askyesno("Confirm", f"Apply current settings to all {len(self.photos)} photos?"):
            return

        try:
            settings = {
                "width": self.photo_width.get(), "height": self.photo_height.get(),
                "copies": self.photo_copies.get(), "brightness": self.photo_brightness.get(),
                "resize_mode": self.photo_resize_mode.get()
            }
            for photo in self.photos:
                photo.update_settings(settings)
            
            self.schedule_render(force_relayout=True)
            self.status_var.set(f"Settings applied to all {len(self.photos)} photos.")
        except tk.TclError:
            messagebox.showerror("Invalid Input", "Please ensure all settings have valid numeric values.")

    def edit_photo(self):
        if not self.current_photo:
            messagebox.showinfo("No Selection", "Please select a photo to edit.")
            return
            
        dialog = EditPhotoDialog(self.root, self.current_photo.path)
        self.root.wait_window(dialog)
        
        if dialog.result:
            new_path = dialog.result
            self.clear_caches_for_path(self.current_photo.path)
            self.current_photo.path = new_path
            
            try:
                with Image.open(new_path) as img:
                    self.current_photo.original_width, self.current_photo.original_height = img.size
                    self.current_photo.icc_profile_data = img.info.get('icc_profile')
            except Exception as e:
                print(f"ERROR: Could not get new dimensions after crop: {e}")

            item_id = self.get_item_id_by_photo_id(self.current_photo.id)
            if item_id:
                self.executor.submit(self.load_and_update_list_thumbnail, item_id, self.current_photo)
            
            self.schedule_render(force_relayout=True)
            self.status_var.set("Photo successfully cropped.")

    def remove_selected_photo(self):
        selection = self.photo_tree.selection()
        if not selection: return
        item_id = selection[0]
        
        photo = self.get_photo_by_item_id(item_id)
        if not photo: return

        self.clear_caches_for_path(photo.path)
        self.photos.remove(photo)
        
        if item_id in self.list_photo_thumbnails: del self.list_photo_thumbnails[item_id]
        if item_id in self.item_id_to_photo_id: del self.item_id_to_photo_id[item_id]
        self.photo_tree.delete(item_id)
        
        if self.current_photo and self.current_photo.id == photo.id:
            self.current_photo = None
        
        self.schedule_render(force_relayout=True)
        self.status_var.set("Photo removed.")

    def remove_all_photos(self):
        if not self.photos: return
        if not messagebox.askyesno("Confirm", "Are you sure you want to remove all photos?"):
            return
            
        self.photo_tree.delete(*self.photo_tree.get_children())
        self.photos.clear()
        self.item_id_to_photo_id.clear()
        self.list_photo_thumbnails.clear()
        self.layouts.clear()
        self.current_photo = None
        
        self.clear_caches(aggressive=True)
        
        self.schedule_render(force_relayout=True)
        self.status_var.set("All photos removed.")

    def copy_photo(self, event=None):
        if not self.current_photo: return
        self._internal_copy_photo_object = self.current_photo
        self.status_var.set(f"Copied '{os.path.basename(self.current_photo.path)}' for duplication.")

    def paste_photo(self, event=None):
        if self._internal_copy_photo_object:
            photo_to_duplicate = self._internal_copy_photo_object
            
            new_photo = Photo(
                path=photo_to_duplicate.path,
                original_width=photo_to_duplicate.original_width,
                original_height=photo_to_duplicate.original_height,
                icc_profile_data=photo_to_duplicate.icc_profile_data,
                **photo_to_duplicate.get_settings()
            )
            self.photos.append(new_photo)
            
            placeholder = ImageTk.PhotoImage(Image.new("RGB", (40, 40), "gray"))
            item_id = self.photo_tree.insert("", "end", text="", image=placeholder, values=(os.path.basename(new_photo.path),))
            self.list_photo_thumbnails[item_id] = placeholder
            self.item_id_to_photo_id[item_id] = new_photo.id
            self.executor.submit(self.load_and_update_list_thumbnail, item_id, new_photo)
            
            self.status_var.set(f"Duplicated '{os.path.basename(new_photo.path)}'.")
            self.schedule_render(force_relayout=True)
            self._internal_copy_photo_object = None
            return

        try:
            clipboard_content = ImageGrab.grabclipboard()
        except Exception as e:
            print(f"WARNING: Could not access system clipboard: {e}")
            messagebox.showwarning("Clipboard Error", "Could not access the system clipboard.")
            return

        pasted_files = []
        if isinstance(clipboard_content, Image.Image):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="pasted_") as tmp:
                    clipboard_content.save(tmp.name, "PNG")
                    pasted_files.append(tmp.name)
            except Exception as e:
                messagebox.showerror("Error", f"Could not save pasted image: {e}")
                return
        elif isinstance(clipboard_content, list):
            pasted_files.extend(clipboard_content)

        if pasted_files:
            self.add_photos(pasted_files)
        else:
            messagebox.showinfo("Clipboard Empty", "No image or image file found on the clipboard.")

    def load_original_image(self, path):
        """Loads the full-resolution original image from disk and handles transparency."""
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        
        # NEW: Handle transparent images by converting to RGB with white background
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Create a white background image
            background = Image.new('RGB', img.size, (255, 255, 255))
            # Paste the image onto the background using the alpha channel as mask
            if img.mode in ('RGBA', 'LA'):
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            else:
                background.paste(img)
            img = background
        
        return img

    def _get_rendering_intent_enum(self):
        """Converts the string variable to an ImageCms enum."""
        intent_str = self.rendering_intent_var.get()
        if intent_str == "Perceptual":
            return ImageCms.Intent.PERCEPTUAL
        return ImageCms.Intent.RELATIVE_COLORIMETRIC

    def _get_cms_profile(self, icc_profile_data):
        """
        Creates an ImageCmsProfile object from raw profile data,
        with a robust fallback to the sRGB working profile.
        """
        if icc_profile_data:
            try:
                return ImageCms.ImageCmsProfile(io.BytesIO(icc_profile_data))
            except Exception as e:
                print(f"WARNING: Could not read embedded ICC profile: {e}. Falling back to sRGB.")
        
        return self.srgb_profile

    def get_working_image(self, path):
        """
        Memory-safe image loader for previews. Gets an image from cache, or loads and
        downscales it if it's very large, to conserve RAM.
        This function returns the image with its embedded profile intact.
        """
        if path in self.master_image_cache:
            self.master_image_cache.move_to_end(path)
            return self.master_image_cache[path]
        
        try:
            img = self.load_original_image(path)
            
            if max(img.width, img.height) > PREVIEW_DOWNSCALE_DIMENSION_PX:
                print(f"INFO: Downscaling large image '{os.path.basename(path)}' for preview cache.")
                img.thumbnail((PREVIEW_DOWNSCALE_DIMENSION_PX, PREVIEW_DOWNSCALE_DIMENSION_PX), Image.LANCZOS)

            self.master_image_cache[path] = img
            if len(self.master_image_cache) > MASTER_CACHE_SIZE:
                self.master_image_cache.popitem(last=False)
            return img
        except Exception as e:
            print(f"ERROR: Failed to load working image {path}: {e}")
            return Image.new("RGB", (100, 100), "red")

    def _perform_resize(self, img, width, height):
        if self.gpu_manager:
            return self.gpu_manager.gpu_resize(img, width, height)
        return img.resize((width, height), Image.LANCZOS)

    def _should_auto_rotate(self, img_width, img_height, target_width, target_height):
        """
        NEW: Determines if rotating the image 90° would provide a better fit
        for the target dimensions by comparing aspect ratios.
        """
        if img_width == 0 or img_height == 0 or target_width == 0 or target_height == 0:
            return False
            
        img_aspect = img_width / img_height
        target_aspect = target_width / target_height
        
        # Calculate how much cropping would be needed for both orientations
        crop_ratio_original = min(target_width / img_width, target_height / img_height)
        crop_ratio_rotated = min(target_width / img_height, target_height / img_width)
        
        # If rotating reduces the amount of cropping needed, do it
        return crop_ratio_rotated > crop_ratio_original

    def _resize_image(self, img, target_width, target_height, rotated=False, resize_mode="crop"):
        """
        UPDATED: Fixed intelligent auto-rotation to work correctly with layout rotation.
    
        The key: When rotated=True, the layout has ALREADY accounted for rotation in its
        placement. We just need to fit the image into the target space correctly.
        """
        try:
            img_copy = img.copy()
        
            # Determine what dimensions we're actually trying to match
            # If layout wants rotation, it's asking us to rotate the photo box,
            # which means we need to fit the image into swapped dimensions
            actual_target_width = target_height if rotated else target_width
            actual_target_height = target_width if rotated else target_height
        
            # Check if auto-rotating the SOURCE image would improve fit
            if self._should_auto_rotate(img_copy.width, img_copy.height, actual_target_width, actual_target_height):
                img_copy = img_copy.rotate(90, expand=True)
        
            # Now resize to the ACTUAL target dimensions (which are swapped if rotated)
            orig_width, orig_height = img_copy.size
            if orig_width == 0 or orig_height == 0: 
                return Image.new("RGB", (int(target_width), int(target_height)), "red")

            if resize_mode == "stretch":
                result = self._perform_resize(img_copy, int(actual_target_width), int(actual_target_height))
            elif resize_mode == "crop":
                scale = max(actual_target_width / orig_width, actual_target_height / orig_height)
                resized = self._perform_resize(img_copy, int(orig_width * scale), int(orig_height * scale))
                left = (resized.width - actual_target_width) / 2
                top = (resized.height - actual_target_height) / 2
                result = resized.crop((left, top, left + actual_target_width, top + actual_target_height))
            else:  # "fit"
                result = Image.new("RGB", (int(actual_target_width), int(actual_target_height)), "white")
                scale = min(actual_target_width / orig_width, actual_target_height / orig_height)
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
            
                if new_width > 0 and new_height > 0:
                    fitted_img = self._perform_resize(img_copy, new_width, new_height)
                    left = (actual_target_width - new_width) // 2
                    top = (actual_target_height - new_height) // 2
                    result.paste(fitted_img, (left, top))
        
            # If layout wanted rotation, rotate the final result to match
            if rotated:
                result = result.rotate(-90, expand=True)
        
            return result
        except Exception as e:
            print(f"ERROR: Failed during image resize: {e}")
            return Image.new("RGB", (int(target_width), int(target_height)), "red")

    def clear_caches(self, aggressive=False):
        """Clears image caches to free memory."""
        print("INFO: Clearing image caches...")
        cleared_items = len(self.thumbnail_cache) + len(self.processed_image_cache)
        self.thumbnail_cache.clear()
        self.processed_image_cache.clear()
        if aggressive:
            cleared_items += len(self.master_image_cache)
            self.master_image_cache.clear()
        
        gc.collect()
        print(f"INFO: Caches cleared. Removed approx {cleared_items} items.")
        self.status_var.set("Image caches cleared to free memory.")

    def clear_caches_for_path(self, path):
        if path in self.master_image_cache:
            del self.master_image_cache[path]

        # Use the new namedtuple structure for checking keys
        thumb_keys_to_del = [key for key in self.thumbnail_cache if key.path == path]
        for key in thumb_keys_to_del:
            if key in self.thumbnail_cache: del self.thumbnail_cache[key]
        
        proc_keys_to_del = [key for key in self.processed_image_cache if key.path == path]
        for key in proc_keys_to_del:
            if key in self.processed_image_cache: del self.processed_image_cache[key]

        gc.collect()

    def schedule_render(self, force_relayout=False):
        if self.render_after_id:
            self.root.after_cancel(self.render_after_id)
        self.render_after_id = self.root.after(250, self.process_render_queue, force_relayout)

    def process_render_queue(self, force_relayout=False):
        if self.is_rendering: return
        self.is_rendering = True
        self.status_var.set("Updating preview...")
        
        new_cache_key = self._get_layout_cache_key()
        
        if force_relayout or new_cache_key != self.layout_cache_key or not self.layout_cache:
            self.layout_cache_key = new_cache_key
            
            paper_name = self.paper_size_var.get()
            paper_w, paper_h = self.user_defined_size if paper_name == "User Defined" else next((w, h) for n, w, h in self.paper_sizes if n == paper_name)
            
            printable_w_mm, printable_h_mm = paper_w, paper_h
            if WIN32_AVAILABLE:
                printer_name = self.printer_var.get()
                if printer_name and printer_name != "No printers found":
                    printer_info = self._get_printer_info(printer_name)
                    if printer_info:
                        printable_w_mm = printer_info["printable_width_mm"]
                        printable_h_mm = printer_info["printable_height_mm"]
            
            future = self.executor.submit(self.generate_layout, paper_w, paper_h, printable_w_mm, printable_h_mm)
            future.add_done_callback(lambda f: self.queue_ui_update(self.on_layout_generated, f, priority=5))
        else:
            self.layouts, _, _ = self.layout_cache
            self.current_page_index = min(self.current_page_index, len(self.layouts) - 1)
            if self.current_page_index < 0: self.current_page_index = 0
            self.render_preview()
            self.update_page_controls()
            self.is_rendering = False
            self.status_var.set("Ready (from cache).")

    def _get_layout_cache_key(self):
        """
        Generates a key representing the current state of all parameters that
        affect the geometric layout of photos on pages. If this key hasn't
        changed, the expensive layout calculation can be skipped. The key includes
        photo dimensions and copies, but intentionally excludes visual properties
        like brightness, which don't alter placement.
        """
        paper_name = self.paper_size_var.get()
        paper_size = self.user_defined_size if paper_name == "User Defined" else paper_name
        photo_props = tuple(sorted((p.id, p.width, p.height, p.copies) for p in self.photos))
        return (self.printer_var.get(), paper_size, self.margin.get(), self.interval.get(), 
                self.layout_mode.get(), self.symmetrical_orientation.get() if self.layout_mode.get() == "Symmetrical" else "Efficient", 
                self.rotation_allowed.get() if self.layout_mode.get() == "Efficient" else False,
                photo_props)

    def on_layout_generated(self, future):
        try:
            with self.busy_cursor():
                result = future.result()
                if result is None: raise ValueError("Layout generation failed unexpectedly.")
                self.layouts, paper_size_mm, printable_area_mm = result
                self.layout_cache = (self.layouts, paper_size_mm, printable_area_mm)
                self.current_page_index = 0
                self.render_preview()
                self.update_page_controls()
        except Exception as e:
            print(f"ERROR: Layout generation failed: {e}")
            self.show_render_error(str(e))
        finally:
            self.is_rendering = False
            self.status_var.set("Ready.")

    def show_render_error(self, error):
        self.canvas.delete("all")
        self.canvas.create_text(10, 10, anchor="nw", fill="red",
                                text=f"Layout Error: {error}",
                                width=self.canvas.winfo_width() - 20)

    def generate_layout(self, physical_paper_w, physical_paper_h, printable_w, printable_h):
        """Dispatches to the correct layout generator based on the selected mode."""
        mode = self.layout_mode.get()
        if mode == "Symmetrical":
            return self.generate_layout_symmetrical(physical_paper_w, physical_paper_h, printable_w, printable_h)
        else:  # Default to Efficient
            return self.generate_layout_efficient(physical_paper_w, physical_paper_h, printable_w, printable_h)

    def generate_layout_symmetrical(self, physical_paper_w, physical_paper_h, printable_w, printable_h):
        """
        UPDATED: Generates a symmetrical layout with centered rows and proper grid centering.
        """
        margin, interval = self.margin.get(), self.interval.get()
        layout_width, layout_height = printable_w, printable_h
    
        # Calculate the actual printable area after accounting for margins
        printable_area_width = layout_width - (2 * margin)
        printable_area_height = layout_height - (2 * margin)
    
        # NEW: Apply orientation from radio buttons
        orientation = self.symmetrical_orientation.get()
        if orientation == "Landscape":
            # Swap width and height for all photos in landscape mode
            photo_widths = [max(p.width, p.height) for p in self.photos]
            photo_heights = [min(p.width, p.height) for p in self.photos]
        else:  # Portrait
            photo_widths = [min(p.width, p.height) for p in self.photos]
            photo_heights = [max(p.width, p.height) for p in self.photos]
    
        # Use the first photo's dimensions as the uniform size for all photos in symmetrical mode
        if self.photos:
            uniform_width = photo_widths[0]
            uniform_height = photo_heights[0]
        else:
            return [], (physical_paper_w, physical_paper_h), (printable_w, printable_h)

        # Check if the uniform photo size fits within the printable area
        if uniform_width > printable_area_width or uniform_height > printable_area_height:
            message = f"The selected photo size ({uniform_width}x{uniform_height}mm) is too large for the printable area with current margins."
            self.queue_ui_update(messagebox.showwarning, "Layout Warning", message, priority=8)
            return [], (physical_paper_w, physical_paper_h), (printable_w, printable_h)

        # Build placeable copies with uniform dimensions
        placeable_copies = []
        for i, p in enumerate(self.photos):
            for _ in range(p.copies):
                placeable_copies.append({
                    "photo_obj": p, 
                    "width": uniform_width, 
                    "height": uniform_height
                })
    
        if not placeable_copies:
            return [], (physical_paper_w, physical_paper_h), (printable_w, printable_h)

        all_layouts = []
    
        while placeable_copies:
            page_layout = []
            current_y = margin  # Start from top margin
        
            remaining_for_next_page = []
            current_row = []
            current_row_width = 0
        
            # Calculate how many photos can fit in one row
            photos_per_row = 0
            test_width = 0
            while test_width + uniform_width <= printable_area_width:
                if photos_per_row > 0:
                    test_width += interval  # Add interval between photos
                test_width += uniform_width
                photos_per_row += 1
        
            photos_per_row = max(1, photos_per_row)  # At least one photo per row
        
            # Calculate the total width of a full row
            full_row_width = (photos_per_row * uniform_width) + ((photos_per_row - 1) * interval)
        
            # Calculate horizontal starting position to center the entire grid
            grid_start_x = margin + (printable_area_width - full_row_width) / 2

            for item in placeable_copies:
                photo = item["photo_obj"]
                photo_w, photo_h = uniform_width, uniform_height
    
                # If current row is full, move to next row
                if len(current_row) >= photos_per_row:
                    # Place the current row
                    current_x = grid_start_x  # Start from the left of the centered grid
                    for row_item in current_row:
                        page_layout.append({
                            "photo_obj": row_item["photo_obj"], 
                            "x": current_x, 
                            "y": current_y,
                            "width": uniform_width, 
                            "height": uniform_height,
                            "rotated": (orientation == "Landscape")
                        })
                        current_x += uniform_width + interval
        
                    # Move to next row
                    current_y += uniform_height + interval
                    current_row = []
                    current_row_width = 0
    
                # Check if we have vertical space for another row
                if (current_y + uniform_height) > (layout_height - margin):
                    remaining_for_next_page.append(item)
                    continue
    
                # Add photo to current row
                current_row.append(item)
                current_row_width += uniform_width + (interval if len(current_row) > 1 else 0)

            # Process any remaining photos in the last row
            if current_row:
                # For the last row, start from the left of the centered grid
                current_x = grid_start_x
                for row_item in current_row:
                    page_layout.append({
                        "photo_obj": row_item["photo_obj"], 
                        "x": current_x, 
                        "y": current_y,
                        "width": uniform_width, 
                        "height": uniform_height,
                        "rotated": (orientation == "Landscape")
                    })
                    current_x += uniform_width + interval
        
            if not page_layout: 
                break  # Stop if a page cannot hold any more photos
            
            all_layouts.append(page_layout)
            placeable_copies = remaining_for_next_page

        return all_layouts, (physical_paper_w, physical_paper_h), (printable_w, printable_h)

    def generate_layout_efficient(self, physical_paper_w, physical_paper_h, printable_w, printable_h):
        """
        FIXED: Proper bin-packing algorithm for efficient mode with rotation support.
        Implements the shelf algorithm with best-fit strategy and compaction.
        """
        margin, interval = self.margin.get(), self.interval.get()
        layout_width, layout_height = printable_w, printable_h
        max_photo_w, max_photo_h = layout_width - (2 * margin), layout_height - (2 * margin)

        # Build list of all photo copies to place
        placeable_copies = []
        excluded_photos = {}
        
        for p in self.photos:
            can_fit_original = p.width <= max_photo_w and p.height <= max_photo_h
            can_fit_rotated = self.rotation_allowed.get() and p.width != p.height and p.height <= max_photo_w and p.width <= max_photo_h

            if can_fit_original or can_fit_rotated:
                for _ in range(p.copies):
                    placeable_copies.append(p)
            elif p.id not in excluded_photos:
                excluded_photos[p.id] = p
        
        if excluded_photos:
            names = [os.path.basename(p.path) for p in excluded_photos.values()]
            message = (f"{len(names)} photo(s) could not be placed as they are too large for the printable area:\n\n" + "\n".join(names))
            self.queue_ui_update(messagebox.showwarning, "Layout Warning", message, priority=8)
        
        if not placeable_copies:
            return [], (physical_paper_w, physical_paper_h), (printable_w, printable_h)

        # Sort by area (largest first) for better packing
        placeable_copies.sort(key=lambda p: p.width * p.height, reverse=True)

        all_layouts = []
        
        while placeable_copies:
            page_layout = []
            shelves = []  # List of (y, height, used_width) tuples
            next_y = margin
            
            remaining_for_next_page = []

            for photo in placeable_copies:
                best_fit = None
                best_wasted_space = float('inf')
                best_orientation = None
                
                # Consider both orientations if rotation is allowed
                orientations = []
                if photo.width <= max_photo_w and photo.height <= max_photo_h:
                    orientations.append((photo.width, photo.height, False))
                if self.rotation_allowed.get() and photo.width != photo.height and photo.height <= max_photo_w and photo.width <= max_photo_h:
                    orientations.append((photo.height, photo.width, True))
                
                # Try to fit in existing shelves
                for w, h, rotated in orientations:
                    for i, (shelf_y, shelf_height, used_width) in enumerate(shelves):
                        # Check if photo fits in this shelf and within width
                        if h <= shelf_height and (used_width + w) <= (layout_width - margin):
                            # Calculate wasted space (area above the photo in the shelf)
                            wasted_space = (shelf_height - h) * w
                            if wasted_space < best_wasted_space:
                                best_fit = i
                                best_wasted_space = wasted_space
                                best_orientation = (w, h, rotated)
                    
                    # Try to fit in a new shelf
                    if (next_y + h) <= (layout_height - margin):
                        # Calculate wasted space for new shelf (entire row width minus photo width)
                        wasted_space = (layout_width - (2 * margin) - w) * h
                        if wasted_space < best_wasted_space:
                            best_fit = -1  # New shelf
                            best_wasted_space = wasted_space
                            best_orientation = (w, h, rotated)
                
                if best_orientation:
                    w, h, rotated = best_orientation
                    
                    if best_fit >= 0:  # Place in existing shelf
                        shelf_y, shelf_height, used_width = shelves[best_fit]
                        x_pos = margin + used_width
                        y_pos = shelf_y
                        
                        # Update shelf
                        shelves[best_fit] = (shelf_y, shelf_height, used_width + w + interval)
                        
                        page_layout.append({
                            "photo_obj": photo,
                            "x": x_pos,
                            "y": y_pos,
                            "width": w,
                            "height": h,
                            "rotated": rotated
                        })
                    else:  # Create new shelf
                        x_pos = margin
                        y_pos = next_y
                        
                        # Add new shelf
                        shelves.append((y_pos, h, w + interval))
                        next_y += h + interval
                        
                        page_layout.append({
                            "photo_obj": photo,
                            "x": x_pos,
                            "y": y_pos,
                            "width": w,
                            "height": h,
                            "rotated": rotated
                        })
                else:
                    # Cannot place this photo on current page
                    remaining_for_next_page.append(photo)
            
            if not page_layout:
                # If we can't place any photos on a new page, stop to avoid infinite loop
                break
                
            # Compact the layout by moving photos upward
            self._compact_layout(page_layout, shelves, margin, interval, layout_width, layout_height)
            
            all_layouts.append(page_layout)
            placeable_copies = remaining_for_next_page

        return all_layouts, (physical_paper_w, physical_paper_h), (printable_w, printable_h)

    def _compact_layout(self, page_layout, shelves, margin, interval, layout_width, layout_height):
        """
        Compact the layout by moving photos upward and leftward to reduce wasted space.
        """
        # Sort shelves by y-position
        shelves.sort(key=lambda s: s[0])
        
        # Group photos by their current shelf
        photos_by_shelf = [[] for _ in shelves]
        for photo in page_layout:
            for i, (shelf_y, shelf_height, _) in enumerate(shelves):
                if abs(photo["y"] - shelf_y) < 1:  # Consider photos in the same shelf if y is close
                    photos_by_shelf[i].append(photo)
                    break
        
        # Compact vertically by moving shelves upward
        current_y = margin
        for i, shelf_photos in enumerate(photos_by_shelf):
            if not shelf_photos:
                continue
                
            # Find the minimum y needed for this shelf
            shelf_height = max(photo["height"] for photo in shelf_photos)
            
            # Move all photos in this shelf to current_y
            for photo in shelf_photos:
                photo["y"] = current_y
            
            # Update shelf position
            shelves[i] = (current_y, shelf_height, shelves[i][2])
            current_y += shelf_height + interval
        
        # Compact horizontally within each shelf
        for i, shelf_photos in enumerate(photos_by_shelf):
            if not shelf_photos:
                continue
                
            # Sort photos in shelf by x-position
            shelf_photos.sort(key=lambda p: p["x"])
            
            current_x = margin
            for photo in shelf_photos:
                photo["x"] = current_x
                current_x += photo["width"] + interval
            
            # Update shelf used width
            shelf_y, shelf_height, _ = shelves[i]
            shelves[i] = (shelf_y, shelf_height, current_x - margin)
    
    def _generate_gamut_warning_overlay(self, image, src_profile, proof_profile):
        """Generates a semi-transparent overlay for out-of-gamut pixels."""
        if not NUMPY_AVAILABLE: return None
        try:
            intent = self._get_rendering_intent_enum()
            transform_fwd = ImageCms.buildTransform(src_profile, proof_profile, "RGB", "RGB", intent, 0)
            transform_bwd = ImageCms.buildTransform(proof_profile, src_profile, "RGB", "RGB", intent, 0)

            img_roundtrip = ImageCms.applyTransform(ImageCms.applyTransform(image, transform_fwd), transform_bwd)
            
            img_np = np.array(image).astype(np.float32)
            roundtrip_np = np.array(img_roundtrip).astype(np.float32)
            
            diff = np.sum(np.abs(img_np - roundtrip_np), axis=2)
            mask_np = (diff > 2.0) 

            if not np.any(mask_np): return None

            warning_color_layer = np.zeros_like(img_np, dtype=np.uint8)
            warning_color_layer[mask_np] = GAMUT_WARNING_COLOR
            overlay = Image.fromarray(warning_color_layer, 'RGB')
            return overlay
        except Exception as e:
            print(f"ERROR: Gamut warning generation failed: {e}")
            return None

    def render_preview(self):
        """
        Renders the layout preview using a soft-proofing transform to simulate
        the final printed output on the calibrated monitor.
        """
        self.canvas.delete("all")
        if not self.layouts or not self.canvas.winfo_viewable() or self.canvas.winfo_width() < 10:
            self.update_page_controls()
            return

        printer_name = self.printer_var.get() if WIN32_AVAILABLE else None
        printer_icc_data = self._get_printer_icc_profile(printer_name) if printer_name else None
        proof_profile = self._get_cms_profile(printer_icc_data)
        proof_profile_id = printer_name if printer_icc_data else "sRGB"

        layout = self.layouts[self.current_page_index]
        _, paper_size_mm, printable_area_mm = self.layout_cache
        physical_w_mm, physical_h_mm = paper_size_mm
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale = min((canvas_w - 10) / physical_w_mm, (canvas_h - 10) / physical_h_mm)
        offset_x = (canvas_w - physical_w_mm * scale) / 2
        offset_y = (canvas_h - physical_h_mm * scale) / 2

        self.canvas.create_rectangle(offset_x, offset_y, offset_x + physical_w_mm * scale, offset_y + physical_h_mm * scale, outline="#cccccc", width=1, fill="white")

        for item in layout:
            photo = item["photo_obj"]
            tw, th = int(item["width"] * scale), int(item["height"] * scale)
            if tw < 1 or th < 1: continue
            
            x_pos = offset_x + ((physical_w_mm - printable_area_mm[0]) / 2 + item["x"]) * scale
            y_pos = offset_y + ((physical_h_mm - printable_area_mm[1]) / 2 + item["y"]) * scale
            
            # UPDATE: Using new namedtuple keys for improved clarity.
            processed_cache_key = ProcessedCacheKey(
                path=photo.path,
                brightness=photo.brightness,
                profile_id=proof_profile_id,
                intent=self.rendering_intent_var.get()
            )
            tk_cache_key = ThumbnailCacheKey(
                path=photo.path,
                brightness=photo.brightness,
                profile_id=proof_profile_id,
                intent=self.rendering_intent_var.get(),
                width=tw,
                height=th,
                gamut_warning=self.gamut_warning_var.get()
            )
            
            try:
                if tk_cache_key in self.thumbnail_cache:
                    tkimg = self.thumbnail_cache[tk_cache_key]
                else:
                    if processed_cache_key in self.processed_image_cache:
                        pil_img_converted = self.processed_image_cache[processed_cache_key]
                    else:
                        working_img = self.get_working_image(photo.path)
                        
                        bright_img = working_img
                        if photo.brightness != 0:
                            factor = 1.0 + (photo.brightness / 100.0)
                            if self.gpu_manager:
                                bright_img = self.gpu_manager.gpu_adjust_brightness(working_img, factor)
                            else:
                                enhancer = ImageEnhance.Brightness(working_img)
                                bright_img = enhancer.enhance(factor)
                        
                        source_profile = self._get_cms_profile(photo.icc_profile_data)
                        
                        intent = self._get_rendering_intent_enum()
                        proof_transform = ImageCms.buildProofTransform(
                           inputProfile=source_profile, outputProfile=self.monitor_icc_profile,
                           proofProfile=proof_profile, inMode="RGB", outMode="RGB",
                           renderingIntent=intent, proofRenderingIntent=ImageCms.Intent.ABSOLUTE_COLORIMETRIC)
                        pil_img_converted = ImageCms.applyTransform(bright_img, proof_transform)
                        
                        self.processed_image_cache[processed_cache_key] = pil_img_converted
                        if len(self.processed_image_cache) > PROCESSED_CACHE_SIZE:
                            self.processed_image_cache.popitem(last=False)

                    pil_img = self._resize_image(pil_img_converted, tw, th, item["rotated"], photo.resize_mode)

                    if self.gamut_warning_var.get():
                        working_img_for_gamut = self.get_working_image(photo.path)
                        source_profile_for_gamut = self._get_cms_profile(photo.icc_profile_data)
                        overlay = self._generate_gamut_warning_overlay(working_img_for_gamut, source_profile_for_gamut, proof_profile)
                        if overlay:
                            overlay_resized = self._resize_image(overlay, tw, th, item["rotated"], photo.resize_mode)
                            pil_img = Image.blend(pil_img, overlay_resized, 0.4)

                    tkimg = ImageTk.PhotoImage(pil_img)
                    self.thumbnail_cache[tk_cache_key] = tkimg
                    if len(self.thumbnail_cache) > THUMBNAIL_CACHE_SIZE:
                        self.thumbnail_cache.popitem(last=False)

                self.canvas.create_image(x_pos, y_pos, image=tkimg, anchor="nw")
                if self.outline.get():
                    self.canvas.create_rectangle(x_pos, y_pos, x_pos + tw, y_pos + th, outline="#dddddd", width=1)
                self.canvas.image = tkimg
            except Exception as e:
                print(f"ERROR: Failed to render preview for one image: {e}")
                self.canvas.create_rectangle(x_pos, y_pos, x_pos + tw, y_pos + th, outline="red", fill="gray", width=1)


    def _parse_page_ranges(self, range_string, max_page):
        pages = set()
        if not range_string: return []
        try:
            for part in range_string.split(','):
                part = part.strip()
                if not part: continue
                if '-' in part:
                    start, end = map(int, part.split('-', 1))
                    if not (1 <= start <= end <= max_page): raise ValueError(f"Range '{start}-{end}' is invalid.")
                    pages.update(range(start, end + 1))
                else:
                    page = int(part)
                    if not (1 <= page <= max_page): raise ValueError(f"Page number '{page}' is out of range (1-{max_page}).")
                    pages.add(page)
            return sorted([p - 1 for p in pages])
        except ValueError as e:
            messagebox.showerror("Invalid Page Range", f"Invalid input: '{range_string}'.\nExample: 1,3,5-8\nDetails: {e}")
            return None

    def print_layout(self):
        if not WIN32_AVAILABLE:
            messagebox.showerror("Unsupported", "Printing is only supported on Windows.")
            return
        if not self.layouts:
            messagebox.showwarning("No Photos", "Please add photos and generate a preview before printing.")
            return
        
        printer_name = self.printer_var.get()
        if not printer_name or printer_name == "No printers found":
            messagebox.showerror("Printer Error", "No printer selected.")
            return

        num_pages = len(self.layouts)
        print_option = self.print_range_var.get()
        indices_to_print = []

        if print_option == "all": indices_to_print = list(range(num_pages))
        elif print_option == "current":
            if 0 <= self.current_page_index < num_pages: indices_to_print = [self.current_page_index]
        elif print_option == "specific":
            indices_to_print = self._parse_page_ranges(self.specific_pages_entry.get(), num_pages)
            if indices_to_print is None: return

        if not indices_to_print:
            messagebox.showerror("Print Error", "No pages selected for printing.")
            return

        self.status_var.set(f"Preparing to print {len(indices_to_print)} page(s)...")
        self.executor.submit(self._print_job_thread, printer_name, indices_to_print)

    def _print_job_thread(self, printer_name, page_indices):
        try:
            with self.busy_cursor():
                self._print_pages_windows(printer_name, page_indices)
            self.queue_ui_update(messagebox.showinfo, "Print Status", "Print job sent to printer successfully.", priority=5)
        except Exception as e:
            print(f"ERROR: Printing failed: {e}")
            self.queue_ui_update(messagebox.showerror, "Print Error", f"An error occurred during printing:\n{e}", priority=1)
        finally:
            self.queue_ui_update(self.status_var.set, "Ready.", priority=10)

    def _print_pages_windows(self, printer_name, page_indices):
        from PIL import ImageWin
        hdc = None
        try:
            devmode = self._get_configured_devmode(printer_name)
            hdc_handle = win32gui.CreateDC("WINSPOOL", printer_name, devmode)
            hdc = win32ui.CreateDCFromHandle(hdc_handle)
            hdc.StartDoc(f"Neka Photocopy Job")

            for i, page_index in enumerate(page_indices):
                self.queue_ui_update(self.status_var.set, f"Printing page {i+1}/{len(page_indices)}...", priority=10)
                
                page_layout = self.layouts[page_index]
                printer_info = self._get_printer_info(printer_name)
                if not printer_info: raise Exception("Could not retrieve printer-specific dimensions.")
                
                page_image = self._create_final_print_image(page_layout, printer_info, printer_name)
                
                hdc.StartPage()
                dib = ImageWin.Dib(page_image)
                printable_w_px = hdc.GetDeviceCaps(win32con.HORZRES)
                printable_h_px = hdc.GetDeviceCaps(win32con.VERTRES)
                dib.draw(hdc.GetHandleOutput(), (0, 0, printable_w_px, printable_h_px))
                hdc.EndPage()
        finally:
            if hdc:
                hdc.EndDoc()
                hdc.DeleteDC()

    def _create_final_print_image(self, page_layout, printer_info, printer_name):
        """
        Creates the final, full-resolution page image for the printer, converting
        each photo from its source profile to the printer's specific profile.
        """
        dpi_x, dpi_y = printer_info["dpi_x"], printer_info["dpi_y"]
        printable_w_px, printable_h_px = printer_info["printable_width_px"], printer_info["printable_height_px"]

        printer_icc_data = self._get_printer_icc_profile(printer_name)
        target_profile = self._get_cms_profile(printer_icc_data)

        page_image = Image.new("RGB", (printable_w_px, printable_h_px), "white")
        draw = ImageDraw.Draw(page_image)

        for item in page_layout:
            try:
                photo = item["photo_obj"]
                master_img = self.load_original_image(photo.path)
                
                bright_img = master_img
                if photo.brightness != 0:
                    factor = 1.0 + (photo.brightness / 100.0)
                    if self.gpu_manager:
                        bright_img = self.gpu_manager.gpu_adjust_brightness(master_img, factor)
                    else:
                        enhancer = ImageEnhance.Brightness(master_img)
                        bright_img = enhancer.enhance(factor)

                source_profile = self._get_cms_profile(photo.icc_profile_data)
                intent = self._get_rendering_intent_enum()

                converted_img = ImageCms.profileToProfile(bright_img, source_profile, target_profile, renderingIntent=intent, outputMode='RGB')
                
                target_w_px = int(item["width"] * dpi_x / 25.4)
                target_h_px = int(item["height"] * dpi_y / 25.4)
                transformed_img = self._resize_image(converted_img, target_w_px, target_h_px, item["rotated"], photo.resize_mode)
                
                x_abs = int(item["x"] * dpi_x / 25.4)
                y_abs = int(item["y"] * dpi_y / 25.4)
                
                page_image.paste(transformed_img, (x_abs, y_abs))
                if self.outline.get():
                    draw.rectangle([x_abs, y_abs, x_abs + target_w_px - 1, y_abs + target_h_px - 1], outline="#cccccc", width=1)
            except Exception as e:
                print(f"ERROR: Failed to process one image for printing: {e}")
        return page_image

    def get_photo_by_item_id(self, item_id):
        photo_id = self.item_id_to_photo_id.get(item_id)
        return next((p for p in self.photos if p.id == photo_id), None) if photo_id is not None else None

    def get_item_id_by_photo_id(self, photo_id):
        return next((item_id for item_id, pid in self.item_id_to_photo_id.items() if pid == photo_id), None)

    def load_printers_background(self):
        self.status_var.set("Loading printers...")
        self.executor.submit(self._load_printers_task)
        
    def _load_printers_task(self):
        printers, default_printer = [], None
        try:
            printers = [p[2] for p in win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS)]
            default_printer = win32print.GetDefaultPrinter()
        except Exception as e:
            print(f"ERROR: Printer detection failed: {e}")
            printers = ["No printers found"]
        # Use medium-low priority for this background task update
        self.queue_ui_update(self._update_printers_ui, printers, default_printer, priority=15)
        
    def _update_printers_ui(self, printers, default_printer):
        if printers and printers[0] != "No printers found":
            self.printers = sorted(printers)
            self.printer_combo['values'] = self.printers
            if default_printer and default_printer in self.printers:
                self.printer_var.set(default_printer)
            elif self.printers:
                self.printer_var.set(self.printers[0])
        else:
            self.printers = ["No printers found"]
            self.printer_var.set(self.printers[0])
            self.printer_combo['values'] = self.printers
            
        self.status_var.set("Printers loaded.")
        self.on_printer_selected()

    def update_printer_settings_from_ui(self, printer_name):
        """
        Updates the printer settings based on UI selections.
        It first tries to find a matching standard paper size supported by the driver.
        If no match is found, or if "User Defined" is selected, it falls back
        to setting a custom paper size.
        """
        if not WIN32_AVAILABLE or not printer_name or printer_name == "No printers found":
            self.printer_settings = None
            return

        hprinter = None
        try:
            # Get printer handle, current settings (devmode), and port name
            hprinter = win32print.OpenPrinter(printer_name)
            # Use level 2 to get the pDevMode and pPortName
            printer_info_struct = win32print.GetPrinter(hprinter, 2)
            devmode = printer_info_struct["pDevMode"]
            port = printer_info_struct["pPortName"]

            # Determine the target paper dimensions from the UI
            selected_paper_str = self.paper_size_var.get()
            target_width_mm, target_height_mm = 0.0, 0.0
            is_user_defined = (selected_paper_str == "User Defined")

            if is_user_defined:
                target_width_mm, target_height_mm = self.user_defined_size
            else:
                paper_info = next((item for item in self.paper_sizes if item[0] == selected_paper_str), None)
                if paper_info:
                    _, target_width_mm, target_height_mm = paper_info

            # Attempt to find a matching standard paper code if not "User Defined"
            matching_code = None
            if not is_user_defined and target_width_mm > 0 and target_height_mm > 0:
                matching_code = self._find_matching_paper_code(printer_name, port, float(target_width_mm), float(target_height_mm))

            # Clear any previous paper-related settings to ensure a clean update
            devmode.Fields &= ~(win32con.DM_PAPERSIZE | win32con.DM_PAPERWIDTH | win32con.DM_PAPERLENGTH)

            # Apply the new settings based on whether a standard size was found
            if matching_code is not None:
                # A standard size was found in the driver, so we use its code.
                print(f"INFO: Found matching standard paper code {matching_code} for '{selected_paper_str}' ({target_width_mm}x{target_height_mm} mm).")
                devmode.PaperSize = matching_code
                devmode.Fields |= win32con.DM_PAPERSIZE
            else:
                # Fallback behavior: Use a custom user-defined size.
                if is_user_defined:
                    print(f"INFO: Using explicitly selected 'User Defined' size ({target_width_mm}x{target_height_mm} mm).")
                else:
                    print(f"INFO: No standard paper size found for '{selected_paper_str}'. Falling back to custom size ({target_width_mm}x{target_height_mm} mm).")
                
                if target_width_mm > 0 and target_height_mm > 0:
                    devmode.PaperSize = win32con.DMPAPER_USER
                    devmode.PaperWidth = int(target_width_mm * 10)  # Tenths of a millimeter
                    devmode.PaperLength = int(target_height_mm * 10) # Tenths of a millimeter
                    devmode.Fields |= (win32con.DM_PAPERSIZE | win32con.DM_PAPERWIDTH | win32con.DM_PAPERLENGTH)
                else:
                    # This case occurs if the selected paper size string is somehow invalid.
                    print(f"WARNING: Could not find dimensions for '{selected_paper_str}'. Using printer default.")

            # Set orientation and force application-level color management (preserves existing logic)
            devmode.Orientation = win32con.DMORIENT_PORTRAIT
            devmode.Fields |= win32con.DM_ORIENTATION

            self._force_application_color_management(devmode)

            # Apply the updated devmode to the printer
            win32print.DocumentProperties(self.root.winfo_id(), hprinter, printer_name, devmode, devmode, win32con.DM_IN_BUFFER | win32con.DM_OUT_BUFFER)
            self.printer_settings = devmode
            self.status_var.set(f"Printer settings updated for {selected_paper_str}.")

        except Exception as e:
            print(f"ERROR: Could not update printer DEVMODE settings for '{printer_name}': {e}")
            self.printer_settings = None
        finally:
            if hprinter:
                win32print.ClosePrinter(hprinter)

    def _force_application_color_management(self, devmode):
        """
        Forces the printer driver to let the application handle color management,
        which is essential for the ICC-based workflow to function correctly.
        """
        if not devmode: return
        try:
            _ = win32con.DMICM_HANDLED
        except AttributeError:
            win32con.DMICM_HANDLED = 4

        print("INFO: Forcing printer to Color mode with Application-Handled ICM.")
        devmode.Fields |= win32con.DM_COLOR | win32con.DM_ICMMETHOD
        devmode.Color = win32con.DMCOLOR_COLOR
        devmode.ICMMethod = win32con.DMICM_HANDLED
                
    def _find_matching_paper_code(self, printer_name, port, target_width_mm, target_height_mm):
        """
        Searches the printer driver's supported paper sizes for a match.
        Returns the matching paper code if found, None otherwise.
    
        Dimensions from DC_PAPERSIZE are in hundredths of millimeters (0.01mm).
        """
        if not WIN32_AVAILABLE:
            return None
    
        try:
            # Query printer capabilities
            paper_codes = win32print.DeviceCapabilities(
                printer_name, port, win32con.DC_PAPERS, None
            )
            paper_sizes = win32print.DeviceCapabilities(
                printer_name, port, win32con.DC_PAPERSIZE, None
            )
        
            # Validate returns
            if not paper_codes or not paper_sizes:
                return None
        
            if len(paper_codes) != len(paper_sizes):
                return None
        
            # Match tolerance
            tolerance_mm = 3.0
        
            # DEBUG: Print what we're looking for
            print(f"DEBUG: Searching for paper size: {target_width_mm}×{target_height_mm}mm")
            print(f"DEBUG: Found {len(paper_codes)} paper sizes from driver")
        
            # Iterate through all available paper sizes
            for i, code in enumerate(paper_codes):
                try:
                    size_data = paper_sizes[i]
                
                    # Handle both dict and tuple formats
                    if isinstance(size_data, dict):
                        width_hundredths = size_data.get('x', 0)
                        height_hundredths = size_data.get('y', 0)
                    elif isinstance(size_data, (tuple, list)) and len(size_data) >= 2:
                        width_hundredths = size_data[0]
                        height_hundredths = size_data[1]
                    else:
                        print(f"DEBUG: Skipping code={code}, invalid format: {type(size_data)}")
                        continue
                
                    # Convert to millimeters
                    width_mm = float(width_hundredths) / 10.0
                    height_mm = float(height_hundredths) / 10.0
                
                    # DEBUG: Print every paper size we're checking
                    print(f"DEBUG: Checking code={code}: {width_mm:.1f}×{height_mm:.1f}mm")
                
                    # Portrait match
                    width_diff = abs(width_mm - target_width_mm)
                    height_diff = abs(height_mm - target_height_mm)
                
                    print(f"DEBUG:   Portrait diff: width={width_diff:.2f}mm, height={height_diff:.2f}mm, tolerance={tolerance_mm}mm")
                
                    if width_diff <= tolerance_mm and height_diff <= tolerance_mm:
                        print(f"SUCCESS: Found matching paper - Code: {code}, "
                              f"Driver: {width_mm:.1f}×{height_mm:.1f}mm, "
                              f"Target: {target_width_mm}×{target_height_mm}mm")
                        return code
                
                    # Landscape match (swapped)
                    width_diff_ls = abs(width_mm - target_height_mm)
                    height_diff_ls = abs(height_mm - target_width_mm)
                
                    print(f"DEBUG:   Landscape diff: width={width_diff_ls:.2f}mm, height={height_diff_ls:.2f}mm")
                
                    if width_diff_ls <= tolerance_mm and height_diff_ls <= tolerance_mm:
                        print(f"SUCCESS: Found matching paper (landscape) - Code: {code}, "
                              f"Driver: {width_mm:.1f}×{height_mm:.1f}mm")
                        return code
                    
                except Exception as e:
                    print(f"DEBUG: Error processing code={code}: {e}")
                    continue
        
            print(f"INFO: No matching standard paper size found for {target_width_mm}×{target_height_mm}mm")
            return None
        
        except Exception as e:
            print(f"WARNING: Could not query capabilities: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_configured_devmode(self, printer_name):
        if not WIN32_AVAILABLE: return None
        if self.printer_settings: return self.printer_settings
        
        print("INFO: DEVMODE not cached, generating from UI state as fallback.")
        self.update_printer_settings_from_ui(printer_name)
        return self.printer_settings
        
    def _get_monitor_icc_profile(self):
        """
        Retrieves the ICC profile for the primary display and returns
        it as a CmsProfile object, falling back to the sRGB working profile.
        """
        if not WIN32_AVAILABLE: return self.srgb_profile
        try:
            hdc = win32gui.GetDC(0) 
            buffer_size = wintypes.DWORD(0)
            ctypes.windll.gdi32.GetICMProfileW(hdc, ctypes.byref(buffer_size), None)
            if buffer_size.value == 0:
                raise ValueError("No monitor profile associated with the primary display.")

            profile_path_buffer = ctypes.create_unicode_buffer(buffer_size.value)
            if not ctypes.windll.gdi32.GetICMProfileW(hdc, ctypes.byref(buffer_size), profile_path_buffer):
                 raise ValueError("Failed to retrieve monitor profile path.")

            win32gui.ReleaseDC(0, hdc)
            profile_path = profile_path_buffer.value
            
            if os.path.exists(profile_path):
                print(f"INFO: Found monitor ICC profile: {profile_path}")
                return ImageCms.ImageCmsProfile(profile_path)
            else:
                raise FileNotFoundError(f"Monitor profile path does not exist: {profile_path}")
        except Exception as e:
            print(f"WARNING: Could not get monitor ICC profile: {e}. Falling back to sRGB for screen display.")
            return self.srgb_profile

    def _get_printer_icc_profile(self, printer_name):
        """
        Retrieves the raw ICC profile data for the specified printer.
        Returns bytes, or None if not found.
        """
        if not WIN32_AVAILABLE or not printer_name: return None
        if printer_name in self.printer_icc_profile_cache:
            return self.printer_icc_profile_cache[printer_name]

        try:
            hdc = win32gui.CreateDC("WINSPOOL", printer_name, None)
            buffer_size = wintypes.DWORD(0)
            ctypes.windll.gdi32.GetICMProfileW(hdc, ctypes.byref(buffer_size), None)
            if buffer_size.value == 0: raise ValueError("No profile associated.")

            profile_path_buffer = ctypes.create_unicode_buffer(buffer_size.value)
            if not ctypes.windll.gdi32.GetICMProfileW(hdc, ctypes.byref(buffer_size), profile_path_buffer):
                 raise ValueError("Failed to retrieve profile path.")

            win32gui.DeleteDC(hdc)
            profile_path = profile_path_buffer.value
            
            if os.path.exists(profile_path):
                print(f"INFO: Found printer ICC profile: {profile_path}")
                with open(profile_path, 'rb') as f:
                    profile_data = f.read()
                    self.printer_icc_profile_cache[printer_name] = profile_data
                    return profile_data
            else:
                raise FileNotFoundError(f"Profile path does not exist: {profile_path}")
        except Exception as e:
            print(f"INFO: Could not get ICC profile for '{printer_name}': {e}. Using sRGB as fallback.")
            self.printer_icc_profile_cache[printer_name] = None
            return None

    def _get_printer_info(self, printer_name):
        if not WIN32_AVAILABLE: return None
        try:
            devmode = self._get_configured_devmode(printer_name)
            hdc_handle = win32gui.CreateDC("WINSPOOL", printer_name, devmode)
            hdc = win32ui.CreateDCFromHandle(hdc_handle)
            info = {
                "dpi_x": hdc.GetDeviceCaps(win32con.LOGPIXELSX), "dpi_y": hdc.GetDeviceCaps(win32con.LOGPIXELSY),
                "physical_width_px": hdc.GetDeviceCaps(win32con.PHYSICALWIDTH), "physical_height_px": hdc.GetDeviceCaps(win32con.PHYSICALHEIGHT),
                "printable_width_px": hdc.GetDeviceCaps(win32con.HORZRES), "printable_height_px": hdc.GetDeviceCaps(win32con.VERTRES)
            }
            hdc.DeleteDC()
            info["printable_width_mm"] = info["printable_width_px"] / info["dpi_x"] * 25.4
            info["printable_height_mm"] = info["printable_height_px"] / info["dpi_y"] * 25.4
            return info
        except Exception as e:
            print(f"ERROR: Failed to get printer info for '{printer_name}': {e}")
            return None

    def show_printer_properties(self):
        if not WIN32_AVAILABLE: return
        printer_name = self.printer_var.get()
        if not printer_name or printer_name == "No printers found": return
        
        try:
            hprinter = win32print.OpenPrinter(printer_name)
            devmode = self._get_configured_devmode(printer_name)
            result = win32print.DocumentProperties(self.root.winfo_id(), hprinter, printer_name, devmode, devmode, win32con.DM_IN_BUFFER | win32con.DM_IN_PROMPT | win32con.DM_OUT_BUFFER)
            win32print.ClosePrinter(hprinter)
            
            if result == win32con.IDOK:
                self.printer_settings = devmode
                self.status_var.set("Printer settings updated from dialog.")
                self.schedule_render(force_relayout=True)
        except Exception as e:
            print(f"ERROR: Could not show printer properties: {e}")
            messagebox.showerror("Printer Properties Error", f"Could not open printer properties dialog:\n{e}")

if __name__ == "__main__":
    try:
        if DND_ENABLED:
            root = TkinterDnD.Tk()
        else:
            root = tk.Tk()
        app = PhotoOrganizerApp(root)
        root.mainloop()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        messagebox.showerror("Startup Error", f"Failed to start the application:\n{e}")