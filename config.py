import cv2
from pathlib import Path
from datetime import datetime
import yaml
import os


class Config:
    """Configuration class for the object detection application."""

    # Model Settings
    WEIGHT_FILE = 'yolo12n.pt'
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45

    # Classes
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

    # Categories for object classification
    CATEGORIES = {
        'people': ['person'],
        'vehicles': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
        'traffic': ['traffic light', 'fire hydrant', 'stop sign', 'parking meter'],
        'animals': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
        'objects': ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase'],
        'sports': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                  'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
        'food': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake'],
        'furniture': ['chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet'],
        'electronics': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'],
        'appliances': ['microwave', 'oven', 'toaster', 'sink', 'refrigerator'],
        'misc': ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    }

    # ============================================================
    # TIME-BASED INFERENCE SETTINGS
    # ============================================================
    INFERENCE_INTERVAL = 5.0  # seconds between inference cycles
    OVERLAY_DURATION = 2.0    # seconds to display overlay
    AUDIO_MIN_GAP = 0.5       # minimum gap between audio messages
    AUDIO_MAX_MESSAGE_WORDS = 15
    
    # ============================================================
    # TTS (TEXT-TO-SPEECH) SETTINGS
    # ============================================================
    TTS_MAX_WORDS = 15                      # Max words for general messages
    TTS_DANGER_MAX_WORDS = 30               # Max words for danger messages
    TTS_NAVIGATION_PREFIX = "Navigation:"   # Prefix for navigation announcements
    TTS_COOLDOWN_SECONDS = 3.0              # Standard cooldown between messages
    TTS_NAVIGATION_COOLDOWN_SECONDS = 0.5   # Shorter cooldown for navigation
    TTS_CRITICAL_DANGER_THRESHOLD = 0.5     # Danger score threshold for critical priority
    TTS_SPEAK_NAVIGATION_SEPARATELY = True  # Speak navigation as separate message

    # Legacy settings (kept for compatibility, not used for triggering)
    YOLO_INTERVAL = 1
    GPT_INTERVAL = 1

    # Detection and Processing Settings
    MOTION_THRESHOLD = 20
    MAX_OBJECTS_PER_FRAME = 50
    YOLO_IMAGE_SIZE = (704, 1024)

    # Video Settings
    FPS = 30
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')

    # Display Settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2
    LINE_THICKNESS = 2

    # Colors (BGR format)
    COLORS = {
        'people': (0, 0, 255),
        'vehicles': (255, 0, 0),
        'traffic': (0, 255, 255),
        'animals': (128, 0, 128),
        'objects': (0, 128, 255),
        'sports': (255, 128, 0),
        'food': (0, 255, 0),
        'furniture': (128, 128, 0),
        'electronics': (255, 0, 255),
        'appliances': (128, 0, 0),
        'misc': (128, 128, 128),
        'default': (0, 255, 0),
        'text': (255, 255, 255),
        'background': (0, 0, 0),
    }
    
    # Modern UI Color Scheme (BGR format)
    UI_COLORS = {
        # Primary colors
        'primary': (255, 107, 40),          # Vibrant blue
        'primary_dark': (200, 80, 30),      # Darker blue
        'accent': (60, 180, 255),           # Orange accent
        
        # Status colors
        'success': (80, 200, 120),          # Green
        'warning': (0, 165, 255),           # Orange
        'danger': (60, 60, 255),            # Red
        'info': (255, 200, 100),            # Cyan
        
        # Neutral colors
        'text_primary': (255, 255, 255),    # White
        'text_secondary': (180, 180, 180),  # Light gray
        'text_muted': (120, 120, 120),      # Gray
        
        # Background colors
        'bg_dark': (30, 30, 30),            # Dark background
        'bg_semi': (20, 20, 20),            # Semi-transparent bg
        'bg_panel': (45, 45, 45),           # Panel background
        'bg_overlay': (0, 0, 0),            # Full overlay bg
        
        # Semantic colors
        'on': (80, 200, 120),               # Green for ON
        'off': (100, 100, 180),             # Muted red for OFF
        'processing': (0, 200, 255),        # Yellow for processing
        'critical': (0, 0, 255),            # Bright red
    }
    
    # UI Spacing and Layout
    UI_LAYOUT = {
        'padding': 15,
        'margin': 10,
        'corner_radius': 8,
        'border_width': 2,
        'line_spacing': 25,
        'section_spacing': 35,
    }
    
    # UI Typography
    UI_FONTS = {
        'title_scale': 0.8,
        'title_thickness': 2,
        'heading_scale': 0.65,
        'heading_thickness': 2,
        'body_scale': 0.5,
        'body_thickness': 1,
        'small_scale': 0.45,
        'small_thickness': 1,
    }

    # GPT Settings
    SYSTEM_SENSITIVITY = 'normal'
    GPT_MODEL = "gpt-4o"
    GPT_MAX_TOKENS = 150
    GPT_TEMPERATURE = 0.7

    # Device Settings (GPU/CPU)
    USE_GPU = True
    DEVICE = None

    def __init__(self):
        """Initialize configuration and create necessary directories."""
        self.ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
        self.OUTPUT_DIR = self.ROOT_DIR / 'output'
        self.TEMP_DIR = self.ROOT_DIR / 'temp'
        self.LOG_DIR = self.ROOT_DIR / 'logs'
        self.WEIGHT_DIR = self.ROOT_DIR / 'weights'

        self.create_directories()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_video_path = self.OUTPUT_DIR / f"output_video_{timestamp}.mp4"
        self.log_file = self.LOG_DIR / f"app_log_{timestamp}.txt"

        self.load_api_keys()
        self._detect_device()
        self.validate_config()

    def create_directories(self):
        for directory in [self.OUTPUT_DIR, self.TEMP_DIR, self.LOG_DIR, self.WEIGHT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_api_keys(self):
        try:
            api_keys_path = self.ROOT_DIR / "OPENAI_API_KEY.yaml"
            if api_keys_path.exists():
                with open(api_keys_path) as f:
                    self.api_keys = yaml.safe_load(f)
                print("API keys loaded successfully")
            else:
                print("Warning: API keys file not found")
                self.api_keys = {}
        except Exception as e:
            print(f"Error loading API keys: {e}")
            self.api_keys = {}

    def _detect_device(self):
        try:
            import torch
            if self.USE_GPU and torch.cuda.is_available():
                self.DEVICE = 'cuda'
                device_name = torch.cuda.get_device_name(0)
                print(f"GPU detected: {device_name}")
                print(f"Using CUDA device: {self.DEVICE}")
            else:
                self.DEVICE = 'cpu'
                if self.USE_GPU:
                    print("Warning: GPU requested but CUDA not available. Using CPU.")
                else:
                    print("Using CPU device (GPU disabled in config)")
        except ImportError:
            self.DEVICE = 'cpu'
            print("Warning: PyTorch not available. Using CPU device.")

    def validate_config(self):
        for category, class_list in self.CATEGORIES.items():
            for class_name in class_list:
                if class_name not in self.CLASSES:
                    print(f"Warning: Category {category} contains invalid class: {class_name}")

        weight_path = self.WEIGHT_DIR / self.WEIGHT_FILE
        if not weight_path.exists():
            print(f"Warning: Weight file not found at {weight_path}")

    def get_category(self, class_name: str) -> str:
        for category, classes in self.CATEGORIES.items():
            if class_name in classes:
                return category
        return 'default'

    def get_color(self, class_name: str) -> tuple:
        category = self.get_category(class_name)
        return self.COLORS.get(category, self.COLORS['default'])

    @property
    def openai_api_key(self) -> str:
        return self.api_keys.get('OPENAI_API_KEY', '')

    @property
    def weight_path(self) -> Path:
        return self.WEIGHT_DIR / self.WEIGHT_FILE


config = Config()
system_sensitivity = config.SYSTEM_SENSITIVITY



