# BlindAssist v1.0

**BlindAssist** is an AI-powered navigation assistant designed to help visually impaired users navigate their environment safely. The system uses real-time computer vision, object detection, and GPT-4 to provide audio guidance and spatial awareness.

## 🎯 Features

- **Real-time Object Detection**: YOLO12-based detection with tracking
- **AI-Powered Analysis**: GPT-4o integration for intelligent scene understanding
- **Spatial Understanding**: Clock-direction based spatial descriptions
- **Audio Feedback**: Text-to-speech navigation guidance
- **ArUco Marker Detection**: Support for 3-digit marker IDs (000-999)
- **Multiple Input Sources**: Camera, video files, or phone camera streams
- **Real-time Overlays**: Visual feedback with danger assessment

## 📋 Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)
- Camera or video input source
- OpenAI API key (for GPT-4o features)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ayanzadeh93/BlindAssist.git
cd BlindAssist
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create `OPENAI_API_KEY.yaml` in the project root:

```yaml
OPENAI_API_KEY: "your-api-key-here"
```

Or create `API_KEYS.yaml`:

```yaml
OPENAI_API_KEY: "your-openai-key"
ANTHROPIC_API_KEY: "your-anthropic-key"  # Optional
```

### 5. Download Model Weights

Place YOLO model weights in the `weights/` directory:
- `yolo12n.pt` (recommended) or `yolo12n.onnx`

The model will auto-download if not found locally.

## 🎮 Usage

### Basic Usage

```bash
python main.py
```

### Command Line Options

```bash
# Use camera device 0
python main.py --input 0

# Process video file
python main.py --video-file path/to/video.mp4

# Use phone camera (IP:port)
python main.py --phone-camera 192.168.1.100:8080

# Background mode (no visualization)
python main.py --background-mode

# Custom inference interval
python main.py --inference-interval 3.0
```

### Keyboard Controls

- **Q** - Quit application
- **P** - Pause/Resume
- **D** - Toggle debug overlay
- **M** - Toggle depth map
- **B** - Toggle detection boxes
- **G** - Toggle GPT overlay
- **A** - Toggle audio
- **T** - Test audio
- **K** - Spatial understanding mode
- **J** - Deep spatial mode (depth + detections)
- **R** - Toggle ArUco marker detection
- **H** - Toggle help
- **U** - Toggle UI panel

## 📁 Project Structure

```
BlindAssist/
├── main.py                      # Main application entry point
├── config.py                    # Configuration management
├── detection_engine.py          # YOLO object detection
├── gpt_interface.py             # GPT-4o API integration
├── audio_system.py              # Text-to-speech system
├── application_state.py         # Application state management
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Configuration

Edit `config.py` or use command-line arguments to customize:

- **Inference Interval**: Time between AI analysis cycles (default: 5.0s)
- **Overlay Duration**: How long overlays are displayed (default: 2.0s)
- **Confidence Threshold**: Object detection confidence (default: 0.25)
- **GPU/CPU**: Device selection for processing

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on GitHub.

## 🙏 Acknowledgments

- YOLO by Ultralytics
- OpenAI GPT-4o
- OpenCV community

---

**Version**: 1.0.0  
**Repository**: https://github.com/Ayanzadeh93/BlindAssist

