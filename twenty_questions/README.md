# 20 Questions Face Guessing Game

An interactive visual preference game that uses a binary FSQ-based FlexTok model to guess the face you're thinking of through 20 yes/no questions.

**Available Interfaces:**
- 🌐 **Web UI** (Gradio) - Browser-based, deployable, recommended ✨
- 🖥️ **Desktop UI** (Tkinter) - Native desktop application

## Overview

This game plays "20 questions" with visual preferences:
1. Think of a face in your mind
2. The system shows you 2 options (Option A and Option B), each displaying multiple sample images
3. Choose which option better matches your imagined face
4. Repeat for up to 20 questions
5. See the final generated face that matches your choices!

The game uses FlexTok's hierarchical tokenization where:
- Early questions capture high-level features (overall appearance, lighting, etc.)
- Later questions refine fine details
- Each choice narrows down from 2^256 possible face clusters

## Installation

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision pillow
pip install diffusers transformers
pip install flextok  # Or install from the parent directory

# For Web UI (Gradio)
pip install gradio

# For Desktop UI (Tkinter) - usually pre-installed with Python
# On Linux, you may need: sudo apt-get install python3-tk
```

### Required Model

The UI uses the FlexTok model from HuggingFace:
- Model: `EPFL-VILAB/flextok_d18_d18_in1k`
- This will be downloaded automatically on first run

## Usage

### Option 1: Web UI (Recommended) 🌐

The web interface runs in your browser and can be accessed from any device on your network.

```bash
cd /home/iyu/ml-flextok/twenty_questions
python twenty_questions_web.py
```

Then open your browser to:
- **Local access**: http://localhost:7860
- **Network access**: http://YOUR_IP:7860 (e.g., http://192.168.1.100:7860)

**Public Sharing:**
To create a public shareable link, set `share=True` in the code:
```python
demo.launch(share=True)  # Creates a temporary public URL
```

### Option 2: Desktop UI (Tkinter) 🖥️

Native desktop application with traditional window interface.

```bash
cd /home/iyu/ml-flextok/twenty_questions
python twenty_questions_ui.py
```

### GPU Configuration

By default, both UIs use GPU if available. To specify a specific GPU:

```bash
# Set before running either UI
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
python twenty_questions_web.py  # or twenty_questions_ui.py
```

## UI Components

### Web UI (Gradio)
- **Status Bar**: Shows current question number and game status
- **Choice History**: Gallery view of all your previous choices
- **Option A & B**: Side-by-side galleries showing sample images (2x2 grid)
- **Choice Buttons**: Large buttons to select Option A or B
- **Start New Game**: Initialize or restart the game
- **Final Result**: Large display of your generated face after completion

### Desktop UI (Tkinter)
- **Top Panel**:
  - Progress Bar: Shows questions answered (X/20)
  - Thumbnail History: Scrollable strip of your choices
- **Center Panel**:
  - Option A (Left): Grid of sample images
  - Option B (Right): Grid of sample images
  - Each shows 4 images in a 2x2 grid
- **Bottom Panel**:
  - Choose Option A/B buttons
  - Restart Game button
- **Final Result Screen**: Popup with save functionality

## Configuration

### Web UI (Gradio)
Modify parameters in the `TwentyQuestionsGame.__init__()` method:

```python
self.max_questions = 20  # Number of questions (default: 20)
self.num_samples_per_quantization = 4  # Images per option (default: 4)
```

### Desktop UI (Tkinter)
Modify parameters in the `TwentyQuestionsUI.__init__()` method:

```python
self.max_questions = 20  # Number of questions (default: 20)
self.num_samples_per_quantization = 4  # Images per option (default: 4)
```

## Game Flow

1. **Initialization**: Model loads (shown with loading screen)
2. **Question Loop** (20 iterations):
   - System generates 2 options based on your previous choices
   - You select which option matches your imagined face better
   - Your choice is added to the thumbnail history
   - Progress bar updates
3. **Final Result**:
   - View the final generated face
   - Save the image or play again

## Technical Details

### Architecture
- **Web UI Framework**: Gradio (web-based interface)
- **Desktop UI Framework**: Tkinter (Python standard library)
- **Image Handling**: PIL/Pillow
- **Model**: FlexTok with FSQ (Finite Scalar Quantization)
- **Async Operations**: Threading (Tkinter) / Gradio's async handling (Web)

### Performance Notes
- First run downloads the model (~several GB)
- Each question generation takes time (especially with GPU inference)
- Loading indicators show progress during generation
- Image generation is non-deterministic (controlled randomness in denoising)

### File Structure
```
twenty_questions/
├── twenty_questions.py       # Original CLI script with core logic
├── twenty_questions_web.py   # Web UI implementation (Gradio)
├── twenty_questions_ui.py    # Desktop UI implementation (Tkinter)
└── README.md                 # This file
```

## Troubleshooting

### Model Loading Issues
```
Error: Failed to initialize model
```
- Ensure you have internet connection for first-time model download
- Check CUDA availability: `torch.cuda.is_available()`
- Verify sufficient GPU memory (recommended: 8GB+ VRAM)

### Memory Issues
```
CUDA out of memory
```
- Reduce `num_samples_per_quantization` (e.g., from 4 to 2)
- Use CPU mode by setting `device = 'cpu'` (slower but uses RAM instead)

### Display Issues
- If images don't display properly, ensure PIL/Pillow is installed correctly
- On some systems, you may need to install `python3-tk` separately

## Deployment

### Local Network Deployment (Web UI)

The web UI can be accessed from any device on your local network:

```bash
python twenty_questions_web.py
# Access from any device: http://YOUR_SERVER_IP:7860
```

### Public Deployment Options

#### Option 1: Gradio Share Link (Temporary)
```python
# In twenty_questions_web.py, modify the launch call:
demo.launch(share=True)  # Creates a temporary public URL (72 hours)
```

#### Option 2: HuggingFace Spaces (Permanent)
Deploy to HuggingFace Spaces for free hosting:

1. Create a new Space at https://huggingface.co/spaces
2. Upload your files:
   - `twenty_questions_web.py`
   - `twenty_questions.py`
   - `requirements.txt` (create with dependencies)
3. The Space will automatically run and provide a public URL

**requirements.txt example:**
```
torch
torchvision
pillow
diffusers
transformers
gradio
flextok
```

#### Option 3: Self-Hosted Server
Deploy on a VPS or cloud instance:

```bash
# Install dependencies
pip install -r requirements.txt

# Run with public access
python twenty_questions_web.py

# For production, use a process manager like PM2 or systemd
# Or run with nohup for background execution:
nohup python twenty_questions_web.py > output.log 2>&1 &
```

### Docker Deployment (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY twenty_questions.py .
COPY twenty_questions_web.py .

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "twenty_questions_web.py"]
```

Build and run:
```bash
docker build -t twenty-questions .
docker run -p 7860:7860 --gpus all twenty-questions
```

## Advanced Usage

### Using Custom Checkpoints

Modify the initialization in `initialize_model()`:

```python
self.flextok_model = load_flextok_model(
    model_name='EPFL-VILAB/flextok_d18_d18_in1k',
    bf16=self.enable_bf16,
    ckpt_path='/path/to/your/checkpoint.pth',  # Add your checkpoint
    fsq_level=[2, 2, 2, 2, 2]  # Optional: override FSQ levels
)
```

### Customizing Image Grid Layout

To change the grid layout, modify `display_question_images()`:

```python
# Current: 2x2 grid (4 images)
cols = 2

# Change to 3x3 grid (9 images):
cols = 3
self.num_samples_per_quantization = 9
```

## License

This project uses the FlexTok model and follows its licensing terms.
