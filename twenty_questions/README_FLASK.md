# 20 Questions - Flask Version

A minimalistic web application for the "20 Questions" face guessing game with a clean frontend-backend architecture.

## Architecture

- **Backend**: Flask REST API (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Communication**: RESTful JSON API

## Features

✨ **Minimalistic Design**
- Clean, modern interface
- No frameworks - pure HTML/CSS/JS
- Responsive layout

🎨 **User Experience**
- Real-time progress tracking
- Smooth transitions
- Image history (chosen & rejected)
- Loading states

🔧 **Technical**
- Separation of concerns (frontend/backend)
- RESTful API design
- CORS enabled for development
- Base64 image encoding

## Installation

### Prerequisites

```bash
pip install -r requirements_flask.txt
```

### Required Files

```
twenty_questions/
├── app.py                      # Flask backend
├── twenty_questions.py         # Core game logic
├── requirements_flask.txt      # Python dependencies
└── static/
    ├── index.html              # Frontend HTML
    ├── style.css               # Minimalistic styles
    └── script.js               # Frontend logic
```

## Usage

### Running the Application

1. **Start the Flask backend:**

```bash
cd /home/iyu/ml-flextok/twenty_questions
python app.py
```

The server will start on `http://127.0.0.1:5000`

2. **Open in browser:**

Navigate to: `http://127.0.0.1:5000`

**Note:** Use `127.0.0.1` instead of `localhost` if localhost doesn't resolve on your system.

### API Endpoints

#### `POST /api/init`
Initialize/restart the game
- **Response**: First question with options A and B

#### `POST /api/choose`
Submit a choice
- **Body**: `{ "choice": "a" }` or `{ "choice": "b" }`
- **Response**: Next question or final result

#### `GET /api/status`
Get current game status
- **Response**: Game state information

## UI Overview

### Layout

```
┌─────────────────────────────────┐
│         20 Questions            │
│   Think of a face. Choose...    │
├─────────────────────────────────┤
│      Progress: 5/20 ████░░      │
├─────────────────────────────────┤
│    Which option matches better? │
├─────────────────────────────────┤
│   Option A    │    Option B     │
│   [Image]     │    [Image]      │
│   [Choose A]  │   [Choose B]    │
├─────────────────────────────────┤
│ ✓ Chosen:  [img][img][img]...   │
│ ✗ Rejected:[img][img][img]...   │
└─────────────────────────────────┘
```

### Design Principles

- **Minimalism**: No unnecessary elements
- **Clarity**: Clear visual hierarchy
- **Responsiveness**: Works on all screen sizes
- **Performance**: Lightweight, fast loading

## Configuration

### Backend Configuration

Edit in `app.py`:

```python
class GameState:
    def __init__(self):
        self.max_questions = 20              # Number of questions
        self.num_samples_per_quantization = 1 # Images per option
```

### Frontend Configuration

Edit in `script.js`:

```javascript
const API_URL = 'http://localhost:5000/api';  // Backend URL
```

## Deployment

### Local Network

```bash
# Backend is already configured for network access (0.0.0.0)
python app.py

# Access from other devices:
# http://YOUR_IP:5000
```

### Production Deployment

#### Option 1: Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

**Note**: Use only 1 worker (`-w 1`) to maintain game state consistency.

#### Option 2: Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_flask.txt .
RUN pip install --no-cache-dir -r requirements_flask.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t twenty-questions-flask .
docker run -p 5000:5000 --gpus all twenty-questions-flask
```

#### Option 3: Cloud Platforms

**Render.com:**
- Connect GitHub repo
- Select "Web Service"
- Build: `pip install -r requirements_flask.txt`
- Start: `gunicorn -w 1 -b 0.0.0.0:$PORT app:app`

**Heroku:**
```bash
# Create Procfile
echo "web: gunicorn -w 1 app:app" > Procfile

# Deploy
heroku create
git push heroku main
```

**Railway.app:**
- Import from GitHub
- Detects Flask automatically
- Add GPU if needed

## Development

### File Structure

```
Backend (app.py):
├── GameState class         # Game logic & state
├── API routes             # REST endpoints
└── Image encoding         # Base64 conversion

Frontend (static/):
├── index.html             # Structure
├── style.css              # Minimalistic design
└── script.js              # API communication
```

### Adding Features

**Backend:**
```python
@app.route('/api/your-endpoint', methods=['POST'])
def your_function():
    # Your logic here
    return jsonify({"data": "value"})
```

**Frontend:**
```javascript
async function yourFunction() {
    const response = await fetch(`${API_URL}/your-endpoint`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: 'value' })
    });
    const result = await response.json();
}
```

## Troubleshooting

### CORS Issues

If you see CORS errors in the browser console:
- Ensure `flask-cors` is installed
- Check that `CORS(app)` is called in `app.py`

### Images Not Loading

- Check browser console for errors
- Verify backend is running on port 5000
- Ensure API_URL in `script.js` is correct

### Model Not Loading

- Check GPU availability
- Verify checkpoint path in `app.py`
- Check console output for error messages

### State Issues

- Using multiple workers breaks state (use `-w 1`)
- Refresh page to reset client state
- Use `/api/status` to check server state

## Performance

- **Initial load**: Model initialization (~5-10 seconds)
- **Per question**: Image generation (~2-5 seconds on GPU)
- **Frontend**: Instant rendering with base64 images

## Comparison with Gradio Version

| Feature | Flask | Gradio |
|---------|-------|--------|
| Customization | High | Medium |
| Setup complexity | Medium | Low |
| UI Control | Full | Limited |
| Deployment | Flexible | HF Spaces optimized |
| Learning curve | Higher | Lower |

## License

This project uses the FlexTok model and follows its licensing terms.
