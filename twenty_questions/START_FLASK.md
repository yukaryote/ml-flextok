# Quick Start Guide - Flask Version

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
cd /home/iyu/ml-flextok/twenty_questions
pip install -r requirements_flask.txt
```

### 2. Start the Flask Backend

```bash
python app.py
```

You should see output like:
```
Initializing model...
Device: cuda
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
 * Running on http://YOUR_IP:5000
```

**Important:** Keep this terminal window open!

### 3. Open in Browser

Open your web browser and navigate to:
```
http://localhost:5000
```

### 4. Start Playing

Click the "Start Game" button and wait for the model to initialize (first time only).

## Troubleshooting

### Error: "Cannot connect to server"

**Cause:** Flask backend is not running

**Solution:**
1. Check if `app.py` is running in a terminal
2. Look for the message "Running on http://127.0.0.1:5000"
3. Make sure no other application is using port 5000

### Error: "Failed to initialize model"

**Cause:** Model checkpoint not found or GPU issues

**Solution:**
1. Check the checkpoint path in `app.py` line 61:
   ```python
   ckpt_path="/home/iyu/ml-flextok/checkpoints/celeba_d18_fsq_4/20251202/checkpoint_best.pt"
   ```
2. Verify the file exists: `ls -la /home/iyu/ml-flextok/checkpoints/celeba_d18_fsq_4/20251202/checkpoint_best.pt`
3. Check GPU availability: `nvidia-smi`

### Port Already in Use

```bash
# Find what's using port 5000
lsof -i :5000

# Kill the process (replace PID with actual process ID)
kill -9 PID

# Or use a different port in app.py:
# app.run(host='0.0.0.0', port=5001, debug=True)
```

### CORS Errors in Browser

Make sure `flask-cors` is installed:
```bash
pip install flask-cors
```

### Images Not Generating

Check the terminal running Flask for error messages. Common issues:
- Out of memory (reduce batch size)
- CUDA errors (check GPU availability)
- Model loading failures (check checkpoint path)

## Checking Backend Status

Open a new terminal and run:
```bash
curl http://localhost:5000/api/status
```

Should return:
```json
{
  "current_question": 0,
  "device": "cuda",
  "initialized": false,
  "max_questions": 20
}
```

## Development Mode

For auto-reload on code changes, the backend is already in debug mode.
Just save your changes and Flask will restart automatically.

## Production Mode

For production deployment, use gunicorn:
```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

**Note:** Use only 1 worker (`-w 1`) to maintain game state!
