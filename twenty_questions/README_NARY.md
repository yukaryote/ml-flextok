# N-ary Options Support

The Flask version of the 20 Questions game now supports **n-ary choices** instead of just binary (A/B) options.

## Configuration

You can configure the number of options per question by modifying the `GameState` class in `app.py`:

```python
class GameState:
    def __init__(self):
        # ... other settings ...
        self.num_options = 4  # Number of options per question (2, 3, 4, etc.)
```

### Supported Values

- **2** - Binary choice (A vs B) - original behavior
- **3** - Ternary choice (1, 2, 3)
- **4** - Quaternary choice (1, 2, 3, 4) - **default**
- **5+** - Works with any FSQ level

**Important**: `num_options` must match your FSQ level in the model checkpoint.

## How It Works

### Backend Changes

1. **Model Initialization**: FSQ level is set to `num_options`
   ```python
   fsq_level = self.num_options
   ```

2. **Question Generation**: Generates N images instead of 2
   ```python
   tokens_to_sample = self.tokens_list[:self.num_options]
   ```

3. **Choice Processing**: Records 1 chosen image and N-1 rejected images
   ```python
   chosen_img, rejected_imgs = game_state.make_choice(choice_idx)
   ```

### Frontend Changes

1. **Dynamic Option Display**: Options are generated dynamically based on server response
   ```javascript
   displayOptions(data.options);  // Works with any number of options
   ```

2. **Choice Submission**: Sends integer index (0, 1, 2, ...) instead of 'a'/'b'
   ```javascript
   { choice: choiceIndex }  // 0-based index
   ```

3. **History**: Shows 1 chosen image per question, but N-1 rejected images

## API Changes

### `/api/init` Response

```json
{
    "status": "success",
    "question": 1,
    "max_questions": 20,
    "num_options": 4,
    "options": [
        "data:image/png;base64,...",  // Option 0
        "data:image/png;base64,...",  // Option 1
        "data:image/png;base64,...",  // Option 2
        "data:image/png;base64,..."   // Option 3
    ]
}
```

### `/api/choose` Request

```json
{
    "choice": 2  // Integer 0 to (num_options-1)
}
```

### `/api/choose` Response

```json
{
    "status": "continue",
    "question": 2,
    "num_options": 4,
    "chosen": "data:image/png;base64,...",
    "rejected": [
        "data:image/png;base64,...",  // Rejected option 0
        "data:image/png;base64,...",  // Rejected option 1
        "data:image/png;base64,..."   // Rejected option 3 (skipped option 2)
    ],
    "options": [...]  // Next question options
}
```

## UI Layout

The frontend automatically adapts to any number of options:

```
┌────────────────────────────────────────┐
│  [Option 1]  [Option 2]  [Option 3]  [Option 4]  │
│    [img]       [img]       [img]       [img]    │
│  [Choose 1] [Choose 2] [Choose 3] [Choose 4]   │
└────────────────────────────────────────┘
```

- **2 options**: 2-column grid
- **3-4 options**: Responsive grid (2x2 or 1x4 depending on screen size)
- **5+ options**: Wraps automatically

## Information Theory

With n-ary choices:
- **Binary (n=2)**: Each question provides 1 bit of information
- **Ternary (n=3)**: Each question provides ~1.58 bits
- **Quaternary (n=4)**: Each question provides 2 bits
- **n-ary**: Each question provides log₂(n) bits

This means with quaternary choices and 20 questions, you can distinguish between:
**2^40 = 1,099,511,627,776 possible faces** (vs 2^20 = 1,048,576 with binary)

## Example Configurations

### High Information (Quaternary)
```python
self.num_options = 4
self.max_questions = 20  # Can distinguish 2^40 faces
```

### Balanced (Ternary)
```python
self.num_options = 3
self.max_questions = 20  # Can distinguish 3^20 ≈ 2^31.7 faces
```

### Simple (Binary)
```python
self.num_options = 2
self.max_questions = 20  # Can distinguish 2^20 faces
```

## Backward Compatibility

The system remains backward compatible:
- Setting `num_options = 2` gives you the original binary choice behavior
- The frontend automatically adapts to any `num_options` value from the backend
- No changes needed to the frontend code when changing `num_options`
