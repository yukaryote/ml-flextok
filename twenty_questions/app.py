"""
Flask backend API for the 20 Questions Face Guessing Game.
Provides REST API endpoints for the game logic.
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, List

from flextok.flextok_wrapper import FlexTok
from flextok.utils.demo import denormalize
from flextok.utils.misc import detect_bf16_support

from twenty_questions import (
    load_flextok_model,
    get_possible_combos,
    zhat_to_tokens,
    sample_images_per_quantization,
    convert_images_to_pil
)

# Setup PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

# Increase torch dynamo recompile limit to handle variable sequence lengths
import torch._dynamo
torch._dynamo.config.cache_size_limit = 20  # Increase from default 8 to 13

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for frontend-backend communication


class GameState:
    """Singleton game state manager."""

    def __init__(self):
        self.flextok_model: Optional[FlexTok] = None
        self.tokens_list: List[torch.Tensor] = []
        self.chosen_tokens: List[torch.Tensor] = []
        self.chosen_images: List[Image.Image] = []
        self.rejected_images: List[Image.Image] = []
        self.current_question = 0
        self.max_questions = 20
        self.num_samples_per_quantization = 1
        self.num_options = 8  # Number of options per question (n-ary choice)
        self.enable_bf16 = detect_bf16_support()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_images_dict: Dict[int, torch.Tensor] = {}
        self.current_tokens: List[int] = []
        self.initialized = False

    def initialize_model(self):
        """Initialize the FlexTok model."""
        if self.initialized:
            return {"status": "already_initialized"}
        fsq_level = self.num_options
        print("Initializing model...")
        self.flextok_model = load_flextok_model(
            model_name='EPFL-VILAB/flextok_d18_d18_in1k',
            bf16=self.enable_bf16,
            ckpt_path="/home/iyu/ml-flextok/checkpoints/synth_d18_fsq_8/20251215/checkpoint_best.pt",
            fsq_level=[fsq_level]
        )

        # Get all possible tokens
        all_zhats = get_possible_combos(self.flextok_model).to(self.device)
        self.tokens_list = zhat_to_tokens(self.flextok_model, all_zhats).unsqueeze(-1)

        # DEBUG: change last one to [[1]] instead of [[2]]
        self.tokens_list[-1] = torch.tensor([[fsq_level - 1]], device=self.tokens_list[-1].device)
        self.tokens_list = list(self.tokens_list.split(1))
        print(f"Possible tokens prepared. Count: {len(self.tokens_list)}")
        self.initialized = True
        print(f"Initialized with {len(self.tokens_list)} possible tokens")
        return {"status": "success", "num_tokens": len(self.tokens_list)}

    def reset_game(self):
        """Reset game state for a new game."""
        self.chosen_tokens = []
        self.chosen_images = []
        self.rejected_images = []
        self.current_question = 0
        self.current_images_dict = {}
        self.current_tokens = []

    def generate_question(self):
        """Generate images for the current question."""
        if not self.initialized:
            raise ValueError("Model not initialized")

        self.current_question += 1

        # Sample images from tokens (limit to num_options)
        tokens_to_sample = self.tokens_list[:self.num_options]

        images_dict = sample_images_per_quantization(
            self.flextok_model,
            tokens_to_sample,
            num_samples_per_quantization=self.num_samples_per_quantization,
            condition_tokens=self.chosen_tokens,
            bf16=self.enable_bf16
        )

        self.current_images_dict = images_dict
        self.current_tokens = list(images_dict.keys())

        # Convert to PIL images for all options
        option_images = []
        for i in range(len(self.current_tokens)):
            img = convert_images_to_pil(images_dict[self.current_tokens[i]][0])[0]
            option_images.append(img)

        return option_images

    def make_choice(self, choice_idx: int):
        """Process user's choice."""

        # Save the chosen token and image
        chosen_token = self.current_tokens[choice_idx]
        chosen_image_tensor = self.current_images_dict[chosen_token]
        chosen_image_pil = convert_images_to_pil(chosen_image_tensor[0])[0]

        self.chosen_tokens.append(
            torch.tensor([[chosen_token]], device=self.tokens_list[0].device)
        )
        self.chosen_images.append(chosen_image_pil)

        # Save all rejected images (all options except the chosen one)
        rejected_images = []
        for i, token in enumerate(self.current_tokens):
            if i != choice_idx:
                rejected_image_tensor = self.current_images_dict[token]
                rejected_image_pil = convert_images_to_pil(rejected_image_tensor[0])[0]
                rejected_images.append(rejected_image_pil)
                self.rejected_images.append(rejected_image_pil)

        return chosen_image_pil, rejected_images


# Global game state
game_state = GameState()


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# API Routes

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/init', methods=['POST'])
def init_game():
    """Initialize the model and start a new game."""
    try:
        # Initialize model if needed
        if not game_state.initialized:
            result = game_state.initialize_model()
            if result["status"] != "success":
                return jsonify({"error": "Failed to initialize model"}), 500

        # Reset game state
        game_state.reset_game()

        # Generate first question
        option_images = game_state.generate_question()

        # Create response with all options
        response = {
            "status": "success",
            "question": 1,
            "max_questions": game_state.max_questions,
            "num_options": game_state.num_options,
            "options": [image_to_base64(img) for img in option_images]
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in init_game: {e}")
        print(error_details)
        return jsonify({"error": str(e), "details": error_details}), 500


@app.route('/api/choose', methods=['POST'])
def make_choice():
    """Process user's choice and get next question."""
    try:
        data = request.json
        choice_idx = data.get('choice')  # Integer index (0, 1, 2, ...)

        if not isinstance(choice_idx, int) or choice_idx < 0 or choice_idx >= game_state.num_options:
            return jsonify({"error": f"Invalid choice. Must be 0-{game_state.num_options-1}"}), 400

        # Record the choice
        chosen_img, rejected_imgs = game_state.make_choice(choice_idx)

        # Check if game is complete
        if game_state.current_question >= game_state.max_questions:
            return jsonify({
                "status": "complete",
                "question": game_state.current_question,
                "chosen": image_to_base64(chosen_img),
                "rejected": [image_to_base64(img) for img in rejected_imgs],
                "final_image": image_to_base64(game_state.chosen_images[-1]),
                "chosen_history": [image_to_base64(img) for img in game_state.chosen_images],
                "rejected_history": [image_to_base64(img) for img in game_state.rejected_images]
            })

        # Generate next question
        option_images = game_state.generate_question()

        return jsonify({
            "status": "continue",
            "question": game_state.current_question,
            "max_questions": game_state.max_questions,
            "num_options": game_state.num_options,
            "chosen": image_to_base64(chosen_img),
            "rejected": [image_to_base64(img) for img in rejected_imgs],
            "options": [image_to_base64(img) for img in option_images]
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in make_choice: {e}")
        print(error_details)
        return jsonify({"error": str(e), "details": error_details}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current game status."""
    return jsonify({
        "initialized": game_state.initialized,
        "current_question": game_state.current_question,
        "max_questions": game_state.max_questions,
        "num_options": game_state.num_options,
        "device": game_state.device
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
