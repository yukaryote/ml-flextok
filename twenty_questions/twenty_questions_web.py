"""
Web-based UI for the "20 Questions" face guessing game using Gradio.
This provides an interactive web interface that can be accessed via browser.
"""
import gradio as gr
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from typing import Optional, Dict, List, Tuple
import numpy as np

from flextok.flextok_wrapper import FlexTokFromHub, FlexTok
from flextok.utils.demo import denormalize
from flextok.utils.misc import detect_bf16_support, get_bf16_context

# Import functions from the original script
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


class TwentyQuestionsGame:
    """Game state manager for the 20 Questions game."""

    def __init__(self):
        self.flextok_model: Optional[FlexTok] = None
        self.tokens_list: List[torch.Tensor] = []
        self.chosen_tokens: List[torch.Tensor] = []
        self.chosen_images: List[Image.Image] = []
        self.rejected_images: List[Image.Image] = []
        self.current_question = 0
        self.max_questions = 20
        self.num_samples_per_quantization = 1
        self.enable_bf16 = detect_bf16_support()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_images_dict: Dict[int, torch.Tensor] = {}
        self.current_tokens: List[int] = []
        self.game_initialized = False

    def initialize_model(self):
        """Initialize the FlexTok model."""
        if self.game_initialized:
            return "Model already initialized!"

        print("Initializing model...")
        self.flextok_model = load_flextok_model(
            model_name='EPFL-VILAB/flextok_d18_d18_in1k',
            bf16=self.enable_bf16,
            ckpt_path="/home/iyu/ml-flextok/checkpoints/celebahq_d18_fsq_2/20251119/checkpoint_latest.pt",
            fsq_level=[2]
        )

        # Get all possible tokens
        all_zhats = get_possible_combos(self.flextok_model).to(self.device)
        self.tokens_list = zhat_to_tokens(self.flextok_model, all_zhats).unsqueeze(-1)

        # DEBUG: change last one to [[1]] instead of [[2]]
        #self.tokens_list[-1] = torch.tensor([[1]], device=self.tokens_list[-1].device)
        self.tokens_list = list(self.tokens_list.split(1))
        print("Possible tokens prepared.", self.tokens_list)
        self.game_initialized = True
        print(f"Initialized with {len(self.tokens_list)} possible tokens")
        return "Model initialized successfully!"

    def reset_game(self):
        """Reset the game state."""
        self.chosen_tokens = []
        self.chosen_images = []
        self.rejected_images = []
        self.current_question = 0
        self.current_images_dict = {}
        self.current_tokens = []

    def generate_question(self) -> Tuple[List[Image.Image], List[Image.Image], str]:
        """Generate images for the next question."""
        if not self.game_initialized:
            raise ValueError("Model not initialized! Click 'Start New Game' first.")

        if self.current_question >= self.max_questions:
            return [], [], "Game complete! See your final result below."

        self.current_question += 1

        # Sample images from tokens
        print(f"Generating question {self.current_question}/{self.max_questions}...")
        images_dict = sample_images_per_quantization(
            self.flextok_model,
            self.tokens_list,
            num_samples_per_quantization=self.num_samples_per_quantization,
            condition_tokens=self.chosen_tokens,
            bf16=self.enable_bf16
        )

        self.current_images_dict = images_dict
        self.current_tokens = list(images_dict.keys())

        # Convert to PIL images - return single image (not list) for gr.Image
        option_a_image = convert_images_to_pil(images_dict[self.current_tokens[0]][0])[0]
        option_b_image = convert_images_to_pil(images_dict[self.current_tokens[1]][0])[0]

        status = f"Question {self.current_question}/{self.max_questions}: Which option better matches your imagined face?"

        return option_a_image, option_b_image, status

    def make_choice(self, choice_idx: int) -> Tuple[Optional[Image.Image], str, List[Image.Image], List[Image.Image]]:
        """Process user's choice and prepare next question or final result."""
        if choice_idx >= len(self.current_tokens):
            return None, "Invalid choice!", [], []

        # Save the choice and rejection
        chosen_token = self.current_tokens[choice_idx]
        rejected_token = self.current_tokens[1 - choice_idx]  # The other option

        chosen_image_tensor = self.current_images_dict[chosen_token]
        rejected_image_tensor = self.current_images_dict[rejected_token]

        chosen_image_pil = convert_images_to_pil(chosen_image_tensor[0])[0]
        rejected_image_pil = convert_images_to_pil(rejected_image_tensor[0])[0]

        self.chosen_tokens.append(
            torch.tensor([[chosen_token]], device=self.tokens_list[0].device)
        )
        self.chosen_images.append(chosen_image_pil)
        self.rejected_images.append(rejected_image_pil)

        # Check if game is complete
        if self.current_question >= self.max_questions:
            status = f"🎉 Game Complete! You've answered all {self.max_questions} questions."
            return None, status, self.chosen_images, self.rejected_images

        status = f"Choice saved! Generating question {self.current_question + 1}/{self.max_questions}..."
        return None, status, self.chosen_images, self.rejected_images


# Global game instance
game = TwentyQuestionsGame()


def start_new_game():
    """Initialize and start a new game."""
    game.reset_game()

    if not game.game_initialized:
        init_msg = game.initialize_model()
        print(init_msg)

    option_a_image, option_b_image, status = game.generate_question()

    return (
        option_a_image,  # option_a_gallery (gr.Image)
        option_b_image,  # option_b_gallery (gr.Image)
        status,  # status_text
        gr.update(interactive=True),  # choose_a_btn
        gr.update(interactive=True),  # choose_b_btn
        [],  # choice_history
        [],  # rejected_history
        gr.update(visible=False),  # final_result
    )


def choose_option_a():
    """Handle choosing Option A."""
    _, status, chosen_history, rejected_history = game.make_choice(0)

    if game.current_question >= game.max_questions:
        # Game complete
        final_image = game.chosen_images[-1] if game.chosen_images else None
        return (
            None,  # option_a_gallery (gr.Image)
            None,  # option_b_gallery (gr.Image)
            status,  # status_text
            gr.update(interactive=False),  # choose_a_btn
            gr.update(interactive=False),  # choose_b_btn
            chosen_history,  # choice_history
            rejected_history,  # rejected_history
            gr.update(value=final_image, visible=True),  # final_result
        )

    # Generate next question
    option_a_image, option_b_image, _ = game.generate_question()
    status = f"Question {game.current_question}/{game.max_questions}: Which option better matches your imagined face?"

    return (
        option_a_image,  # option_a_gallery (gr.Image)
        option_b_image,  # option_b_gallery (gr.Image)
        status,  # status_text
        gr.update(interactive=True),  # choose_a_btn
        gr.update(interactive=True),  # choose_b_btn
        chosen_history,  # choice_history
        rejected_history,  # rejected_history
        gr.update(visible=False),  # final_result
    )


def choose_option_b():
    """Handle choosing Option B."""
    _, status, chosen_history, rejected_history = game.make_choice(1)

    if game.current_question >= game.max_questions:
        # Game complete
        final_image = game.chosen_images[-1] if game.chosen_images else None
        return (
            None,  # option_a_gallery (gr.Image)
            None,  # option_b_gallery (gr.Image)
            status,  # status_text
            gr.update(interactive=False),  # choose_a_btn
            gr.update(interactive=False),  # choose_b_btn
            chosen_history,  # choice_history
            rejected_history,  # rejected_history
            gr.update(value=final_image, visible=True),  # final_result
        )

    # Generate next question
    option_a_image, option_b_image, _ = game.generate_question()
    status = f"Question {game.current_question}/{game.max_questions}: Which option better matches your imagined face?"

    return (
        option_a_image,  # option_a_gallery (gr.Image)
        option_b_image,  # option_b_gallery (gr.Image)
        status,  # status_text
        gr.update(interactive=True),  # choose_a_btn
        gr.update(interactive=True),  # choose_b_btn
        chosen_history,  # choice_history
        rejected_history,  # rejected_history
        gr.update(visible=False),  # final_result
    )


def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(title="20 Questions - Face Guessing Game") as demo:
        gr.Markdown(
            """
            # 🎮 20 Questions - Face Guessing Game

            Think of a face in your mind, then answer 20 questions by choosing which option
            better matches your imagined face. The AI will progressively narrow down to your specific face!

            **How it works:**
            - Each question shows you two options (A and B) with sample images
            - Choose the option that better matches your imagined face
            - After 20 questions, see the final generated face!
            """
        )

        # Status and progress
        status_text = gr.Textbox(
            label="Status",
            value="Click 'Start New Game' to begin!",
            interactive=False,
            lines=2
        )

        # Choice history
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ✅ Chosen Images")
                choice_history = gr.Gallery(
                    label="Your Chosen Images",
                    columns=[20],
                    rows=1,
                    height=80,
                    object_fit="contain",
                    preview=False
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### ❌ Rejected Images")
                rejected_history = gr.Gallery(
                    label="Your Rejected Images",
                    columns=[20],
                    rows=1,
                    height=80,
                    object_fit="contain",
                    preview=False
                )

        # Main game area
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🅰️ Option A")
                option_a_gallery = gr.Image(
                    label="Option A",
                    height=300,
                    interactive=False,
                    show_label=False
                )
                choose_a_btn = gr.Button(
                    "Choose Option A",
                    variant="primary",
                    size="lg",
                    interactive=False
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🅱️ Option B")
                option_b_gallery = gr.Image(
                    label="Option B",
                    height=300,
                    interactive=False,
                    show_label=False
                )
                choose_b_btn = gr.Button(
                    "Choose Option B",
                    variant="primary",
                    size="lg",
                    interactive=False
                )

        # Control buttons
        with gr.Row():
            start_btn = gr.Button("🎯 Start New Game", variant="secondary", size="lg")
            gr.Markdown("*First game will download the model (~few GB) and may take a moment to initialize*")

        # Final result (hidden until game complete)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎉 Final Result")
                final_result = gr.Image(
                    label="Your Generated Face!",
                    visible=False,
                    height=300
                )

        # Event handlers
        start_btn.click(
            fn=start_new_game,
            inputs=[],
            outputs=[
                option_a_gallery,
                option_b_gallery,
                status_text,
                choose_a_btn,
                choose_b_btn,
                choice_history,
                rejected_history,
                final_result
            ]
        )

        choose_a_btn.click(
            fn=choose_option_a,
            inputs=[],
            outputs=[
                option_a_gallery,
                option_b_gallery,
                status_text,
                choose_a_btn,
                choose_b_btn,
                choice_history,
                rejected_history,
                final_result
            ]
        )

        choose_b_btn.click(
            fn=choose_option_b,
            inputs=[],
            outputs=[
                option_a_gallery,
                option_b_gallery,
                status_text,
                choose_a_btn,
                choose_b_btn,
                choice_history,
                rejected_history,
                final_result
            ]
        )

        gr.Markdown(
            """
            ---
            ### ℹ️ About

            This game uses a binary FSQ-based FlexTok model trained on CelebA-HQ. Each question corresponds
            to one FlexTok token, with early tokens capturing high-level features and later tokens refining details.

            **Technical Details:**
            - Model: FlexTok with Finite Scalar Quantization (FSQ)
            - Dataset: CelebA-HQ faces
            - Total possible faces: 2^256 clusters
            - Questions: 20 (binary choices narrow down the space)
            """
        )

    return demo


def main():
    """Main entry point for the web application."""
    demo = create_ui()

    # Launch the web interface
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
        share=False,  # Set to True to create a public link
        show_error=True
    )


if __name__ == "__main__":
    main()
