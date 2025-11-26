"""
Tkinter-based UI for the "20 Questions" face guessing game.
This provides an interactive interface for the FlexTok visual preference game.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
from typing import Optional, Dict, List
import os
import sys

# Add parent directory to path to import flextok modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms.functional as TF
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


class TwentyQuestionsUI:
    """Main UI class for the 20 Questions face guessing game."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("20 Questions - Face Guessing Game")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2b2b2b")

        # Game state
        self.flextok_model: Optional[FlexTok] = None
        self.tokens_list: List[torch.Tensor] = []
        self.chosen_tokens: List[torch.Tensor] = []
        self.chosen_images: List[Image.Image] = []
        self.current_question = 0
        self.max_questions = 20
        self.num_samples_per_quantization = 4
        self.enable_bf16 = detect_bf16_support()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Current question data
        self.current_images_dict: Dict[int, torch.Tensor] = {}
        self.current_tokens: List[int] = []

        # UI elements
        self.thumbnail_labels: List[tk.Label] = []
        self.left_image_labels: List[tk.Label] = []
        self.right_image_labels: List[tk.Label] = []

        # Setup UI
        self.setup_ui()

        # Show loading screen and initialize model
        self.show_loading_screen("Initializing model...")
        threading.Thread(target=self.initialize_model, daemon=True).start()

    def setup_ui(self):
        """Setup the main UI layout."""
        # Top panel - Progress and thumbnail history
        self.setup_top_panel()

        # Center panel - Image comparison grids
        self.setup_center_panel()

        # Bottom panel - Buttons and question counter
        self.setup_bottom_panel()

        # Loading overlay (hidden by default)
        self.setup_loading_overlay()

    def setup_top_panel(self):
        """Setup the top panel with progress bar and thumbnail history."""
        top_frame = tk.Frame(self.root, bg="#2b2b2b", height=120)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        top_frame.pack_propagate(False)

        # Progress bar
        progress_frame = tk.Frame(top_frame, bg="#2b2b2b")
        progress_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        tk.Label(
            progress_frame,
            text="Progress:",
            font=("Arial", 12, "bold"),
            bg="#2b2b2b",
            fg="white"
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=self.max_questions,
            length=400
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.question_label = tk.Label(
            progress_frame,
            text="Question 0/20",
            font=("Arial", 12, "bold"),
            bg="#2b2b2b",
            fg="white"
        )
        self.question_label.pack(side=tk.LEFT, padx=(10, 0))

        # Thumbnail history strip
        thumbnail_frame = tk.Frame(top_frame, bg="#3b3b3b")
        thumbnail_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        tk.Label(
            thumbnail_frame,
            text="Your Choices:",
            font=("Arial", 10),
            bg="#3b3b3b",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)

        # Scrollable thumbnail container
        self.thumbnail_canvas = tk.Canvas(thumbnail_frame, bg="#3b3b3b", highlightthickness=0)
        self.thumbnail_scrollbar = ttk.Scrollbar(
            thumbnail_frame,
            orient=tk.HORIZONTAL,
            command=self.thumbnail_canvas.xview
        )
        self.thumbnail_container = tk.Frame(self.thumbnail_canvas, bg="#3b3b3b")

        self.thumbnail_canvas.configure(xscrollcommand=self.thumbnail_scrollbar.set)
        self.thumbnail_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.thumbnail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.thumbnail_canvas.create_window((0, 0), window=self.thumbnail_container, anchor=tk.NW)
        self.thumbnail_container.bind("<Configure>", self.on_thumbnail_configure)

    def setup_center_panel(self):
        """Setup the center panel with left and right image grids."""
        center_frame = tk.Frame(self.root, bg="#2b2b2b")
        center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left grid
        left_frame = tk.LabelFrame(
            center_frame,
            text="Option A",
            font=("Arial", 14, "bold"),
            bg="#3b3b3b",
            fg="white",
            relief=tk.RAISED,
            borderwidth=3
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.left_grid_frame = tk.Frame(left_frame, bg="#3b3b3b")
        self.left_grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right grid
        right_frame = tk.LabelFrame(
            center_frame,
            text="Option B",
            font=("Arial", 14, "bold"),
            bg="#3b3b3b",
            fg="white",
            relief=tk.RAISED,
            borderwidth=3
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.right_grid_frame = tk.Frame(right_frame, bg="#3b3b3b")
        self.right_grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def setup_bottom_panel(self):
        """Setup the bottom panel with choice buttons."""
        bottom_frame = tk.Frame(self.root, bg="#2b2b2b", height=100)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        bottom_frame.pack_propagate(False)

        # Button frame
        button_frame = tk.Frame(bottom_frame, bg="#2b2b2b")
        button_frame.pack(expand=True)

        # Left choice button
        self.left_button = tk.Button(
            button_frame,
            text="Choose Option A",
            font=("Arial", 16, "bold"),
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            width=20,
            height=2,
            command=lambda: self.make_choice(0),
            state=tk.DISABLED
        )
        self.left_button.pack(side=tk.LEFT, padx=20)

        # Right choice button
        self.right_button = tk.Button(
            button_frame,
            text="Choose Option B",
            font=("Arial", 16, "bold"),
            bg="#2196F3",
            fg="white",
            activebackground="#0b7dda",
            width=20,
            height=2,
            command=lambda: self.make_choice(1),
            state=tk.DISABLED
        )
        self.right_button.pack(side=tk.LEFT, padx=20)

        # Restart button
        self.restart_button = tk.Button(
            button_frame,
            text="Restart Game",
            font=("Arial", 12),
            bg="#f44336",
            fg="white",
            activebackground="#da190b",
            width=15,
            command=self.restart_game
        )
        self.restart_button.pack(side=tk.LEFT, padx=20)

    def setup_loading_overlay(self):
        """Setup loading overlay for async operations."""
        self.loading_frame = tk.Frame(self.root, bg="#000000")
        self.loading_label = tk.Label(
            self.loading_frame,
            text="Loading...",
            font=("Arial", 24, "bold"),
            bg="#000000",
            fg="white"
        )
        self.loading_label.pack(expand=True)
        self.loading_progress = ttk.Progressbar(
            self.loading_frame,
            mode='indeterminate',
            length=300
        )
        self.loading_progress.pack(pady=20)

    def show_loading_screen(self, message: str):
        """Show the loading overlay."""
        self.loading_label.config(text=message)
        self.loading_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.loading_progress.start()
        self.root.update()

    def hide_loading_screen(self):
        """Hide the loading overlay."""
        self.loading_progress.stop()
        self.loading_frame.place_forget()
        self.root.update()

    def initialize_model(self):
        """Initialize the FlexTok model and prepare for the game."""
        try:
            # Load model
            self.flextok_model = load_flextok_model(
                model_name='EPFL-VILAB/flextok_d18_d18_in1k',
                bf16=self.enable_bf16,
                ckpt_path=None,
                fsq_level=None
            )

            # Get all possible tokens
            all_zhats = get_possible_combos(self.flextok_model)
            self.tokens_list = zhat_to_tokens(self.flextok_model, all_zhats)

            # DEBUG: change last one to [[1]] instead of [[2]]
            self.tokens_list[-1] = torch.tensor([[1]], device=self.tokens_list[-1].device)

            print(f"Initialized with {len(self.tokens_list)} possible tokens")

            # Start first question
            self.root.after(0, self.hide_loading_screen)
            self.root.after(100, self.next_question)

        except Exception as e:
            self.root.after(0, self.hide_loading_screen)
            self.root.after(0, lambda: messagebox.showerror(
                "Initialization Error",
                f"Failed to initialize model:\n{str(e)}"
            ))

    def next_question(self):
        """Generate and display the next question."""
        if self.current_question >= self.max_questions:
            self.show_final_result()
            return

        self.current_question += 1
        self.progress_var.set(self.current_question - 1)
        self.question_label.config(text=f"Question {self.current_question}/{self.max_questions}")

        # Disable buttons during generation
        self.left_button.config(state=tk.DISABLED)
        self.right_button.config(state=tk.DISABLED)

        # Show loading and generate images in background
        self.show_loading_screen(f"Generating options for question {self.current_question}...")
        threading.Thread(target=self.generate_question_images, daemon=True).start()

    def generate_question_images(self):
        """Generate images for the current question (runs in background thread)."""
        try:
            # Sample images from tokens
            images_dict = sample_images_per_quantization(
                self.flextok_model,
                self.tokens_list,
                num_samples_per_quantization=self.num_samples_per_quantization,
                condition_tokens=self.chosen_tokens,
                bf16=self.enable_bf16
            )

            self.current_images_dict = images_dict
            self.current_tokens = list(images_dict.keys())

            # Update UI in main thread
            self.root.after(0, self.display_question_images)

        except Exception as e:
            self.root.after(0, self.hide_loading_screen)
            self.root.after(0, lambda: messagebox.showerror(
                "Generation Error",
                f"Failed to generate images:\n{str(e)}"
            ))

    def display_question_images(self):
        """Display the generated images in the UI grids."""
        self.hide_loading_screen()

        # Clear previous grids
        for widget in self.left_grid_frame.winfo_children():
            widget.destroy()
        for widget in self.right_grid_frame.winfo_children():
            widget.destroy()

        self.left_image_labels.clear()
        self.right_image_labels.clear()

        # Display images in grids
        for idx, token in enumerate(self.current_tokens[:2]):  # Only show first 2 options
            images_tensor = self.current_images_dict[token]
            images_pil = convert_images_to_pil(images_tensor)

            target_frame = self.left_grid_frame if idx == 0 else self.right_grid_frame
            target_list = self.left_image_labels if idx == 0 else self.right_image_labels

            # Calculate grid layout (2x2 for 4 images)
            cols = 2
            for img_idx, img_pil in enumerate(images_pil):
                # Resize for display
                display_size = (200, 200)
                img_resized = img_pil.resize(display_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img_resized)

                row = img_idx // cols
                col = img_idx % cols

                label = tk.Label(target_frame, image=photo, bg="#3b3b3b")
                label.image = photo  # Keep reference
                label.grid(row=row, column=col, padx=5, pady=5)
                target_list.append(label)

        # Enable choice buttons
        self.left_button.config(state=tk.NORMAL)
        self.right_button.config(state=tk.NORMAL)

    def make_choice(self, choice_idx: int):
        """Handle user's choice (0 for left, 1 for right)."""
        if choice_idx >= len(self.current_tokens):
            return

        chosen_token = self.current_tokens[choice_idx]
        chosen_image_tensor = self.current_images_dict[chosen_token]
        chosen_image_pil = convert_images_to_pil(chosen_image_tensor)[0]

        # Save choice
        self.chosen_tokens.append(
            torch.tensor([[chosen_token]], device=self.tokens_list[0].device)
        )
        self.chosen_images.append(chosen_image_pil)

        # Add thumbnail to history
        self.add_thumbnail(chosen_image_pil)

        # Update progress
        self.progress_var.set(self.current_question)

        # Move to next question
        self.next_question()

    def add_thumbnail(self, image: Image.Image):
        """Add a thumbnail to the history strip."""
        thumbnail_size = (60, 60)
        img_resized = image.resize(thumbnail_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_resized)

        label = tk.Label(self.thumbnail_container, image=photo, bg="#3b3b3b")
        label.image = photo  # Keep reference
        label.pack(side=tk.LEFT, padx=2)

        self.thumbnail_labels.append(label)

        # Auto-scroll to the right
        self.thumbnail_canvas.update_idletasks()
        self.thumbnail_canvas.xview_moveto(1.0)

    def on_thumbnail_configure(self, event):
        """Update scroll region when thumbnail container changes."""
        self.thumbnail_canvas.configure(scrollregion=self.thumbnail_canvas.bbox("all"))

    def show_final_result(self):
        """Display the final result after 20 questions."""
        # Create final result window
        result_window = tk.Toplevel(self.root)
        result_window.title("Final Result!")
        result_window.geometry("600x700")
        result_window.configure(bg="#2b2b2b")

        # Title
        tk.Label(
            result_window,
            text="Your Face is Complete!",
            font=("Arial", 20, "bold"),
            bg="#2b2b2b",
            fg="white"
        ).pack(pady=20)

        # Display final image (last chosen image)
        if self.chosen_images:
            final_image = self.chosen_images[-1]
            display_size = (400, 400)
            img_resized = final_image.resize(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_resized)

            img_label = tk.Label(result_window, image=photo, bg="#2b2b2b")
            img_label.image = photo
            img_label.pack(pady=20)

        # Buttons
        button_frame = tk.Frame(result_window, bg="#2b2b2b")
        button_frame.pack(pady=20)

        tk.Button(
            button_frame,
            text="Save Image",
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            command=lambda: self.save_final_image(final_image),
            width=15
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            button_frame,
            text="Play Again",
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            command=lambda: [result_window.destroy(), self.restart_game()],
            width=15
        ).pack(side=tk.LEFT, padx=10)

    def save_final_image(self, image: Image.Image):
        """Save the final image to disk."""
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filepath:
            image.save(filepath)
            messagebox.showinfo("Success", f"Image saved to:\n{filepath}")

    def restart_game(self):
        """Restart the game from the beginning."""
        # Reset game state
        self.chosen_tokens.clear()
        self.chosen_images.clear()
        self.current_question = 0
        self.progress_var.set(0)

        # Clear thumbnails
        for label in self.thumbnail_labels:
            label.destroy()
        self.thumbnail_labels.clear()

        # Clear grids
        for widget in self.left_grid_frame.winfo_children():
            widget.destroy()
        for widget in self.right_grid_frame.winfo_children():
            widget.destroy()

        # Start first question
        self.next_question()


def main():
    """Main entry point for the UI application."""
    # Setup PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Create and run UI
    root = tk.Tk()
    app = TwentyQuestionsUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
