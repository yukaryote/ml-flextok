"""
Here, we play a "20 questions" style face guessing game with visual preferences proposals as questions. 
We do this by sequentially proposing two images from an FSQ-binarized (each token only has two possible values) 
FlexTok model trained on CelebA-HQ; each image corresponds to one of the two possible values of a single FlexTok token. 
The user is then asked to pick which of the two images better matches the face they have in mind. 
This process is repeated for a total of 20 questions (i.e., 20 tokens),
Each token proposal step is conditioned on the user's past preferences. 

The first token in the FlexTok sequence captures the most essential high-level information about an image, 
while the last (256th) token captures the most detailed information. 
In essence, that means FlexTok partitions the distribution of all possible images into 2^256 clusters.
"""
from itertools import product
from typing import Optional
import tqdm
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from diffusers.models import AutoencoderKL

from flextok.flextok_wrapper import FlexTokFromHub, FlexTok
from flextok.utils.demo import imgs_from_urls, denormalize, batch_to_pil
from flextok.utils.misc import detect_bf16_support, get_bf16_context, get_generator

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Global no_grad
torch.set_grad_enabled(False)

# Automatically set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Detect if bf16 is enabled or not
enable_bf16 = detect_bf16_support()
print('BF16 enabled:', enable_bf16)


def load_flextok_model(
        model_name: str, 
        bf16: bool = True, 
        ckpt_path: Optional[str] = None,
        fsq_level: Optional[list[int]] = None
    ) -> FlexTok:
    """
    Load FlexTok model from HuggingFace hub or from checkpoint.
    Args:
        model_name: Name of the FlexTok model on HuggingFace hub.
        bf16: Whether to use bf16 context.
        ckpt_path: Optional[str] = None, path to checkpoint to load weights from.
        fsq_level: Optional[list[int]] = None, list of FSQ levels to override the default ones.
    Returns:
        flextok_model: The loaded FlexTok model.
    """
    flextok = FlexTokFromHub.from_pretrained('EPFL-VILAB/flextok_d18_d18_in1k')
    if ckpt_path is not None:
        if fsq_level is not None:
            print("Overriding FSQ levels with:", fsq_level)
            from flextok.regularizers.quantize_fsq import FSQ
            from flextok.model.postprocessors.heads import LinearHead
            from flextok.model.preprocessors.linear import LinearLayer
            new_levels = fsq_level

            # Get the original FSQ configuration
            old_fsq: FSQ = flextok.regularizer

            # Check if the encoder output dimension matches the fsq levels length
            old_fsq_output_dim = old_fsq.dim
            if old_fsq_output_dim != len(new_levels):
                # project encoder dim to new fsq dim
                print(f"Adjusting encoder output dimension from {old_fsq_output_dim} to {len(new_levels)}")
                new_enc_linear_head = LinearHead(
                    read_key=flextok.encoder.module_dict["enc_to_latents"].read_key,
                    write_key=flextok.encoder.module_dict["enc_to_latents"].write_key,
                    dim=flextok.encoder.module_dict["enc_to_latents"].dim_in,
                    dim_out=len(new_levels),
                    use_mup_readout=False,
                )

                flextok.encoder.module_dict['enc_to_latents'] = new_enc_linear_head

                # project back from fsq to decoder input dim
                print(f"Adjusting decoder input dimension to {len(new_levels)}")
                new_dec_linear_head = LinearLayer(
                    read_key=flextok.decoder.module_dict["dec_from_latents"].read_key,
                    write_key=flextok.decoder.module_dict["dec_from_latents"].write_key,
                    dim_in=len(new_levels),
                    dim=flextok.decoder.module_dict["dec_from_latents"].dim_out,
                )
                flextok.decoder.module_dict['dec_from_latents'] = new_dec_linear_head

            # Create new FSQ with modified levels
            new_fsq = FSQ(
                latents_read_key=old_fsq.latents_read_key,
                quants_write_key=old_fsq.quants_write_key,
                tokens_write_key=old_fsq.tokens_write_key,
                levels=new_levels,
                drop_quant_p=old_fsq.drop_quant_p,
                corrupt_tokens_p=old_fsq.corrupt_tokens_p,
                min_corrupt_tokens_p=old_fsq.min_corrupt_tokens_p,
                apply_corrupt_tokens_p=old_fsq.apply_corrupt_tokens_p,
                packed_call=old_fsq.packed_call,
            )

            # Replace the FSQ module
            flextok.regularizer = new_fsq
        checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)
        flextok.load_state_dict(checkpoint['model_state_dict'])
    flextok = flextok.to(device).eval()
    return flextok


def get_possible_combos(flextok_model: FlexTok):
    """
    Get all possible zhats (quantized latents) from the FlexTok model.
    Args:
        flextok_model: The FlexTok model.
        num_samples: Number of samples to generate.
    Returns:
        batch of zhats (num_samples, d).
    """
    # Get the FSQ from flextok model
    fsq = flextok_model.regularizer
    fsq_levels = fsq._levels  # e.g., [8, 8, 8, 5, 5, 5]
    print("FSQ levels:", fsq_levels)
    print("codebook size:", fsq.codebook_size)

    quantizations = [torch.linspace(-1, 1, steps=L) for L in fsq_levels]
    all_combinations = list(product(*quantizations))
    print("Total combinations (must equal codebook size):", len(all_combinations))
    
    return torch.stack([torch.tensor(comb) for comb in all_combinations], dim=0)


def zhat_to_tokens(flextok_model: FlexTok, zhats: torch.Tensor):
    """
    Given list of zhats, generate tokens from zhats.
    Args:
        flextok_model: The FlexTok model.
        zhats: zhats (N, d).
    Returns:
        tokens (N, 1).
    """
    fsq = flextok_model.regularizer
    tokens = fsq.codes_to_indices(zhats)  # (N, 1)
    print("tokens shape:", tokens.shape)
    return tokens.long()  # type: ignore


def sample_images_per_quantization(
        flextok_model: FlexTok, 
        possible_tokens_list: list[torch.Tensor], 
        num_samples_per_quantization: int,
        condition_tokens: list[torch.Tensor] = [],
        bf16: bool = True
    ) -> dict[int, torch.Tensor]:
    """
    Given all possible tokens, sample images from each token.
    Args:
        flextok_model: The FlexTok model.
        possible_tokens_list: N-length list of tokens, each of shape (1, 1) (N=2 in binary case)
        num_samples_per_quantization: Number of samples to generate per quantization combination.
        condition_tokens: Optional list of past tokens to be passed as conditioning.
        bf16: Whether to use bf16 context.
    Returns:
        images: dict with keys as elements of tokens_list and elements are tensors of shape (num_samples_per_quantization, C, H, W)
    """
    imgs_per_quantization = {}
    for i in tqdm.tqdm(range(0, len(possible_tokens_list))):
        print("tokens list", possible_tokens_list)
        batch_tokens_list = possible_tokens_list[i:i+1]  # list with one (1, 1)-shape tensor
        print("batch_tokens_list", batch_tokens_list)
        print('condition_tokens', condition_tokens)
        final_condition_tokens = condition_tokens + batch_tokens_list
        print("final_condition_tokens", final_condition_tokens)
        reconstructions = []
        for _ in range(num_samples_per_quantization):
            with get_bf16_context(bf16):
                with torch.no_grad():
                    reconst = flextok_model.detokenize(
                        final_condition_tokens,
                        timesteps=25, # Number of denoising steps
                        guidance_scale=7.5, # Classifier-free guidance scale
                        perform_norm_guidance=True, # APG, see https://arxiv.org/abs/2410.02416
                        # Optionally control initial noise. Note that while the initial noise is deterministic, the rest of the model isn't.
                        generator=None,
                        verbose=False, # Enable to show denoising progress bar with tqdm
                    )
                    reconstructions.append(reconst.cpu())
        imgs_per_quantization[possible_tokens_list[i].item()] = torch.stack(reconstructions, dim=0)
    return imgs_per_quantization


def convert_images_to_pil(images: torch.Tensor) -> list[Image.Image]:
    images_denorm = denormalize(images).clamp(0, 1)
    images_pil_list = []
    for img in images_denorm:
        images_pil = TF.to_pil_image(img.cpu())
        images_pil_list.append(images_pil)
    return images_pil_list


def main(
    num_samples_per_quantization: int = 1,
    enable_bf16: bool = True
):
    """
    Main function to run the 20 Questions demo.
    """
    # Load FlexTok model
    flextok_model = load_flextok_model(
        model_name='EPFL-VILAB/flextok_d18_d18_in1k',
        bf16=enable_bf16,
        ckpt_path=None,
        fsq_level=None
    )

    # Get all possible quantization combinations
    all_zhats = get_possible_combos(flextok_model)

    # Convert zhats to tokens
    tokens_list = zhat_to_tokens(flextok_model, all_zhats)
    # DEBUG: change last one to [[1]] instead of [[2]]
    tokens_list[-1] = torch.tensor([[1]], device=tokens_list[-1].device)
    print("Tokens list (REPLACED LAST WITH [[1]]):", [t.item() for t in tokens_list])
    print("Total tokens to sample from:", len(tokens_list))

    chosen_tokens = []
    chosen_images = []
    iters = 0
    terminated = False
    while iters < 20 and not terminated:
        # TODO: display chosen_images so far in a sequence on the top of the screen
        # Sample images from tokens
        images_dict = sample_images_per_quantization(
            flextok_model,
            tokens_list,
            num_samples_per_quantization=num_samples_per_quantization,
            condition_tokens=chosen_tokens,
            bf16=enable_bf16
        )  # dict[int, torch.Tensor] with tensor values of shape (num_samples_per_quantization, C, H, W)

        # TODO: display each set of images in separate grids on the left and right of the screen
        # TODO: implement logic to let user choose which side is preferred
        user_choice_token: Optional[int] = None  # Placeholder for user choice
        user_choice_image = images_dict[user_choice_token]  # Placeholder for user choice image
        user_choice_image_pil = convert_images_to_pil(user_choice_image)[0]
        chosen_tokens.append(torch.tensor([[user_choice_token]], device=tokens_list[0].device))
        chosen_images.append(user_choice_image_pil)
        iters += 1


if __name__ == "__main__":
    main()