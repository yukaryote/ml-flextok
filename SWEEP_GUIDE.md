# Wandb Sweep Guide for FlexTok Fine-tuning

This guide shows how to use wandb sweeps to find optimal hyperparameters for fine-tuning FlexTok.

## Quick Start

### Option 1: Using the Python script (Recommended)

```bash
# 1. Initialize and run a sweep with 10 trials
python run_sweep.py --init --count 10

# 2. Or initialize first, then run agent separately
python run_sweep.py --init
python run_sweep.py --agent YOUR_SWEEP_ID --count 10
```

### Option 2: Using wandb CLI directly

```bash
# 1. Initialize sweep
wandb sweep sweep_config.yaml

# 2. Run agent (copy the sweep ID from step 1)
wandb agent YOUR_SWEEP_ID
```

## Sweep Configurations

### `sweep_config.yaml` (Full)
- Sweeps: learning_rate, encoder_lr, decoder_lr, warmup_steps, max_grad_norm, weight_decay
- Method: Bayesian optimization
- Good for comprehensive search

### `sweep_config_simple.yaml` (Simple)
- Sweeps: learning_rate, warmup_steps
- Method: Bayesian optimization
- Good for quick exploration

## Modifying Training Script for Sweeps

The training script already supports command-line overrides via wandb config. You can pass hyperparameters like:

```bash
python train_flextok.py --config configs/train_celebahq.yaml \
    learning_rate=1e-5 \
    warmup_steps=200 \
    max_grad_norm=1.0
```

## Monitoring Results

1. Go to your wandb project dashboard
2. Navigate to the "Sweeps" tab
3. Click on your sweep to see:
   - Parallel coordinates plot
   - Parameter importance
   - Best runs

## Tips

1. **Start with simple sweep**: Use `sweep_config_simple.yaml` first to find a good learning rate range
2. **Early termination**: The sweep config includes early termination to stop poorly performing runs
3. **Parallel runs**: Run multiple agents on different GPUs:
   ```bash
   # Terminal 1 (GPU 0)
   CUDA_VISIBLE_DEVICES=0 python run_sweep.py --agent SWEEP_ID --count 5

   # Terminal 2 (GPU 1)
   CUDA_VISIBLE_DEVICES=1 python run_sweep.py --agent SWEEP_ID --count 5
   ```

4. **Monitor memory**: If you get OOM, reduce batch_size in the config:
   ```bash
   python train_flextok.py --config configs/train_celebahq.yaml \
       batch_size=8 \
       gradient_accumulation_steps=8
   ```

## Common Issues

### Issue: "wandb not installed"
**Solution**: `pip install wandb`

### Issue: "Not logged in to wandb"
**Solution**: `wandb login`

### Issue: Runs failing with OOM
**Solution**:
1. Reduce batch size in `configs/train_celebahq.yaml`
2. Or add `batch_size=8` to sweep parameters

### Issue: Training too slow
**Solution**:
1. Reduce `num_epochs` in config for sweep
2. Use early termination (already configured)
3. Reduce validation frequency

## Example: Finding Best Learning Rate

```bash
# 1. Create a learning rate focused sweep
wandb sweep sweep_config_simple.yaml

# 2. Run 20 trials
python run_sweep.py --agent SWEEP_ID --count 20

# 3. Check wandb dashboard for best learning rate

# 4. Update configs/train_celebahq.yaml with best value
# learning_rate: <best_value>
# warmup_steps: <best_value>

# 5. Run full training with best hyperparameters
python train_flextok.py --config configs/train_celebahq.yaml
```

## Customizing Sweep

To add more parameters to sweep, edit `sweep_config.yaml`:

```yaml
parameters:
  # Add any parameter from your config
  batch_size:
    values: [8, 16, 32]

  use_ema:
    values: [true, false]

  ema_decay:
    distribution: uniform
    min: 0.99
    max: 0.9999
```

Then the sweep will automatically try these combinations!
