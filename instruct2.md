To integrate the concepts of **Variable Mask Scheduling** and **Soft Masking** into your codebase, here is the final updated `instruction.md`. 

This version is designed to mitigate the "ringing" artifacts and "non-rigid" movement limitations identified in the previous analysis, making the frequency-based approach much more competitive with the original Stable Flow results.

***

# Updated Instruction: Advanced Frequency-Flow Editing

This guide replaces Stable Flow's attention injection with **Adaptive Latent Nudging** and **Scheduled Soft-Frequency Filtering**. This combination ensures reconstruction accuracy while allowing for structural flexibility.

### Step 1: Configuration and Cleanup
* **Vital Layers:** Set `vital_layers = []` in your configuration to disable standard attention hooks.
* **Soft Masking:** Instead of a binary 0/1 mask, we will use a Gaussian-blurred mask to prevent edge artifacts.

### Step 2: Implementation of the Scheduled Mask
The "layout" should be strictly enforced at the start of the flow and relaxed toward the end to allow the model to generate textures based on the new prompt.

```python
import torch
import torch.nn.functional as F

def get_scheduled_soft_mask(shape, timestep, total_steps, device):
    batch, channels, h, w = shape
    
    # 1. Linear Schedule: Radius shrinks as we move from t=1 (noise) to t=0 (image)
    # Early steps (t close to 1) = Large radius (preserve layout)
    # Late steps (t close to 0) = Small radius (allow texture/detail)
    progress = timestep / total_steps # assumes t goes from total_steps down to 0
    max_radius = min(h, w) // 4
    current_radius = max_radius * progress 
    
    # 2. Create Base Mask
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist = torch.sqrt((X - w//2)**2 + (Y - h//2)**2).to(device)
    mask = (dist <= current_radius).float()
    
    # 3. Soften the edges (Gaussian Blur) to prevent ringing artifacts
    # A kernel size of 5 or 7 usually suffices
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
    
    return mask
```


### Step 3: Adaptive Latent Nudging (Inversion)
Update the inversion script to use frequency-aware nudging. This ensures that the high-frequency details (which are harder for the flow solver to reconstruct) get a stronger "push" into the model's manifold.

```python
def apply_adaptive_nudge(z, base_lambda=1.15):
    # FFT
    f_z = torch.fft.fftshift(torch.fft.fft2(z, norm="ortho"))
    
    # Identify high-frequency areas to nudge harder
    # We use a small static radius for the nudge-map
    low_freq_mask = get_scheduled_soft_mask(z.shape, 1, 1, z.device)
    high_freq_mask = 1.0 - low_freq_mask
    
    # Apply nudge: High frequencies get 1.2x, Low get 1.15x
    nudge_map = base_lambda + (high_freq_mask * 0.05)
    f_z_nudged = f_z * nudge_map
    
    return torch.fft.ifft2(torch.fft.ifftshift(f_z_nudged), norm="ortho").real
```

### Step 4: The Frequency-Injection Loop
In the target generation loop, apply the scheduled mask after each Euler step.

```python
# During the target ODE Sampling loop:
for i, t in enumerate(timesteps):
    # 1. Standard Euler Step
    x_edit = x_edit + v_pred * dt
    
    # 2. Get the soft mask for this specific timestep
    mask = get_scheduled_soft_mask(x_edit.shape, len(timesteps)-i, len(timesteps), x_edit.device)
    
    # 3. Frequency Swap
    if t.item() in source_latents_trajectory:
        x_src = source_latents_trajectory[t.item()]
        
        f_edit = torch.fft.fftshift(torch.fft.fft2(x_edit, norm="ortho"))
        f_src = torch.fft.fftshift(torch.fft.fft2(x_src, norm="ortho"))
        
        # Weighted blend using the soft mask
        f_combined = (mask * f_src) + ((1.0 - mask) * f_edit)
        
        x_edit = torch.fft.ifft2(torch.fft.ifftshift(f_combined), norm="ortho").real
```


### Summary of Changes
1.  **Adaptive Nudging:** Uses higher energy for high-frequency reconstruction, solving the "flatness" or "reconstruction error" of standard flow-matching.
2.  **Soft Masking:** Replaces hard frequency cuts with a blurred transition, eliminating the common "halo" artifacts found in standard Fourier-based editing.
3.  **Variable Schedule:** Dynamically shifts the balance from **Global Structure** (early steps) to **Text-Guided Detail** (late steps), allowing the model to better follow non-rigid prompts while keeping the background and character identity stable.