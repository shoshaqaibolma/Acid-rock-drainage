"""
Training Evolution Tracker
==========================
Saves profiles at multiple checkpoints during 10k training to:
1. Track model development over time
2. Detect overfitting
3. Identify optimal stopping point
4. Visualize learning dynamics
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import glob
import os
import sys
import types

import ard_b1_physicsnemo_improved as pinn

# Configuration
params = {
    'domain_L': 5.0,
    'v': 0.0,
    'D': [2.0e-9, 5.0e-10, 5.0e-10, 1.0e-9, 9.0e-9],
    'henry': {'Hcp_O2': 1.27e-3},
    'pO2_top': 0.21,
    'pO2_bottom': 0.0,
    'pH_top': 5.0,
    'kL_O2': 1e-5,
    'r0_pyrite': 1e-12,
    'r0_fe_ox': 0.0,
    'z_kinetic': 2.5,
    'O2_Km': 1e-5,
}

device = torch.device('cpu')

# Select architecture
model_type = sys.argv[1] if len(sys.argv) > 1 else 'mlp'

print("="*80)
print(f"TRAINING EVOLUTION TRACKER: {model_type.upper()}")
print("="*80)
print("Checkpoints: Every 1000 steps (0, 1000, 2000, ..., 10000)")
print("Outputs: Profiles + metrics at each checkpoint")
print("="*80 + "\n")

# Create output directory
os.makedirs('evolution', exist_ok=True)

# ========== TRAIN WITH PERIODIC EVALUATION ==========
cfg = types.SimpleNamespace(
    device='cpu',
    model=model_type,
    batch_size=8,
    seq_len=64,
    lr=1e-4,
    wd=1e-4,
    steps=10000,
    log_every=500,
    use_time=False,
    debug_residuals=False,
    seed=42
)

# Custom training loop with intermediate saves
print("Starting training with evolution tracking...\n")

# Initialize
pinn.random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

model = pinn.build_model(cfg, device, params)
loss_fn = pinn.PhysicsLoss(params, device).to(device)
all_params = list(model.parameters()) + list(loss_fn.parameters())
opt = torch.optim.Adam(all_params, lr=cfg.lr, weight_decay=cfg.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

L_domain = float(params['domain_L'])

# Tracking arrays
evolution_data = []
checkpoint_steps = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

print(f"Step | Total Loss | Henry [%] | EN_p95 | LR")
print("-"*60)

import time
start_time = time.time()

for step in range(cfg.steps + 1):
    # Training step
    z, t = pinn.make_batch(cfg.batch_size, cfg.seq_len, L_domain, device, cfg.use_time)
    L_total, logs = loss_fn(model, z, t, step, cfg)
    
    if step > 0:  # Skip gradient update at step 0
        opt.zero_grad(set_to_none=True)
        L_total.backward()
        all_params_list = list(model.parameters()) + list(loss_fn.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params_list, max_norm=1.0)
        
        if torch.isfinite(grad_norm):
            opt.step()
            sched.step()
    
    # Log progress
    if step % 500 == 0:
        lr = opt.param_groups[0]['lr']
        print(f"{step:5d} | {logs['L_total']:10.3e} | {logs['henry_rel_err']*100:8.2f} | "
              f"{logs['EN_p95']:.2e} | {lr:.2e}")
    
    # Save checkpoint and analyze at key steps
    if step in checkpoint_steps:
        print(f"\n  [CHECKPOINT {step}] Saving profile and metrics...")
        
        # Analyze current model
        results = pinn.analyze_predictions(model, params, device, num_points=200)
        
        # Save profile CSV
        csv_file = f'evolution/{model_type}_step{step:05d}_profile.csv'
        df = pd.DataFrame({
            'z_m': results['z'],
            'O2_M': results['O2'],
            'Fe2_M': results['Fe2'],
            'Fe3_M': results['Fe3'],
            'SO4_M': results['SO4'],
            'pH': results['pH']
        })
        df.to_csv(csv_file, index=False)
        
        # Track evolution
        evolution_data.append({
            'step': step,
            'henry_rel_err': results['henry_rel_err'],
            'EN_max': results['EN_max'],
            'O2_min': results['O2'].min(),
            'O2_max': results['O2'].max(),
            'Fe2_min': results['Fe2'].min(),
            'Fe2_max': results['Fe2'].max(),
            'pH_min': results['pH'].min(),
            'pH_max': results['pH'].max(),
        })
        
        print(f"     Henry: {results['henry_rel_err']:.2%}, EN: {results['EN_max']:.2e} M")
        print(f"     Saved: {csv_file}\n")
    
    # Save checkpoint
    if step > 0 and step % 1000 == 0:
        pinn.save_checkpoint(model, params, step, ckpt_dir="ckpts")

elapsed = time.time() - start_time
print("-"*60)
print(f"TRAINING COMPLETE: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print("="*80)

# ========== EVOLUTION ANALYSIS ==========
print("\nANALYZING TRAINING EVOLUTION...")

df_evolution = pd.DataFrame(evolution_data)
df_evolution.to_csv(f'evolution/{model_type}_evolution_metrics.csv', index=False)
print(f"Saved: evolution/{model_type}_evolution_metrics.csv")

# Create evolution plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Henry's Law evolution
ax = axes[0, 0]
ax.plot(df_evolution['step'], df_evolution['henry_rel_err']*100, 'b-o', linewidth=2, markersize=6)
ax.axhline(y=1.0, color='g', linestyle='--', label='1% target', alpha=0.7)
ax.axhline(y=10.0, color='orange', linestyle='--', label='10% acceptable', alpha=0.7)
ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Henry Rel. Error [%]', fontsize=12, fontweight='bold')
ax.set_title("Henry's Law Convergence", fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 2: Electroneutrality evolution
ax = axes[0, 1]
ax.plot(df_evolution['step'], df_evolution['EN_max'], 'r-o', linewidth=2, markersize=6)
ax.axhline(y=1e-3, color='g', linestyle='--', label='1e-3 M target', alpha=0.7)
ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax.set_ylabel('EN Max [mol/L]', fontsize=12, fontweight='bold')
ax.set_title('Electroneutrality Convergence', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 3: O2 range evolution (overfitting detector)
ax = axes[1, 0]
O2_range = (df_evolution['O2_max'] - df_evolution['O2_min']) * 1e6
ax.plot(df_evolution['step'], O2_range, 'g-o', linewidth=2, markersize=6)
ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax.set_ylabel('O₂ Range [µM]', fontsize=12, fontweight='bold')
ax.set_title('O₂ Gradient Development', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 4: Fe2 range evolution
ax = axes[1, 1]
Fe2_range = (df_evolution['Fe2_max'] - df_evolution['Fe2_min']) * 1e6
ax.plot(df_evolution['step'], Fe2_range, 'm-o', linewidth=2, markersize=6)
ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Fe²⁺ Range [µM]', fontsize=12, fontweight='bold')
ax.set_title('Fe²⁺ Gradient Development', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

fig.suptitle(f'{model_type.upper()}: Training Evolution (10k steps)', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])

evolution_plot = f'evolution/{model_type}_training_evolution.png'
plt.savefig(evolution_plot, dpi=200, bbox_inches='tight')
print(f"Saved: {evolution_plot}")
plt.close()

# ========== OVERFITTING DETECTION ==========
print("\n" + "="*80)
print("OVERFITTING ANALYSIS")
print("="*80)

# Check if metrics are still improving or plateaued/degrading
henry_trend = df_evolution['henry_rel_err'].values
en_trend = df_evolution['EN_max'].values

# Last 3 checkpoints
if len(henry_trend) >= 4:
    henry_recent = henry_trend[-3:]
    en_recent = en_trend[-3:]
    
    henry_improving = henry_recent[-1] < henry_recent[0] * 1.1
    en_improving = en_recent[-1] < en_recent[0] * 1.1
    
    print(f"\nHenry's Law (last 3 checkpoints):")
    print(f"  {henry_recent[0]:.2%} -> {henry_recent[1]:.2%} -> {henry_recent[-1]:.2%}")
    print(f"  Status: {'Improving' if henry_improving else 'Plateaued/Degrading'}")
    
    print(f"\nElectroneutrality (last 3 checkpoints):")
    print(f"  {en_recent[0]:.2e} -> {en_recent[1]:.2e} -> {en_recent[-1]:.2e}")
    print(f"  Status: {'Improving' if en_improving else 'Plateaued/Degrading'}")
    
    # Recommend optimal checkpoint
    best_checkpoint_idx = np.argmin(henry_trend + en_trend * 1e4)  # Weighted combination
    best_step = checkpoint_steps[best_checkpoint_idx]
    print(f"\nRecommended checkpoint: Step {best_step}")
    print(f"  Henry: {henry_trend[best_checkpoint_idx]:.2%}")
    print(f"  EN: {en_trend[best_checkpoint_idx]:.2e} M")

print("\n" + "="*80)
print("EVOLUTION TRACKING COMPLETE")
print("="*80)
print(f"\nFiles in evolution/ directory:")
print(f"  - {len(checkpoint_steps)} profile CSVs")
print(f"  - 1 evolution metrics CSV")
print(f"  - 1 evolution plot PNG")
print("\nUse these to:")
print("  • Identify optimal training duration")
print("  • Detect overfitting (metrics degrading)")
print("  • Visualize profile development")
print("  • Compare early vs late training")
print("="*80)

