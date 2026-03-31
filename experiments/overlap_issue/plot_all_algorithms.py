#!/usr/bin/env python3
"""Plot EMOGI zero-copy overhead across all algorithms (BFS, CC, SSSP, PageRank) on orkut-links."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

# ── Figure 1: Per-iteration comparison for each algorithm ──
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('EMOGI Zero-Copy (UVM_DIRECT) Overhead per Algorithm\nDataset: orkut-links (3M vertices, 117M edges, 0.94 GB)', fontsize=14, fontweight='bold')

# BFS
ax = axes[0, 0]
bfs_gpu = pd.read_csv('orkut_gpumem.csv')
bfs_uvm = pd.read_csv('orkut_uvm_direct.csv')
x = bfs_gpu['iter'].values
w = 0.35
ax.bar(x - w/2, bfs_gpu['kernel_ms'], w, label='GPUMEM', color='#2196F3')
ax.bar(x + w/2, bfs_uvm['kernel_ms'], w, label='UVM_DIRECT', color='#F44336')
ax.set_xlabel('Iteration')
ax.set_ylabel('Kernel Time (ms)')
ax.set_title('BFS (root=1895)')
ax.legend()
ax.set_xticks(x)

# CC
ax = axes[0, 1]
cc_gpu = pd.read_csv('cc_profile_gpumem.csv')
cc_uvm = pd.read_csv('cc_profile_uvm_direct.csv')
x = cc_gpu['iter'].values
ax.bar(x - w/2, cc_gpu['kernel_ms'], w, label='GPUMEM', color='#2196F3')
ax.bar(x + w/2, cc_uvm['kernel_ms'], w, label='UVM_DIRECT', color='#F44336')
ax.set_xlabel('Iteration')
ax.set_ylabel('Kernel Time (ms)')
ax.set_title('Connected Components')
ax.legend()
ax.set_xticks(x)

# SSSP
ax = axes[1, 0]
sssp_gpu = pd.read_csv('sssp_profile_gpumem.csv')
sssp_uvm = pd.read_csv('sssp_profile_uvm_direct.csv')
x = sssp_gpu['iter'].values
ax.bar(x - w/2, sssp_gpu['kernel_ms'], w, label='GPUMEM', color='#2196F3')
ax.bar(x + w/2, sssp_uvm['kernel_ms'], w, label='UVM_DIRECT', color='#F44336')
ax.set_xlabel('Iteration')
ax.set_ylabel('Kernel Time (ms)')
ax.set_title('SSSP (root=1895)')
ax.legend()
ax.set_xticks(x)

# PageRank
ax = axes[1, 1]
pr_gpu = pd.read_csv('pagerank_profile_gpumem.csv')
pr_uvm = pd.read_csv('pagerank_profile_uvm_direct.csv')
x = pr_gpu['iter'].values
ax.bar(x - w/2, pr_gpu['kernel_ms'], w, label='GPUMEM', color='#2196F3', alpha=0.8)
ax.bar(x + w/2, pr_uvm['kernel_ms'], w, label='UVM_DIRECT', color='#F44336', alpha=0.8)
ax.set_xlabel('Iteration')
ax.set_ylabel('Kernel Time (ms)')
ax.set_title('PageRank (α=0.85, tol=0.01)')
ax.legend()

plt.tight_layout()
plt.savefig('all_algorithms_per_iter.png', bbox_inches='tight')
print('Saved: all_algorithms_per_iter.png')

# ── Figure 2: Aggregate comparison bar chart ──
fig2, ax2 = plt.subplots(figsize=(10, 6))

algos = ['BFS', 'CC', 'SSSP', 'PageRank']
gpumem_totals = [
    bfs_gpu['kernel_ms'].sum(),
    cc_gpu['kernel_ms'].sum(),
    sssp_gpu['kernel_ms'].sum(),
    pr_gpu['kernel_ms'].sum(),
]
uvm_totals = [
    bfs_uvm['kernel_ms'].sum(),
    cc_uvm['kernel_ms'].sum(),
    sssp_uvm['kernel_ms'].sum(),
    pr_uvm['kernel_ms'].sum(),
]
slowdowns = [u / g for g, u in zip(gpumem_totals, uvm_totals)]

x = np.arange(len(algos))
w = 0.35
bars1 = ax2.bar(x - w/2, gpumem_totals, w, label='GPUMEM', color='#2196F3')
bars2 = ax2.bar(x + w/2, uvm_totals, w, label='UVM_DIRECT', color='#F44336')

# Add slowdown labels
for i, (b, s) in enumerate(zip(bars2, slowdowns)):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
             f'{s:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.set_xlabel('Algorithm')
ax2.set_ylabel('Total Kernel Time (ms)')
ax2.set_title('EMOGI: Total Kernel Time — GPUMEM vs UVM_DIRECT\n(orkut-links, RTX A4000 16GB)')
ax2.set_xticks(x)
ax2.set_xticklabels(algos)
ax2.legend()
plt.tight_layout()
plt.savefig('all_algorithms_aggregate.png', bbox_inches='tight')
print('Saved: all_algorithms_aggregate.png')

# ── Figure 3: Per-iteration slowdown ratio ──
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))
fig3.suptitle('Per-Iteration Slowdown: UVM_DIRECT / GPUMEM', fontsize=14, fontweight='bold')

datasets = [
    ('BFS', bfs_gpu, bfs_uvm, 'kernel_ms'),
    ('CC', cc_gpu, cc_uvm, 'kernel_ms'),
    ('SSSP', sssp_gpu, sssp_uvm, 'kernel_ms'),
    ('PageRank', pr_gpu, pr_uvm, 'kernel_ms'),
]

for idx, (name, gpu, uvm, col) in enumerate(datasets):
    ax = axes3[idx // 2, idx % 2]
    ratio = uvm[col].values / gpu[col].values
    x = gpu['iter'].values
    colors = ['#F44336' if r > 1.5 else '#FF9800' if r > 1.1 else '#4CAF50' for r in ratio]
    ax.bar(x, ratio, color=colors)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Slowdown (x)')
    ax.set_title(name)
    # Annotate peak
    peak_idx = np.argmax(ratio)
    ax.annotate(f'{ratio[peak_idx]:.2f}x', xy=(x[peak_idx], ratio[peak_idx]),
                xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('all_algorithms_slowdown.png', bbox_inches='tight')
print('Saved: all_algorithms_slowdown.png')

# Print summary
print('\n=== Summary ===')
for name, gtot, utot, sd in zip(algos, gpumem_totals, uvm_totals, slowdowns):
    print(f'{name:10s}: GPUMEM={gtot:8.2f}ms  UVM_DIRECT={utot:8.2f}ms  Slowdown={sd:.2f}x')
