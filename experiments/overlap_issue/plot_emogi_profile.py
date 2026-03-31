#!/usr/bin/env python3
"""
EMOGI BFS Profiling: Zero-Copy Page Fault Bottleneck Analysis
Generates:
  1. orkut-links: GPUMEM vs UVM_DIRECT vs UVM_READONLY per-iteration comparison
  2. uk-2007: UVM_DIRECT per-iteration breakdown showing page fault overhead
"""

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    iters, levels, kernels = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row['iter']))
            levels.append(int(row['level']))
            kernels.append(float(row['kernel_ms']))
    return np.array(iters), np.array(levels), np.array(kernels)


def main():
    # =========================================================================
    # Figure 1: orkut-links — GPUMEM vs UVM_DIRECT vs UVM_READONLY
    # =========================================================================
    gpu_i, gpu_l, gpu_k = load_csv('orkut_gpumem.csv')
    uvm_i, uvm_l, uvm_k = load_csv('orkut_uvm_direct.csv')
    ro_i, ro_l, ro_k = load_csv('orkut_uvm_ro.csv')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('EMOGI BFS: Zero-Copy Page Fault Overhead\n(orkut-links, 3M nodes / 117M edges, source=10, RTX A4000)',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(gpu_i))
    width = 0.25

    # Per-iteration kernel time comparison
    ax1.bar(x - width, gpu_k, width, label='GPUMEM (baseline)', color='#2ecc71', edgecolor='white')
    ax1.bar(x, uvm_k, width, label='UVM_DIRECT (zero-copy)', color='#e74c3c', edgecolor='white')
    ax1.bar(x + width, ro_k, width, label='UVM_READONLY (page migrate)', color='#3498db', edgecolor='white')

    ax1.set_ylabel('Kernel Time (ms)', fontsize=12)
    ax1.set_xlabel('BFS Iteration (Level)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in gpu_i])
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_title('Per-Iteration Kernel Time', fontsize=12)

    # Slowdown ratio
    slowdown_direct = uvm_k / gpu_k
    slowdown_ro = ro_k / gpu_k

    ax2.plot(x, slowdown_direct, 'o-', color='#e74c3c', linewidth=2, markersize=6, label='UVM_DIRECT / GPUMEM')
    ax2.plot(x, slowdown_ro, 's--', color='#3498db', linewidth=2, markersize=6, label='UVM_READONLY / GPUMEM')
    ax2.axhline(y=1.0, color='#2ecc71', linestyle=':', linewidth=1.5, label='Baseline (1.0x)')
    ax2.set_ylabel('Slowdown vs GPUMEM', fontsize=12)
    ax2.set_xlabel('BFS Iteration (Level)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i) for i in gpu_i])
    ax2.legend(fontsize=10)
    ax2.set_title('Slowdown Ratio (higher = more page fault stall)', fontsize=12)
    ax2.set_ylim(0, max(max(slowdown_direct), max(slowdown_ro)) * 1.15)

    # Annotate peak slowdowns
    peak_d = np.argmax(slowdown_direct)
    ax2.annotate(f'{slowdown_direct[peak_d]:.1f}x', xy=(x[peak_d], slowdown_direct[peak_d]),
                 xytext=(x[peak_d]+0.3, slowdown_direct[peak_d]+1),
                 fontsize=10, fontweight='bold', color='#e74c3c',
                 arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    peak_r = np.argmax(slowdown_ro)
    ax2.annotate(f'{slowdown_ro[peak_r]:.1f}x', xy=(x[peak_r], slowdown_ro[peak_r]),
                 xytext=(x[peak_r]+0.3, slowdown_ro[peak_r]+2),
                 fontsize=10, fontweight='bold', color='#3498db',
                 arrowprops=dict(arrowstyle='->', color='#3498db'))

    plt.tight_layout()
    fig.savefig('emogi_orkut_comparison.png', dpi=200, bbox_inches='tight')
    print("Saved: emogi_orkut_comparison.png")

    # =========================================================================
    # Figure 2: uk-2007 UVM_DIRECT — page fault stall dominates
    # =========================================================================
    uk_i, uk_l, uk_k = load_csv('uk2007_uvm_direct.csv')

    # Estimate "pure compute" baseline from the smallest kernel time (tail iterations)
    # where no edges are accessed (frontier is empty)
    baseline_ms = np.min(uk_k)  # ~35.9 ms (just scanning 105M labels)
    stall_ms = uk_k - baseline_ms
    stall_ms = np.maximum(stall_ms, 0)

    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(16, 10))
    fig2.suptitle('EMOGI BFS UVM_DIRECT: Page Fault Stall on uk-2007\n'
                  '(105M nodes / 3.3B edges, 26.4 GB edge array, source=10, RTX A4000)',
                  fontsize=14, fontweight='bold')

    x2 = np.arange(len(uk_i))
    width2 = 0.6

    ax3.bar(x2, np.full(len(uk_i), baseline_ms), width2, label=f'Label scan baseline (~{baseline_ms:.1f} ms)',
            color='#2ecc71', edgecolor='white', linewidth=0.3)
    ax3.bar(x2, stall_ms, width2, bottom=baseline_ms, label='Page fault stall (on-demand transfer)',
            color='#e74c3c', edgecolor='white', linewidth=0.3)

    ax3.set_ylabel('Kernel Time (ms)', fontsize=12)
    ax3.set_xlabel('BFS Iteration', fontsize=12)
    step = max(1, len(uk_i) // 20)
    ax3.set_xticks(x2[::step])
    ax3.set_xticklabels(uk_i[::step])
    ax3.legend(fontsize=10, loc='upper right')
    ax3.set_title('Per-Iteration Kernel Time Decomposition', fontsize=12)

    # Percentage of time that is page fault stall
    stall_pct = stall_ms / uk_k * 100
    ax4.bar(x2, stall_pct, width2, color='#e74c3c', edgecolor='white', linewidth=0.3)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_ylabel('Page Fault Stall %', fontsize=12)
    ax4.set_xlabel('BFS Iteration', fontsize=12)
    ax4.set_xticks(x2[::step])
    ax4.set_xticklabels(uk_i[::step])
    ax4.set_title('Fraction of Kernel Time Spent on Page Fault Stalls', fontsize=12)
    ax4.set_ylim(0, 100)

    plt.tight_layout()
    fig2.savefig('emogi_uk2007_uvm_direct.png', dpi=200, bbox_inches='tight')
    print("Saved: emogi_uk2007_uvm_direct.png")

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY: EMOGI BFS Zero-Copy Bottleneck Analysis")
    print("="*80)

    print("\n--- orkut-links (fits in GPU, 0.94 GB edge array) ---")
    print(f"  GPUMEM total kernel:     {gpu_k.sum():.2f} ms")
    print(f"  UVM_DIRECT total kernel: {uvm_k.sum():.2f} ms (slowdown: {uvm_k.sum()/gpu_k.sum():.2f}x)")
    print(f"  UVM_READONLY total kernel:{ro_k.sum():.2f} ms (slowdown: {ro_k.sum()/gpu_k.sum():.2f}x)")
    print(f"  Peak iteration slowdown (UVM_DIRECT): iter {gpu_i[peak_d]}, {slowdown_direct[peak_d]:.2f}x")
    print(f"  Page fault stall time (UVM_DIRECT, total): {(uvm_k.sum()-gpu_k.sum()):.2f} ms")

    print(f"\n--- uk-2007 (26.4 GB edge array, UVM_DIRECT only) ---")
    print(f"  Total kernel time:       {uk_k.sum():.2f} ms")
    print(f"  Label scan baseline:     {baseline_ms:.2f} ms/iter x {len(uk_i)} = {baseline_ms*len(uk_i):.2f} ms")
    print(f"  Page fault stall total:  {stall_ms.sum():.2f} ms ({stall_ms.sum()/uk_k.sum()*100:.1f}%)")
    print(f"  Peak stall iteration:    iter {uk_i[np.argmax(stall_ms)]} ({uk_k[np.argmax(stall_ms)]:.2f} ms, stall {stall_ms.max():.2f} ms)")


if __name__ == '__main__':
    main()
