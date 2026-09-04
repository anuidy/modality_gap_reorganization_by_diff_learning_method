# GCL-like Experiment Placeholder

## Status

Deferred. This document only reserves a location for a possible future
GCL-like control group.

The GCL-like experiment will be discussed only after the currently defined
paired branches have completed training and their results have been audited.

## Current experiment matrix

The current formal matrix remains unchanged at eight runs:

- CLIP: Standard / Count-Matched Mixed
- VISTA: Standard / Count-Matched Mixed
- BEiT-3: Standard / Count-Matched Mixed
- ALBEF: ITC-only / Full ALBEF

This placeholder is not a ninth run and must not be loaded by the training
entry point.

## Intentionally undecided

No decision has been made yet about:

- the exact GCL paper, implementation, or terminology to follow;
- which model families would receive the additional control group;
- whether sample count, relation count, optimizer steps, or compute budget may
  differ from the existing branches;
- the loss definition or loss weighting;
- data construction, augmentation, batch size, learning rate, or schedule;
- checkpoint selection or evaluation comparisons.

Do not infer defaults for these fields and do not implement training code from
this placeholder.

## Activation condition

Before this experiment is activated, create a separate reviewed protocol that
defines the intervention, fairness controls, compute accounting, run matrix,
and configuration values. Only then may executable configuration and training
code be added.
