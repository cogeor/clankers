# Task: MPC Performance + Walking Smoothness

## Problem 1: Simulation Performance
The MPC viz runs slowly. Need to profile and find bottlenecks.
Candidates: MPC solve time, physics substeps, debug build overhead.

## Problem 2: Walking Smoothness
Robot stumbles/hesitates during trot and walk gaits. Need quantitative
metric (cycle-to-cycle error) and root cause analysis.

## Approach
1. Profile MPC solve times, identify if solver or physics is the bottleneck
2. Define gait cycle repeatability metric
3. Run headless simulation, measure cycle-to-cycle body state error
4. Identify root causes and fix
