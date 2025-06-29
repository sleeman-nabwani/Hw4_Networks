# Multi-Server Discrete Event Simulator

A discrete event simulator for multi-server queueing systems with probabilistic load balancing.

## What it does

Simulates a system with multiple servers where:
- Customers arrive following a Poisson process
- Each customer is routed to a server based on configured probabilities
- Each server has its own queue with finite capacity
- Servers process customers with exponential service times
- Customers are rejected if their assigned server's queue is full

## Usage

```bash
./simulator T M P_1 P_2 ... P_M λ Q_1 Q_2 ... Q_M μ_1 μ_2 ... μ_M
```

### Parameters
- **T**: Simulation time
- **M**: Number of servers
- **P_1...P_M**: Routing probabilities (must sum to 1.0)
- **λ**: Arrival rate
- **Q_1...Q_M**: Queue capacity for each server
- **μ_1...μ_M**: Service rate for each server

### Examples

Single server:
```bash
./simulator 5000 1 1.0 200 20 40
```

Two servers with 30%/70% load split:
```bash
./simulator 5000 2 0.3 0.7 100 10 15 25 30
```

## Output

Returns five values: `A B T_end T_w T_s`
- **A**: Customers served
- **B**: Customers rejected
- **T_end**: Last departure time
- **T_w**: Average wait time
- **T_s**: Average service time

## Setup

### Requirements
- Python 3.7+
- NumPy

### Build
```bash
make
```

Or run directly:
```bash
python simulator.py [parameters]
```

## Files
- `simulator.py` - Main simulation code
- `makefile` - Build script
- `simulator` - Executable wrapper script