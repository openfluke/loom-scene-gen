# Simple Gen - Autoregressive 3D Scene Generator

Neural network-based scene generator that learns to reconstruct 3D block structures from training data using an autoregressive approach.

## Overview

This project trains a shallow neural network to memorize and reproduce 3D scenes by generating blocks sequentially (autoregressively). Given a scene name like "table" or "archway", it generates the exact sequence of blocks with position, rotation, and dimensions.

**Architecture**: Teacher-forcing autoregressive model

- Input: `[scene_one_hot (13), prev_block_params (9)]` → 22 dimensions
- Output: `[x, y, z, rotX, rotY, rotZ, width, height, depth]` → 9 dimensions
- Network: 22 → 512 → 128 → 9 (3 layers, leaky_relu → leaky_relu → linear)
- Training: 500 epochs with 10x data repetition for strong memorization

## Features

- ✅ **Interactive Mode**: Type scene names to generate JSON instantly
- ✅ **Model Persistence**: Trains once, saves model, reloads on restart
- ✅ **Autoregressive Generation**: Generates blocks one-by-one using previous output as input
- ✅ **Teacher Forcing**: Trains on ground truth sequences for stable learning
- ✅ **Full Scene Export**: Includes camera, lighting, and materials for immediate 3D viewing
- ✅ **Shallow Architecture**: 3-layer network avoids vanishing gradients (gradients flow properly!)

## Quick Start

### 1. Train and Generate

```bash
cd examples/simple_gen
go run .
```

First run:

- Loads 13 training scenes from `./scenes/*.json`
- Trains network for 500 epochs (~30 seconds)
- Saves model to `./saved_model.json`
- Enters interactive mode

Subsequent runs:

- Loads saved model instantly
- Ready to generate immediately

### 2. Interactive Commands

```
>> table
Generating 'table' (6 blocks)...
✓ Generated 6 blocks
✓ Saved to: output/table_generated.json

>> archway
Generating 'archway' (8 blocks)...
✓ Generated 8 blocks
✓ Saved to: output/archway_generated.json

>> test
Testing all 13 training scenes...

>> retrain
Deleting saved model and retraining...

>> quit
Goodbye!
```

## Training Data

The model trains on 13 scenes located in `./scenes/`:

| Scene        | Blocks | Description                      |
| ------------ | ------ | -------------------------------- |
| archway      | 8      | Two pillars with horizontal beam |
| bridge       | 12     | Multi-block bridge structure     |
| chair        | 6      | Simple chair with legs and back  |
| domino       | 11     | Line of domino pieces            |
| maze         | 12     | Maze walls and corridors         |
| pyramid      | 6      | Stacked pyramid structure        |
| ramp         | 7      | Inclined ramp blocks             |
| seesaw       | 4      | Seesaw with fulcrum              |
| shelf        | 9      | Multi-level shelf                |
| simple_house | 6      | Basic house structure            |
| stair_case   | 9      | Ascending staircase              |
| table        | 6      | Table top with 4 legs            |
| tower        | 10     | Vertical tower structure         |

Total: **106 training examples** (with 10x repetition = 1,060 samples)

## How It Works

### 1. Encoding

Each scene is encoded as a **one-hot vector** (13 dimensions):

```
"table" → [0,0,0,0,0,0,0,0,0,0,0,1,0]
"archway" → [1,0,0,0,0,0,0,0,0,0,0,0,0]
```

### 2. Autoregressive Generation

Blocks are generated **sequentially** using the previous block as context:

```
Input:  [scene_one_hot, START_TOKEN]
Output: [x₁, y₁, z₁, rot₁, w₁, h₁, d₁]

Input:  [scene_one_hot, block₁_params]
Output: [x₂, y₂, z₂, rot₂, w₂, h₂, d₂]

...repeat until reaching known block count
```

Start token: `[0, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2]`

### 3. Teacher Forcing

During training, the network receives **ground truth previous blocks** (not its own predictions) to stabilize learning:

```go
for i, block := range scene.Blocks {
    prevBlock := startToken
    if i > 0 {
        prevBlock = blockToParams(scene.Blocks[i-1])  // Ground truth!
    }

    input := [scene_one_hot, prevBlock]
    target := blockToParams(block)

    batches.append({Input: input, Target: target})
}
```

### 4. Network Architecture

**Why Shallow?**
Deep networks (8+ layers) suffer from vanishing gradients even with proper initialization. Our 3-layer design ensures gradients flow:

```
Layer 0: Dense 22→512   (leaky_relu)  ← gradients: ~0.01
Layer 1: Dense 512→128  (leaky_relu)  ← gradients: ~0.1
Layer 2: Dense 128→9    (linear)      ← gradients: ~1.0
```

**Parameter Ranges**:

- Position (x, y, z): Clamped to [-2.0, 2.0], scaled by 10.0
- Rotation (rotX, rotY, rotZ): Clamped to [-1.0, 1.0], scaled by π
- Dimensions (w, h, d): Clamped to [0.02, 2.0], scaled by 5.0

## Output Format

Generated JSON files include:

- ✅ Scene metadata (camera position, FOV, orbit controls)
- ✅ Lighting (ambient + directional with shadows)
- ✅ Generated blocks with varied colors
- ✅ Material properties (metalness, roughness)
- ✅ Physics properties (castShadow, receiveShadow)

Example structure:

```json
[
  {
    "type": "scene",
    "background": 1711910,
    "camera": {
      "position": [6, 6, 8],
      "lookAt": [0, 1, 0],
      "fov": 60
    }
  },
  {
    "type": "light",
    "lightType": "ambient",
    "intensity": 0.6
  },
  {
    "type": "light",
    "lightType": "directional",
    "intensity": 0.8,
    "castShadow": true
  },
  {
    "type": "mesh",
    "name": "Generated_Block_1",
    "pos": [0, 1.99, 0],
    "euler": [0, 0, 0],
    "shape": {
      "type": "box",
      "width": 2.94,
      "height": 0.16,
      "depth": 1.98
    },
    "material": {
      "color": 16739179,
      "metalness": 0.1,
      "roughness": 0.7
    }
  }
]
```

## Training Results

Target loss: **< 0.05** for good memorization

Typical training progression:

```
Epoch 10:  Loss ~0.15  (learning structure)
Epoch 50:  Loss ~0.08  (memorizing patterns)
Epoch 100: Loss ~0.04  (good reproduction)
Epoch 200: Loss ~0.02  (near-perfect)
Epoch 500: Loss ~0.01  (excellent memorization)
```

**Success metrics**:

- Loss < 0.05: ✅ Good memorization, scenes reproduce well
- Loss < 0.10: ⚠️ Partial memorization, approximate scenes
- Loss > 0.10: ❌ Poor memorization, needs more training

## Architecture Evolution

This implementation is the result of extensive experimentation:

### ❌ Failed Approaches

1. **Deep networks (8 layers)**: Vanishing gradients killed learning
2. **One-hot → Dense**: Broken weight initialization, signal died
3. **64D embeddings with scale=5.0**: Stronger input didn't fix gradient flow
4. **tanh early layers**: Still couldn't backprop through 8 layers
5. **ReLU activations**: Gradient death in early layers

### ✅ What Worked

1. **Shallow architecture (3 layers)**: Gradients flow properly
2. **leaky_relu**: Prevents dead neurons, allows negative gradients
3. **Direct one-hot input**: No embedding complexity needed
4. **Autoregressive generation**: Natural sequential structure
5. **Teacher forcing**: Stable training with ground truth
6. **10x data repetition**: Strong memorization on small dataset
7. **Aggressive clamping**: Prevents output explosion

## Project Structure

```
simple_gen/
├── main.go                 # Main program with training & generation
├── network_config.json     # 3-layer network architecture
├── saved_model.json        # Trained model weights (auto-generated)
├── scenes/                 # Training data (13 scenes)
│   ├── table.json
│   ├── archway.json
│   └── ...
├── output/                 # Generated scenes (auto-created)
│   ├── table_generated.json
│   ├── archway_generated.json
│   └── ...
└── README.md              # This file
```

## Configuration

Edit `network_config.json` to modify architecture:

```json
{
  "id": "memorization_ultra_shallow",
  "batch_size": 1,
  "layers_per_cell": 3,
  "seed": 42,
  "layers": [
    {
      "type": "dense",
      "activation": "leaky_relu",
      "input_size": 22,
      "output_size": 512
    },
    {
      "type": "dense",
      "activation": "leaky_relu",
      "input_size": 512,
      "output_size": 128
    },
    {
      "type": "dense",
      "activation": "linear",
      "input_size": 128,
      "output_size": 9
    }
  ]
}
```

Training parameters in `main.go`:

```go
config := &nn.TrainingConfig{
    Epochs:          500,        // Increase for better memorization
    LearningRate:    0.05,       // Higher LR for small network
    LossType:        "mse",
    GradientClip:    1.0,        // Prevent exploding gradients
    Verbose:         true,
}
```

## Troubleshooting

### Model not learning (loss stuck > 0.1)

- Increase epochs to 1000
- Check gradient flow: early layers should have norm > 0.001
- Ensure data repetition is enabled (10x)

### Generated blocks look wrong

- Check loss < 0.05 before generating
- Verify clamping ranges match training data
- Ensure scene name is in training vocabulary

### Out of range errors

- Check network input size matches: `13 (scene) + 9 (prev_block) = 22`
- Verify output size is 9 (pos×3, rot×3, dims×3)

## Dependencies

- Go 1.19+
- github.com/openfluke/loom/nn

## Future Improvements

- [ ] Add variational autoencoder (VAE) for novel scene generation
- [ ] Implement diffusion model for higher quality
- [ ] Add stop token for variable-length sequences
- [ ] Support custom scene names (generalization)
- [ ] Multi-scale architecture for larger scenes
- [ ] Add noise injection for controlled variation
- [ ] Export to other 3D formats (glTF, OBJ)

## Credits

Built with [Loom](https://github.com/openfluke/loom) - Go neural network library

## License

Apache2
