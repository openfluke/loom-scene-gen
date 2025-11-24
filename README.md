Simple Gen - Autoregressive 3D Scene GeneratorNeural network-based scene generator that learns to reconstruct 3D block structures from training data using an autoregressive approach.OverviewThis project trains a state-of-the-art sequence model (LSTM + Dense) to memorize and reproduce 3D scenes by generating blocks sequentially (autoregressively). Given a scene name like "table" or "archway", it generates the exact sequence of blocks with position, rotation, and dimensions.Architecture: Teacher-forcing autoregressive model designed for stability and context.Input: [scene_one_hot (13), prev_block_params (9)] → 22 dimensionsOutput: [x, y, z, rotX, rotY, rotZ, width, height, depth] → 9 dimensionsNetwork Stack: Input Encoder (Dense) → LSTM Memory Core → Output Head (Dense)✅ New Core Features (Stepping & Stability)FeatureDescriptionLSTM Memory CoreThe network now uses a Long Short-Term Memory (LSTM) layer to capture long-range dependencies in block sequences (e.g., remembering the start position of a tower's foundation) .AdamW OptimizerSwitched from simple SGD to the highly effective AdamW optimizer for faster, more stable weight updates, and customized for aggressive memorization ($\beta_1=0.95$, Weight Decay $= 0.0$).Cosine Annealing w/ WarmupImplements a Learning Rate Scheduler that starts with a gradual Warmup phase to prevent gradient explosion and then decays the LR using Cosine Annealing for precise weight convergence (required for LSTM stability).Gradient ClippingA safety mechanism ($\mathbf{5.0}$ norm) is applied after the backward pass to explicitly prevent exploding gradients, which plagued previous runs.Stepping APIUses the nn.StepForward and nn.StepBackward methods to train samples one-by-one, which is the necessary pattern for training recurrent (LSTM) models.Quick Start1. Train and Generatecd examples/simple_gen
go run .
First run:Loads 13 training scenes from ./scenes/*.jsonTrains network for 250,000 steps (longer due to stability requirements)Saves model to ./saved_model.jsonEnters interactive modeSubsequent runs:Loads saved model instantlyReady to generate immediately2. Interactive Commands>> table
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
Training DataThe model trains on 13 scenes located in ./scenes/:SceneBlocksarchway8bridge12chair6domino11maze12pyramid6ramp7seesaw4shelf9simple_house6stair_case9table6tower10Total: 106 unique training examples, amplified 20x for a total of 2,120 samples per epoch to ensure aggressive memorization.How It Works1. Model ArchitectureThe network uses a 3-layer recurrent architecture defined in network_config.json:Dense Layer (encoder): Maps the 22-dimensional input (Scene ID + Previous Block) to the 64-dimensional hidden space.LSTM Layer (memory): The core of the model. It maintains a cell state and hidden state to learn the cumulative context and order of blocks in a scene (e.g., "After I placed Block 1 for the chair seat, Block 2 must be a leg 2 units below").Dense Layer (output_head): Maps the 64-dimensional hidden state to the 9-dimensional output (the predicted next block parameters).2. Autoregressive GenerationBlocks are generated sequentially using the context:Input:  [scene_one_hot, START_TOKEN]
Output: [Block 1 params]

Input:  [scene_one_hot, Block 1 params]
Output: [Block 2 params]

...repeat until reaching known block count
3. Training Strategy (Stability Focus)The training loop focuses on overcoming LSTM instability:Warmup (20,000 steps): The learning rate starts near zero and slowly ramps up to the peak $\mathbf{0.0005}$. This prevents the initial explosive updates that ruin LSTMs.Cosine Annealing (230,000 steps): After warmup, the learning rate smoothly decays towards zero. This allows the model to fine-tune weights precisely for low-loss memorization.Gradient Clipping: The $\mathbf{5.0}$ clip value automatically scales down any extreme gradients during BPTT, ensuring stable training continuation.Output FormatGenerated JSON files include:✅ Scene metadata (camera position, FOV, orbit controls)✅ Lighting (ambient + directional with shadows)✅ Generated blocks with varied colors✅ Material and Physics properties.