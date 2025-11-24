package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	nn "github.com/openfluke/loom/nn"
)

const modelPath = "./saved_model.json"
const configPath = "./network_config.json"

// --- Data Structures ---

// SceneObject represents a generic object in the scene
type SceneObject map[string]interface{}

// Block represents a mesh/block with full parameters
type Block struct {
	Name     string
	Position [3]float64
	Rotation [3]float64
	Width    float64
	Height   float64
	Depth    float64
}

// Scene holds the name and list of blocks for a scene
type Scene struct {
	Name    string
	SceneID int
	Objects []SceneObject
	Blocks  []Block
}

// BPE vocabulary for scene types
var sceneVocab = map[string]int{
	"archway":      0,
	"bridge":       1,
	"chair":        2,
	"domino":       3,
	"maze":         4,
	"pyramid":      5,
	"ramp":         6,
	"seesaw":       7,
	"shelf":        8,
	"simple_house": 9,
	"stair_case":   10,
	"table":        11,
	"tower":        12,
}

const (
	vocabSize      = 13
	blockParamSize = 9
	// Input size: 13 (scene one-hot) + 9 (prev block params) = 22
	inputVectorSize = vocabSize + blockParamSize
	// NOTE: We define these as variables now so the training function can use them
	learningRate = 0.001
	totalSteps   = 20000
	warmupSteps  = 10000 // 10k steps for LR ramp-up
)

func main() {
	fmt.Println("=== Autoregressive Scene Generator (AdamW Stepping) ===")
	fmt.Println("Interactive mode: type scene names to generate")
	fmt.Println()

	scenes, err := loadAllScenes("./scenes")
	if err != nil {
		log.Fatalf("Error loading scenes: %v", err)
	}

	fmt.Printf("Loaded %d training scenes\n", len(scenes))
	for i := range scenes {
		scenes[i].SceneID = sceneVocab[scenes[i].Name]
		extractBlocksFromScene(&scenes[i])
		fmt.Printf("  [%d] %s: %d blocks\n", scenes[i].SceneID, scenes[i].Name, len(scenes[i].Blocks))
	}

	var network *nn.Network

	if _, err := os.Stat(modelPath); err == nil {
		fmt.Printf("\n✓ Found saved model at %s\n", modelPath)
		fmt.Println("Loading model...")
		network, err = nn.LoadModel(modelPath, "memorization_model")
		if err != nil {
			log.Fatalf("Error loading model: %v", err)
		}
		fmt.Println("✓ Model loaded successfully!")
	} else {
		fmt.Println("\n✗ No saved model found. Training new model...")

		network, err = nn.BuildNetworkFromFile(configPath)
		if err != nil {
			log.Fatalf("Error loading network config: %v", err)
		}
		// Set the correct input size in the network's config if it wasn't already
		if netCfg := network.GetLayer(0, 0, 0); netCfg != nil {
			netCfg.InputHeight = inputVectorSize
			network.SetLayer(0, 0, 0, *netCfg)
		}

		network.InitializeWeights()
		fmt.Printf("Network: %d total layers\n", network.TotalLayers())

		trainForMemorization(network, scenes)

		fmt.Printf("\nSaving model to %s...\n", modelPath)
		if err := network.SaveModel(modelPath, "memorization_model"); err != nil {
			log.Printf("Warning: Failed to save model: %v", err)
		} else {
			fmt.Println("✓ Model saved!")
		}
	}

	// Interactive mode
	fmt.Println("\n=== Interactive Generation Mode ===")
	// ... (interactive mode logic remains the same)

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print(">> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(strings.ToLower(scanner.Text()))
		if input == "" {
			continue
		}

		switch input {
		case "quit", "exit", "q":
			fmt.Println("Goodbye!")
			return

		case "retrain":
			fmt.Println("\nDeleting saved model and retraining...")
			os.Remove(modelPath)

			network, err = nn.BuildNetworkFromFile(configPath)
			if err != nil {
				log.Fatalf("Error loading network config: %v", err)
			}
			network.InitializeWeights()

			trainForMemorization(network, scenes)

			if err := network.SaveModel(modelPath, "memorization_model"); err != nil {
				log.Printf("Warning: Failed to save model: %v", err)
			} else {
				fmt.Println("✓ Model retrained and saved!")
			}

		case "test":
			fmt.Println("\nTesting all training scenes...")
			testAllScenes(network, scenes)

		default:
			if _, ok := sceneVocab[input]; !ok {
				fmt.Printf("❌ Unknown scene '%s'. Available: %s\n", input, getSceneNames())
				continue
			}

			var numBlocks int
			for _, scene := range scenes {
				if scene.Name == input {
					numBlocks = len(scene.Blocks)
					break
				}
			}

			if numBlocks == 0 {
				fmt.Printf("❌ Scene '%s' has no blocks\n", input)
				continue
			}

			fmt.Printf("\nGenerating '%s' (%d blocks)...\n", input, numBlocks)
			generatedBlocks := generateScene(network, input, numBlocks)

			outputScene := Scene{
				Name:   input,
				Blocks: generatedBlocks,
			}
			outputPath := filepath.Join("./output", input+"_generated.json")
			if err := saveSceneWithBlocks(&outputScene, outputPath); err != nil {
				fmt.Printf("❌ Error saving: %v\n", err)
			} else {
				fmt.Printf("✓ Generated %d blocks\n", len(generatedBlocks))
				fmt.Printf("✓ Saved to: %s\n", outputPath)

				for i := 0; i < 3 && i < len(generatedBlocks); i++ {
					b := generatedBlocks[i]
					fmt.Printf("  Block %d: pos=(%.2f, %.2f, %.2f) size=(%.2f, %.2f, %.2f)\n",
						i+1, b.Position[0], b.Position[1], b.Position[2],
						b.Width, b.Height, b.Depth)
				}
			}
		}
		fmt.Println()
	}
}

// --- Training and Generation Logic ---

type TrainingExample struct {
	Input  []float32
	Target []float32
}

// trainForMemorization uses AdamW and Step API to aggressively overfit
func trainForMemorization(network *nn.Network, scenes []Scene) {
	var examples []TrainingExample

	startToken := []float32{0, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2}

	// 1. Build training data (teacher forcing)
	for repeat := 0; repeat < 20; repeat++ { // Increased repetition for better memorization
		for _, scene := range scenes {
			if len(scene.Blocks) == 0 {
				continue
			}

			sceneEmbed := encodeSceneType(scene.Name)

			for i := 0; i < len(scene.Blocks); i++ {
				var prevBlock []float32
				if i == 0 {
					prevBlock = startToken
				} else {
					prevBlock = blockToParams(scene.Blocks[i-1])
				}

				currentBlock := blockToParams(scene.Blocks[i])

				examples = append(examples, TrainingExample{
					Input:  append(sceneEmbed, prevBlock...),
					Target: currentBlock,
				})
			}
		}
	}

	// 2. Setup CUSTOM AdamW Optimizer for Overfitting and a Scheduler for stability
	customOptimizer := nn.NewAdamWOptimizer(
		0.95,  // Beta1 (Momentum) - High momentum for aggressive updates
		0.999, // Beta2 (Variance Decay) - Default, but could be 0.9999 for ultra-smooth
		1e-8,  // Epsilon (Stability)
		0.0,   // WeightDecay - Set to 0.0 for pure memorization (no regularization)
	)
	network.SetOptimizer(customOptimizer)

	// Setup Cosine Annealing Scheduler with Warmup
	// We wrap Cosine Annealing with Warmup
	cosineScheduler := nn.NewCosineAnnealingScheduler(learningRate, 0.00001, totalSteps-warmupSteps)
	scheduler := nn.NewWarmupScheduler(warmupSteps, 0.00001, learningRate, cosineScheduler)

	// 3. Setup Stepping State
	state := network.InitStepState(inputVectorSize)

	fmt.Printf("Total training examples: %d (with 20x repetition)\n", len(examples))
	fmt.Printf("Input size: %d (scene: %d + prev_block: %d)\n",
		inputVectorSize, vocabSize, blockParamSize)
	fmt.Printf("Training %d steps with LR=%.4f (Warmup + Cosine Annealing)...\n", totalSteps, learningRate)
	fmt.Println("Target: Loss < 0.01 for good reproduction")
	fmt.Println("----------------------------------------------------------------")

	var avgLoss float32
	start := time.Now()

	for step := 0; step < totalSteps; step++ {
		// Get LR from scheduler
		currentLR := scheduler.GetLR(step)

		idx := step % len(examples)
		example := examples[idx]

		// A. Forward Pass
		state.SetInput(example.Input)
		network.StepForward(state)
		output := state.GetOutput()

		// B. Compute Loss (MSE) and Gradients
		loss := float32(0.0)
		gradOutput := make([]float32, len(output))

		for i := range output {
			diff := output[i] - example.Target[i]
			loss += diff * diff
			// MSE gradient: 2 * (output - target) / output_size
			gradOutput[i] = 2.0 * diff / float32(len(output))
		}
		avgLoss += loss

		// C. Backward Pass
		network.StepBackward(state, gradOutput)

		// D. Update Weights using AdamW (via the set optimizer)
		network.ApplyGradients(currentLR)

		// E. Logging
		if (step+1)%5000 == 0 {
			finalLoss := avgLoss / 5000.0
			avgLoss = 0.0

			// Find prediction accuracy by comparing first position param (x-pos)
			origX := example.Target[0] * 10.0
			predX := output[0] * 10.0
			error := math.Abs(float64(origX - predX))

			fmt.Printf("Step %6d | LR: %.6f | Avg Loss: %.6f | Last X Error: %.3f\n",
				step+1, currentLR, finalLoss, error)
		}
	}

	finalLoss := avgLoss / float32(totalSteps%5000)
	fmt.Printf("\n✓ Training complete in %v. Final Loss: %.6f\n", time.Since(start), finalLoss)
}

// generateScene generates blocks autoregressively
func generateScene(network *nn.Network, sceneName string, numBlocks int) []Block {
	sceneEmbed := encodeSceneType(sceneName)
	blocks := []Block{}

	prevBlock := []float32{0, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2} // Start token

	state := network.InitStepState(inputVectorSize)

	// Generate EXACTLY numBlocks (no stop token)
	for i := 0; i < numBlocks; i++ {
		input := append(sceneEmbed, prevBlock...)

		// 1. Forward Pass (using Step API)
		state.SetInput(input)
		network.StepForward(state)
		output := state.GetOutput() // Output is the next block's parameters

		// 2. Clamp to reasonable ranges
		for j := 0; j < 3; j++ {
			output[j] = clamp(output[j], -2.0, 2.0)
		}
		for j := 3; j < 6; j++ {
			output[j] = clamp(output[j], -1.0, 1.0)
		}
		for j := 6; j < 9; j++ {
			output[j] = clamp(output[j], 0.02, 2.0)
		}

		block := paramsToBlock(output, i)
		blocks = append(blocks, block)

		// Set current output as next input
		prevBlock = output
	}

	return blocks
}

// --- Helper Functions (Unchanged) ---

func getSceneNames() string {
	names := make([]string, 0, len(sceneVocab))
	for name := range sceneVocab {
		names = append(names, name)
	}
	return strings.Join(names, ", ")
}

func testAllScenes(network *nn.Network, scenes []Scene) {
	for _, scene := range scenes {
		fmt.Printf("\nScene: %s (original %d blocks)\n", scene.Name, len(scene.Blocks))
		generatedBlocks := generateScene(network, scene.Name, len(scene.Blocks))

		compareLimit := len(scene.Blocks)
		if compareLimit > len(generatedBlocks) {
			compareLimit = len(generatedBlocks)
		}

		totalError := 0.0
		for i := 0; i < compareLimit; i++ {
			orig := scene.Blocks[i]
			gen := generatedBlocks[i]

			posError := math.Sqrt(
				math.Pow(orig.Position[0]-gen.Position[0], 2) +
					math.Pow(orig.Position[1]-gen.Position[1], 2) +
					math.Pow(orig.Position[2]-gen.Position[2], 2))

			totalError += posError

			if i < 2 {
				fmt.Printf("  Block %d: pos=(%.2f,%.2f,%.2f) vs (%.2f,%.2f,%.2f) err=%.3f\n",
					i+1,
					orig.Position[0], orig.Position[1], orig.Position[2],
					gen.Position[0], gen.Position[1], gen.Position[2],
					posError)
			}
		}

		avgError := totalError / float64(compareLimit)
		fmt.Printf("  Average position error: %.3f\n", avgError)

		outputScene := scene
		outputScene.Blocks = generatedBlocks
		outputPath := filepath.Join("./output", scene.Name+"_test.json")
		if err := saveSceneWithBlocks(&outputScene, outputPath); err != nil {
			log.Printf("  Error saving: %v", err)
		} else {
			fmt.Printf("  ✓ Saved to: %s\n", outputPath)
		}
	}
}

// encodeSceneType creates one-hot encoding for scene type
func encodeSceneType(sceneName string) []float32 {
	oneHot := make([]float32, vocabSize)
	if idx, ok := sceneVocab[sceneName]; ok {
		oneHot[idx] = 1.0
	}
	return oneHot
}

// blockToParams extracts 9 parameters from a Block (Normalization is key!)
func blockToParams(block Block) []float32 {
	return []float32{
		// Position is usually large, scale by 10.0
		float32(block.Position[0]) / 10.0,
		float32(block.Position[1]) / 10.0,
		float32(block.Position[2]) / 10.0,
		// Rotation is -pi to pi, normalize by pi
		float32(block.Rotation[0]) / math.Pi,
		float32(block.Rotation[1]) / math.Pi,
		float32(block.Rotation[2]) / math.Pi,
		// Dimensions are usually small, scale by 5.0
		float32(block.Width) / 5.0,
		float32(block.Height) / 5.0,
		float32(block.Depth) / 5.0,
	}
}

// paramsToBlock converts 9 parameters back to a Block (Inverse normalization)
func paramsToBlock(params []float32, index int) Block {
	return Block{
		Name: fmt.Sprintf("Generated_Block_%d", index+1),
		Position: [3]float64{
			float64(params[0]) * 10.0,
			float64(params[1]) * 10.0,
			float64(params[2]) * 10.0,
		},
		Rotation: [3]float64{
			float64(params[3]) * math.Pi,
			float64(params[4]) * math.Pi,
			float64(params[5]) * math.Pi,
		},
		Width:  float64(params[6]) * 5.0,
		Height: float64(params[7]) * 5.0,
		Depth:  float64(params[8]) * 5.0,
	}
}

// clamp restricts value to range [min, max]
func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// loadAllScenes loads all .json files from a directory (Same as original)
func loadAllScenes(dir string) ([]Scene, error) {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var scenes []Scene
	for _, file := range files {
		if filepath.Ext(file.Name()) == ".json" {
			path := filepath.Join(dir, file.Name())
			data, err := ioutil.ReadFile(path)
			if err != nil {
				log.Printf("Warning: Failed to read %s: %v", path, err)
				continue
			}

			var objects []SceneObject
			if err := json.Unmarshal(data, &objects); err != nil {
				log.Printf("Warning: Failed to parse %s: %v", path, err)
				continue
			}

			sceneName := strings.TrimSuffix(file.Name(), ".json")
			scene := Scene{
				Name:    sceneName,
				Objects: objects,
				Blocks:  []Block{},
			}

			scenes = append(scenes, scene)
		}
	}

	return scenes, nil
}

// normalizeRotation forces rotation to [-π, π]
func normalizeRotation(r float64) float64 {
	r = math.Mod(r+math.Pi, 2*math.Pi) - math.Pi
	return r
}

// extractBlocksFromScene extracts blocks/meshes from scene objects (Same as original)
func extractBlocksFromScene(scene *Scene) {
	for _, obj := range scene.Objects {
		objType, ok := obj["type"].(string)
		if !ok || objType != "mesh" {
			continue
		}

		name, _ := obj["name"].(string)
		if strings.Contains(strings.ToLower(name), "ground") ||
			strings.Contains(strings.ToLower(name), "floor") {
			continue
		}

		block := Block{Name: name}

		if pos, ok := obj["pos"].([]interface{}); ok && len(pos) >= 3 {
			block.Position[0], _ = pos[0].(float64)
			block.Position[1], _ = pos[1].(float64)
			block.Position[2], _ = pos[2].(float64)
		}

		if euler, ok := obj["euler"].([]interface{}); ok && len(euler) >= 3 {
			block.Rotation[0], _ = euler[0].(float64)
			block.Rotation[1], _ = euler[1].(float64)
			block.Rotation[2], _ = euler[2].(float64)
		}

		block.Rotation[0] = normalizeRotation(block.Rotation[0])
		block.Rotation[1] = normalizeRotation(block.Rotation[1])
		block.Rotation[2] = normalizeRotation(block.Rotation[2])

		if shape, ok := obj["shape"].(map[string]interface{}); ok {
			block.Width, _ = shape["width"].(float64)
			block.Height, _ = shape["height"].(float64)
			block.Depth, _ = shape["depth"].(float64)

			if radius, ok := shape["radius"].(float64); ok {
				block.Width = radius * 2
				block.Height = radius * 2
				block.Depth = radius * 2
			}
		}

		if block.Width == 0 {
			block.Width = 1.0
		}
		if block.Height == 0 {
			block.Height = 1.0
		}
		if block.Depth == 0 {
			block.Depth = 1.0
		}

		scene.Blocks = append(scene.Blocks, block)
	}
}

// saveSceneWithBlocks saves scene to JSON (Same as original)
func saveSceneWithBlocks(scene *Scene, outputPath string) error {
	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	var allObjects []SceneObject

	for _, obj := range scene.Objects {
		objType, _ := obj["type"].(string)
		if objType != "mesh" {
			allObjects = append(allObjects, obj)
		}
	}

	hasScene := false
	for _, obj := range allObjects {
		if t, ok := obj["type"].(string); ok && t == "scene" {
			hasScene = true
			break
		}
	}
	if !hasScene {
		allObjects = append([]SceneObject{
			{
				"type":       "scene",
				"background": 0x1A1F26,
				"fog": map[string]interface{}{
					"color": 0x1A1F26,
					"near":  0.5,
					"far":   150.0,
				},
				"camera": map[string]interface{}{
					"position":     []float64{6, 6, 8},
					"lookAt":       []float64{0, 1, 0},
					"orbitEnabled": true,
					"orbitTarget":  []float64{0, 1, 0},
					"fov":          60.0,
					"near":         0.1,
					"far":          1000.0,
				},
			},
			{
				"type":      "light",
				"lightType": "ambient",
				"color":     0xFFFFFF,
				"intensity": 0.6,
				"name":      "Ambient Light",
				"layer":     "lights",
			},
			{
				"type":       "light",
				"lightType":  "directional",
				"color":      0xFFFFFF,
				"intensity":  0.8,
				"position":   []float64{5, 10, 7},
				"target":     []float64{0, 0, 0},
				"castShadow": true,
				"name":       "Main Light",
				"layer":      "lights",
			},
		}, allObjects...)
	}

	colors := []int{0xFF6B6B, 0x4ECDC4, 0x45B7D1, 0xFFA07A, 0x98D8C8, 0xF7DC6F, 0xBB8FCE, 0x85C1E2}
	for i, block := range scene.Blocks {
		obj := SceneObject{
			"type":  "mesh",
			"name":  block.Name,
			"layer": "generated",
			"pos":   []float64{block.Position[0], block.Position[1], block.Position[2]},
			"euler": []float64{block.Rotation[0], block.Rotation[1], block.Rotation[2]},
			"shape": map[string]interface{}{
				"type":   "box",
				"width":  block.Width,
				"height": block.Height,
				"depth":  block.Depth,
			},
			"material": map[string]interface{}{
				"type":      "standard",
				"color":     colors[i%len(colors)],
				"metalness": 0.1,
				"roughness": 0.7,
			},
			"castShadow":    true,
			"receiveShadow": true,
		}
		allObjects = append(allObjects, obj)
	}

	data, err := json.MarshalIndent(allObjects, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	return ioutil.WriteFile(outputPath, data, 0644)
}
