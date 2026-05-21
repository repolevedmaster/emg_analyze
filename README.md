# Neural Semantic Alignment for EMG Signals

This repository contains a proof-of-concept implementation that explores the structural alignment between human electromyography (EMG) signals and the semantic latent space of artificial neural networks.

## Hypothesis

The project is built on the hypothesis that the human thought process during subvocalization generates minute muscle movements that can be mapped to an AI model's internal semantic state. 

1. **EMG Latent Space**: When a person thinks of a word or phrase, consistent micro-muscle activations occur (e.g., subvocalization, facial muscles). These activations create a distinct pattern in the EMG data.
2. **AI Semantic Latent Space**: Artificial intelligence models process inputs through hidden states and latent semantic spaces before generating output tokens.
3. **Alignment**: By training an encoder to project raw EMG time-series data directly into a space structurally similar to an AI's semantic latent space, we can map human internal speech to generated text.

## Architecture

The main script (`neural_semantic_alignment.py`) implements this concept using PyTorch.

- **EMG Encoder**: A 1D Convolutional Neural Network (CNN) that extracts spatial and temporal features from an 8-channel, 2000-frame EMG signal, transforming it into a high-dimensional latent vector.
- **Semantic Projection**: A series of dense layers (Linear, BatchNorm, GELU) that maps the extracted EMG features into a defined semantic latent space.
- **Decoder**: A linear classification layer that generates the final class (sentence) prediction from the semantic latent representation.

## Data Processing

Unlike simpler models that average the entire time-series data into a single vector (losing temporal information), this model preserves the temporal dimension. The preprocessing pipeline involves:
- Reshaping the data to 8 channels.
- Padding or truncating sequences to a fixed length of 2000 frames.
- Standardizing the data on a per-channel basis.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- scikit-learn

## Dataset (Kaggle)

The required dataset for this project is hosted on Kaggle due to its large size. 

1. Download the dataset from Kaggle: `[https://www.kaggle.com/datasets/rabinnepal/silent-speech-raw-emg-dataset]`
2. Extract the file if necessary, and place `data.pickle` in the root directory of this repository.

## Usage

Ensure `data.pickle` is located in the same directory as the script. Run the decoder with:

```bash
python neural_semantic_alignment.py
```

The script will:
1. Load and preprocess the dataset.
2. Train the alignment model.
3. Evaluate the model on a test set, printing the final accuracy.
4. Output a step-by-step simulation of the inference process, demonstrating how the raw EMG signal is transformed into the final text output.
