# VLM Thought Process Visualization

This application visualizes the thought process during VLM (Visual Language Model) response generation. Inspired by Google DeepMind's paper, *Chain-of-Thought Reasoning without Prompting*, this project leverages VILA to reveal the model's step-by-step reasoning. By modifying the model output, it displays the token generated at each step along with its corresponding logits.

## How It Works

1. **Input**: A video and a reasoning question.
2. **Processing**: VILA processes the input using Chain-of-Thought decoding.
3. **Output**: The application displays each generated token and its logits, showing the thought process.

## Getting Started

1. Clone the VILA repository.
2. Modify the model output to display token generation and logits (use `run_token.py` with `run_vila.py`).
3. Adjust parameters as needed:

    - **`--temperature`**: Controls the randomness of predictions by scaling the logits before applying softmax. Lower values make the model more conservative, higher values make it more creative.

    - **`--top_p`**: Applies nucleus sampling, where only the most probable tokens whose cumulative probability exceeds this value are considered, increasing diversity in outputs.

    - **`--top_k`**: Limits the model to consider only the top-k highest-probability tokens during generation, helping to control randomness.

    - **`--num_beams`**: Specifies the number of beams for beam search, where more beams increase search diversity but require more computation.

    - **`--max_new_tokens`**: Limits the maximum number of new tokens generated by the model in a single output sequence.

4. Run the application with your desired video and question input.

## Example
1. **Input**: `train/videos/video_3399.mp4` & `Track the object that stops the motion of the launched object after being launched the first time.`
2. **Output**:
    ```plaintext
    Step 1: The (0.6595), I (0.1446), After (0.1034), To (0.0925)
    Step 2: object (0.9660), person (0.0340)
    ...
    ```

## Thought Process Examples

- **Example 1**: The object that stops the motion of the launched object is placed on the table.
- **Example 2**: The object that stops the motion of the launched bottle is seen placed on the table.

## References

- [Google DeepMind Paper: Chain-of-Thought Reasoning without Prompting](https://arxiv.org/pdf/2402.10200)
