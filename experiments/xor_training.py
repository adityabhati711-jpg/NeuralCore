
"""XOR training experiment using pure NumPy NeuralCore."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuralcore.activations import sigmoid
from neuralcore.layers import Dense
from neuralcore.losses import BinaryCrossEntropyWithLogits
from neuralcore.network import Activation, Sequential
from neuralcore.optimizers import Adam


def main():
    rng = np.random.default_rng(42)

    x = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
    ])

    y = np.array([
        [0.],
        [1.],
        [1.],
        [0.],
    ])

    model = Sequential([
        Dense(2, 8, rng=rng, init="xavier"),
        Activation("tanh"),
        Dense(8, 1, rng=rng, init="xavier"),
    ])

    loss_fn = BinaryCrossEntropyWithLogits()
    optimizer = Adam(lr=0.05)

    epochs = 3000
    history = []

    initial_loss = loss_fn.forward(model.forward(x), y)

    for epoch in range(epochs):
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)

        if not np.isfinite(loss):
            raise FloatingPointError("Training loss became non-finite.")

        history.append(loss)

        model.backward(loss_fn.backward())
        optimizer.step(model.parameters_and_gradients())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}: loss = {loss:.6f}")

    logits = model.forward(x)
    final_loss = loss_fn.forward(logits, y)
    probabilities = sigmoid(logits)
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = np.mean(predictions == y)

    print(f"\nInitial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Accuracy:     {accuracy:.2%}")

    for inputs, probability, prediction in zip(
        x, probabilities, predictions
    ):
        print(
            f"Input {inputs.astype(int).tolist()} "
            f"-> probability {probability.item():.4f}, "
            f"prediction {prediction.item()}"
        )

    if accuracy != 1.0:
        raise AssertionError("XOR was not fully learned.")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, epochs + 1), history, linewidth=2)
    ax.set_title("NeuralCore — XOR Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "xor_loss.png", dpi=200)
    plt.close(fig)

    print("\nSaved: outputs/xor_loss.png")


if __name__ == "__main__":
    main()
