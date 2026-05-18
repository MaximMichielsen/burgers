from fem.constants import (
    INPUT_UNITS,
    HIDDEN_UNITS,
    OUTPUT_UNITS,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
)


def create_agent_architecture(
    input_layer: int,
    hidden_layer: int,
    output_layer: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
) -> dict:
    """Create a dictionary containing an artificial neural network's hyperparameters."""
    architecture = {
        "input_layer": input_layer,
        "output_layer": output_layer,
        "hidden_layer": hidden_layer,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
    }
    return architecture


architecture_standard = create_agent_architecture(
    INPUT_UNITS, HIDDEN_UNITS, OUTPUT_UNITS, BATCH_SIZE, EPOCHS, LEARNING_RATE
)

# robijns used 32000 training examples and 8000 validation examples
architecture_robijns = create_agent_architecture(
    INPUT_UNITS, 64, OUTPUT_UNITS, BATCH_SIZE, 100, learning_rate=LEARNING_RATE
)
