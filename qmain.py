from typing import List, Tuple
import numpy as np
import torch
from torch import nn
import layers
from constants import test_bench_template


class Model:
    def __init__(self, model: nn.Sequential):
        self.model: nn.Sequential = model
        self.layers: List[layers.Layer] = (
            []
        )  # Assume Layer is a superclass in layers module

    def __str__(self) -> str:
        return "\n".join(str(layer) for layer in self.layers)

    def parse_layers(self) -> None:
        """Parse layers from a PyTorch model and populate self.layers with Verilog-compatible layer representations."""
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                self.layers.append(
                    layers.QLinear.layer_from(layer, i)
                )  # Quantized Linear Layer
            elif isinstance(layer, nn.ReLU):
                self.layers.append(
                    layers.QReLU(self.model[i - 1].out_features, i)
                )  # Quantized ReLU
            else:
                raise ValueError(f"Unknown layer type {type(layer).__name__}")

    # def forward_range(self, ranges: List[List[float]]) -> None:
    #     """Propagate input ranges through each layer for range-based forward calculations."""
    #     start = np.array(ranges)
    #     for layer in self.layers:
    #         start = layer.forward_range(start)

    def get_vars(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Generate Verilog variable definitions for inputs and outputs with fixed bit sizes for quantized model."""
        # Define input and output names based on the first and last layer's shape
        in_params = [f"in{i}" for i in range(self.layers[0].shape[0])]
        out_params = [f"out{i}" for i in range(self.layers[-1].shape[-1])]

        # For a quantized model, use fixed bit sizes for inputs (int8) and outputs (int32)
        in_definitions = [
            f"    input signed [7:0] {in_params[i]};  // int8 input"
            for i in range(self.layers[0].shape[0])
        ]

        out_definitions = [
            f"    output signed [31:0] {out_params[i]};  // int32 output accumulator"
            for i in range(self.layers[-1].shape[-1])
        ]

        return in_params, out_params, in_definitions, out_definitions

    def emit(self) -> str:
        """Generate Verilog code for the model, including module definition and inter-layer connections."""
        out = ["`timescale 1ns / 1ps"]
        in_params, out_params, in_definitions, out_definitions = self.get_vars()

        # Top module definition
        top = [
            f"module top({','.join(in_params)}, {','.join(out_params)});",
            *in_definitions,
            *out_definitions,
        ]

        # Wiring between layers
        in_wires = in_params
        out_wires = []

        for i, layer in enumerate(self.layers):
            # Emit Verilog code for each layer and append it to `out`
            out.append(layer.emit())

            # Define output wires for the current layer, ensuring correct bit width
            out_wires = [f"layer_{i}_out_{j}" for j in range(layer.shape[-1])]
            for j in range(layer.shape[-1]):
                top.append(f"    wire signed [31:0] layer_{i}_out_{j};")

            # Connect the current layer outputs to the next layer inputs
            top.append(
                f"    {layer.name} layer_{i}({','.join(in_wires)}, {','.join(out_wires)});"
            )

            # Update `in_wires` for the next layer to be the current layer's `out_wires`
            in_wires = out_wires

        # Assign output wires to top-level outputs, using the correct bit widths
        assigns = [
            f"    assign {out_params[i]} = {out_wire};"
            for i, out_wire in enumerate(out_wires)
        ]
        top.extend(assigns)
        top.append("endmodule")

        # Combine module definitions and top logic into final Verilog code
        out.append("\n".join(top))
        return "\n".join(out)

    def emit_test_bench(self) -> str:
        """Generate Verilog test bench for the model."""
        in_params, out_params, in_definitions, out_definitions = self.get_vars()
        assigns = [f"        assign {param} = 0;" for param in in_params]

        return test_bench_template.format(
            in_params=", ".join(in_params),
            out_params=", ".join(out_params),
            in_definitions="\n    ".join(in_definitions),
            out_definitions="\n    ".join(out_definitions),
            assignments="\n".join(assigns),
        )


def test() -> None:
    """Testing function to generate and save Verilog code and test bench for a sample model."""
    simple_model = nn.Sequential(
        nn.Linear(2, 1),
        nn.ReLU(),
    )

    # Set weights and bias for deterministic behavior
    simple_model[0].weight = nn.Parameter(torch.tensor([[1.0, -1.0]]))
    simple_model[0].bias = nn.Parameter(torch.tensor([1.0]))

    model = Model(simple_model)
    model.parse_layers()
    # model.forward_range([[1.0, 100.0], [0.0, 1024.0]])

    print("Model Layers:\n", model)
    verilog_code = model.emit()
    test_bench_code = model.emit_test_bench()

    # Write Verilog code to file
    with open("test.v", "w") as verilog_file:
        verilog_file.write(verilog_code)

    # Write test bench to file
    with open("test_tb.v", "w") as tb_file:
        tb_file.write(test_bench_code)


if __name__ == "__main__":
    test()