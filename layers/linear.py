from typing import List
import numpy as np
from layers.utils import range_to_bits
from torch import nn


class Linear:
    @classmethod
    def layer_from(cls, layer: nn.Linear, index: int):
        """Create a Linear layer instance from a PyTorch layer with given index."""
        print(layer.weight.detach().numpy().T)
        print(layer.bias.detach().numpy())
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            weight=layer.weight.detach().numpy().T,
            bias=layer.bias.detach().numpy(),
            index=index,
        )

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: np.ndarray,
        bias: np.ndarray,
        index: int,
    ):
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight: np.ndarray = weight
        self.bias: np.ndarray = bias

        # Verify the weight and bias shapes
        self.verify_weights()

        self.name: str = f"layer_{index}_linear_{in_features}_{out_features}"
        self.shape: tuple = (in_features, out_features)
        self.in_bits: List[int] = []
        self.out_bits: List[int] = []

    def __str__(self) -> str:
        return f"Linear({self.in_features} -> {self.out_features})"

    def verify_weights(self) -> None:
        """Validate the shapes of weight and bias arrays."""
        if self.weight is None:
            raise ValueError("Weight is not defined")

        expected_weight_shape = (self.in_features, self.out_features)
        expected_bias_shape = (self.out_features,)

        if self.weight.shape != expected_weight_shape:
            raise ValueError(
                f"Weight shape is incorrect; expected {expected_weight_shape}, got {self.weight.shape}"
            )

        if self.bias is not None and self.bias.shape != expected_bias_shape:
            raise ValueError(
                f"Bias shape is incorrect; expected {expected_bias_shape}, got {self.bias.shape}"
            )

    def forward_range(self, in_range: np.ndarray) -> np.ndarray:
        """Compute the output range for this layer based on input ranges."""
        out_range = np.array([in_range.T[0] @ self.weight, in_range.T[1] @ self.weight])
        out_range = (out_range + self.bias).T

        self.in_bits = [range_to_bits(*r) for r in in_range]
        self.out_bits = [range_to_bits(*r) for r in out_range]

        return out_range

    def emit(self) -> str:
        """Generate Verilog code for this linear layer."""
        # Generate Verilog code snippets for adding bias and multiplying weights
        add_bias_lines = [
            f"add{i} = mul{i} + {self.bias[i]};\n" for i in range(self.out_features)
        ]
        multiply_weight_lines = [
            f"mul{i} = mul{i} + in{j} * {self.weight[j][i]};\n"
            for i in range(self.out_features)
            for j in range(self.in_features)
        ]

        # Input and output parameters with definitions for Verilog
        in_params = [f"in{i}" for i in range(self.in_features)]
        out_params = [f"out{i}" for i in range(self.out_features)]
        in_definitions = [
            f"input [{self.in_bits[i] - 1}:0] {in_params[i]};\n"
            for i in range(self.in_features)
        ]
        out_definitions = [
            f"output [{self.out_bits[i] - 1}:0] {out_params[i]};\n"
            for i in range(self.out_features)
        ]

        # Definitions for intermediate multipliers and adders in Verilog
        mul_definitions = [
            f"reg [{self.out_bits[i] - 1}:0] mul{i};\n"
            for i in range(self.out_features)
        ]
        add_definitions = [
            f"reg [{self.out_bits[i] - 1}:0] add{i};\n"
            for i in range(self.out_features)
        ]

        # Final assignment statements in Verilog
        assign_lines = [f"assign out{i} = add{i};\n" for i in range(self.out_features)]

        # Return the full Verilog module as a formatted string
        return f"""
module {self.name}({", ".join(in_params)}, {", ".join(out_params)});
    {"    ".join(in_definitions)}
    {"    ".join(out_definitions)}
    
    {"    ".join(mul_definitions)}
    {"    ".join(add_definitions)}
        
    always @(*)
    begin
        {"        ".join(multiply_weight_lines)}
        {"        ".join(add_bias_lines)}
    end
    {"  ".join(assign_lines)}
endmodule
"""
