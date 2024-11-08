from typing import List
import numpy as np
from layers.utils import range_to_bits
from torch import nn


class Linear:
    @classmethod
    def layer_from(cls, layer: nn.Linear, index: int):
        """Create a Linear layer instance from a PyTorch layer with given index."""
        # Quantize weights and bias to int8
        scale = np.max(np.abs(layer.weight.detach().numpy())) / 127.0
        quantized_weight = np.round(layer.weight.detach().numpy() / scale).astype(
            np.int8
        )
        quantized_bias = np.round(layer.bias.detach().numpy() / scale).astype(np.int8)

        print(f"Quantized weight (int8): {quantized_weight}")
        print(f"Quantized bias (int8): {quantized_bias}")

        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            weight=quantized_weight.T,  # Transpose for compatibility with Verilog
            bias=quantized_bias,
            index=index,
            scale=scale,  # Save scale factor for dequantization
        )

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: np.ndarray,
        bias: np.ndarray,
        index: int,
        scale: float,
    ):
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight: np.ndarray = weight  # Now int8 quantized weights
        self.bias: np.ndarray = bias  # Now int8 quantized bias
        self.scale: float = scale  # Scale factor for dequantization

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

    # def forward_range(self, in_range: np.ndarray) -> np.ndarray:
    #     """Compute the output range for this layer based on input ranges."""
    #     out_range = np.array([in_range.T[0] @ self.weight, in_range.T[1] @ self.weight])
    #     out_range = (out_range + self.bias).T

    #     self.in_bits = [range_to_bits(*r) for r in in_range]
    #     self.out_bits = [range_to_bits(*r) for r in out_range]

    #     return out_range

    def emit(self) -> str:
        """Generate Verilog code for this quantized linear layer with MAC initialization."""
        # Verilog code snippets for initializing, adding bias, and multiplying int8 weights
        init_mul_lines = [f"mul{i} = 32'sd0;" for i in range(self.out_features)]
        add_bias_lines = [
            f"add{i} = mul{i} + {int(self.bias[i])};  // Bias int8\n"
            for i in range(self.out_features)
        ]
        multiply_weight_lines = [
            f"mul{i} = mul{i} + in{j} * {int(self.weight[j][i])};  // Weight int8\n"
            for i in range(self.out_features)
            for j in range(self.in_features)
        ]

        # Define 8-bit input, 32-bit accumulator (output)
        in_params = [f"in{i}" for i in range(self.in_features)]
        out_params = [f"out{i}" for i in range(self.out_features)]
        in_definitions = [
            f"input signed [7:0] {in_params[i]};  // int8 input\n"
            for i in range(self.in_features)
        ]
        out_definitions = [
            f"output signed [31:0] {out_params[i]};  // 32-bit accumulator output\n"
            for i in range(self.out_features)
        ]

        # 32-bit accumulator definitions in Verilog for intermediate values
        mul_definitions = [
            f"reg signed [31:0] mul{i};\n" for i in range(self.out_features)
        ]
        add_definitions = [
            f"reg signed [31:0] add{i};\n" for i in range(self.out_features)
        ]

        # Final assignment statements in Verilog
        assign_lines = [
            f"assign {out_params[i]} = add{i};\n" for i in range(self.out_features)
        ]

        # Return the full Verilog module as a formatted string
        return f"""
    module {self.name}({", ".join(in_params)}, {", ".join(out_params)});
        {"    ".join(in_definitions)}
        {"    ".join(out_definitions)}
        
        {"    ".join(mul_definitions)}
        {"    ".join(add_definitions)}
            
        always @(*)
        begin
            // Initialize accumulators
            {"        ".join(init_mul_lines)}
            
            // Perform MAC operations
            {"        ".join(multiply_weight_lines)}
            
            // Add bias
            {"        ".join(add_bias_lines)}
        end
        {"  ".join(assign_lines)}
    endmodule
    """
