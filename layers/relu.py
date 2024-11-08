from typing import List, Tuple
import numpy as np
from layers.utils import range_to_bits


class ReLU:
    def __init__(self, shape: int, index: int):
        self.shape: Tuple[int] = (shape,)
        self.name: str = f"layer_{index}_relu_{shape}"
        self.in_bits: List[int] = []
        self.out_bits: List[int] = []

    def __str__(self) -> str:
        return f"ReLU({self.shape})"

    def forward_range(self, in_range: np.ndarray) -> np.ndarray:
        """Compute the output range after applying ReLU to the input range."""
        out_range = np.maximum(in_range, 0)

        self.in_bits = [range_to_bits(*in_range[i]) for i in range(self.shape[0])]
        self.out_bits = [range_to_bits(*out_range[i]) for i in range(self.shape[0])]

        return out_range

    def emit(self) -> str:
        """Generate Verilog code for this ReLU layer."""
        # Generate Verilog statements for ReLU operation
        relu_code = [f"out{i} = in{i} > 0 ? in{i} : 0;\n" for i in range(self.shape[0])]

        # Define input and output parameters for Verilog
        in_params = [f"in{i}" for i in range(self.shape[0])]
        out_params = [f"out{i}" for i in range(self.shape[0])]
        in_definitions = [
            f"input [{self.in_bits[i] - 1}:0] {in_params[i]};\n"
            for i in range(self.shape[0])
        ]
        out_definitions = [
            f"output reg [{self.out_bits[i] - 1}:0] {out_params[i]};\n"
            for i in range(self.shape[0])
        ]

        # Return the full Verilog module as a formatted string
        return f"""
module {self.name}({", ".join(in_params)}, {", ".join(out_params)});
    {"    ".join(in_definitions)}
    {"    ".join(out_definitions)}
                
    always @(*) 
    begin
        {"        ".join(relu_code)}
    end
endmodule
"""
