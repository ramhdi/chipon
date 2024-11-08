from layers import linear, relu, qlinear, qrelu

Linear = linear.Linear
ReLU = relu.ReLU
QLinear = qlinear.Linear
QReLU = qrelu.ReLU

__all__ = ["Linear", "ReLU"]
