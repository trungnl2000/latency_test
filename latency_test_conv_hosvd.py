import torch as th
import torch.nn as nn
import argparse

from custom_op.conv_hosvd_with_var import wrap_convHOSVD_with_var_layer
 
class ConvLayer_hosvd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, SVD_var, active, current_index, forward_time, backward_time):
        super(ConvLayer_hosvd, self).__init__()
        self.conv = wrap_convHOSVD_with_var_layer(nn.Conv2d(in_channels, out_channels, kernel_size), 
                                                  SVD_var, active, current_index, forward_time, backward_time)

    def forward(self, x):
        return self.conv(x)
    
def main(**kwargs):
    # Unpack values from kwargs
    device = kwargs["device"]
    number_of_iteration = kwargs["number_of_iteration"] + 1  # 1 warm-up iterations
    in_channels = kwargs["in_channels"]
    out_channels = kwargs["out_channels"]
    kernel_size = kwargs["kernel_size"]
    batch_size = kwargs["batch_size"]
    height = kwargs["height"]
    width = kwargs["width"]
    SVD_var = kwargs["SVD_var"]
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    ####################################################
    print("_____HOSVD conv_____")
    forward_time = th.zeros(number_of_iteration)
    backward_time = th.zeros(number_of_iteration)
    for i in range(number_of_iteration):
        # Generate input data
        data = th.randn(batch_size, in_channels, height, width).to(device)
        labels = th.randint(0, 10, (batch_size,)).long().to(device)
        active = True
        # Create layer
        conv_layer_svd = ConvLayer_hosvd(in_channels, out_channels=out_channels, kernel_size=kernel_size, SVD_var=SVD_var, 
                                         active=active, current_index=i, forward_time=forward_time, backward_time=backward_time).to(device)
        # Forward pass
        output = conv_layer_svd(data)
        loss = criterion(output.view(batch_size, -1), labels)
        # Backward pass
        loss.backward()

    forward_time_mean = th.mean(forward_time[10:])
    backward_time_mean = th.mean(backward_time[10:])
    print("Average forward time: ", forward_time_mean.item(), " ms")
    print("Average backward time: ", backward_time_mean.item(), " ms")

# Set up argparse to get command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose device and parameters to run the program.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Device to run the program (cpu or gpu).")
    parser.add_argument("--number_of_iteration", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--in_channels", type=int, default=576, help="Number of input channels.")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of output channels.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--height", type=int, default=7, help="Height of input data.")
    parser.add_argument("--width", type=int, default=7, help="Width of input data.")
    parser.add_argument("--SVD_var", type=float, default=0.8, help="Variance of HOSVD")
    args = parser.parse_args()

    # Choose device based on command line argument
    device = "cuda" if args.device == "cuda" and th.cuda.is_available() else "cpu"
    if args.device == "cuda" and not th.cuda.is_available():
        print("GPU is not available, using CPU instead.")
    print(f"Using device: {device}")

    main(**vars(args))
