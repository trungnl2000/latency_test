import torch as th
import torch.nn as nn
import argparse
import os

from custom_op.conv_avg import wrap_conv_layer

class ConvLayer_avg(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, radius, active, current_index, forward_time, backward_time):
        super(ConvLayer_avg, self).__init__()
        self.conv = wrap_conv_layer(nn.Conv2d(in_channels, out_channels, kernel_size, stride), 
                                    radius, active, current_index, forward_time, backward_time)

    def forward(self, x):
        return self.conv(x)
    
def write_to_file(filename, header, data):
    if not os.path.exists("result"):
        os.mkdir("result")
    filename = os.path.join("result", filename)
    file_exists = os.path.isfile(filename)
    with open(filename, 'a') as f:
        if not file_exists:
            f.write(header + "\n")
        f.write(data + "\n")

def main(**kwargs):
    # Unpack values from kwargs
    device = kwargs["device"]
    number_of_iteration = kwargs["number_of_iteration"] + 1  # 1 warm-up iterations
    in_channels = kwargs["in_channels"]
    out_channels = kwargs["out_channels"]
    stride = kwargs["stride"]
    kernel_size = kwargs["kernel_size"]
    batch_size = kwargs["batch_size"]
    height = kwargs["height"]
    width = kwargs["width"]
    radius = kwargs["radius"]
    output_file = kwargs["output_file"] + "_" + device + ".out"
    # Loss function
    criterion = nn.CrossEntropyLoss()

    ####################################################
    print("_____GradFilter conv_____")
    forward_time = th.zeros(number_of_iteration)
    backward_time = th.zeros(number_of_iteration)
    for i in range(number_of_iteration):
        # Generate input data
        data = th.randn(batch_size, in_channels, height, width).to(device)
        labels = th.randint(0, 10, (batch_size,)).long().to(device)
        active = True
        # Create layer
        conv_layer_avg = ConvLayer_avg(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, radius=radius, 
                                         active=active, current_index=i, forward_time=forward_time, backward_time=backward_time).to(device)
        # Forward pass
        output = conv_layer_avg(data)
        loss = criterion(output.view(batch_size, -1), labels)
        # Backward pass
        loss.backward()

    forward_time_mean = th.mean(forward_time[10:])
    backward_time_mean = th.mean(backward_time[10:])
    print("Average forward time: ", forward_time_mean.item(), " ms")
    print("Average backward time: ", backward_time_mean.item(), " ms")

    # Prepare data to write
    header = "device number_of_iteration in_channels out_channels kernel_size batch_size height width radius forward_time_mean(ms) backward_time_mean(ms)"
    data = f"{device} {number_of_iteration-1} {in_channels} {out_channels} {kernel_size} {batch_size} {height} {width} {radius} {forward_time_mean.item()} {backward_time_mean.item()}"
    write_to_file(output_file, header, data)

# Set up argparse to get command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose device and parameters to run the program.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Device to run the program (cpu or gpu).")
    parser.add_argument("--number_of_iteration", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--in_channels", type=int, default=576, help="Number of input channels.")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of output channels.")
    parser.add_argument("--stride", type=int, default=1, help="Stride.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--height", type=int, default=7, help="Height of input data.")
    parser.add_argument("--width", type=int, default=7, help="Width of input data.")
    parser.add_argument("--radius", type=int, default=2, help="Gradfilt Patch size")
    parser.add_argument("--output_file", type=str, default="latency_test_conv_avg", help="Output file to write results.")
    args = parser.parse_args()

    # Choose device based on command line argument
    device = "cuda" if args.device == "cuda" and th.cuda.is_available() else "cpu"
    if args.device == "cuda" and not th.cuda.is_available():
        print("GPU is not available, using CPU instead.")
    print(f"Using device: {device}")

    main(**vars(args))
