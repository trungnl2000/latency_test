import torch as th
import torch.nn as nn

from custom_op.conv_svd_with_var import wrap_convSVD_with_var_layer
from custom_op.conv_hosvd_with_var import wrap_convHOSVD_with_var_layer
from custom_op.conv_avg import wrap_conv_layer
from custom_op.conv import wrap_conv
# Tạo dữ liệu đầu vào
batch_size = 128
in_channels = 576
height = 7
width = 7
data = th.randn(batch_size, in_channels, height, width)

labels = th.randint(0, 10, (batch_size,)).long()

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = wrap_conv(nn.Conv2d(in_channels, out_channels, kernel_size))

    def forward(self, x):
        return self.conv(x)

class ConvLayer_avg(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, radius, active):
        super(ConvLayer_avg, self).__init__()
        self.conv = wrap_conv_layer(nn.Conv2d(in_channels, out_channels, kernel_size), radius, active)

    def forward(self, x):
        return self.conv(x)
    
class ConvLayer_svd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, SVD_var, active):
        super(ConvLayer_svd, self).__init__()
        self.conv = wrap_convSVD_with_var_layer(nn.Conv2d(in_channels, out_channels, kernel_size), SVD_var, active)

    def forward(self, x):
        return self.conv(x)
    
class ConvLayer_hosvd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, SVD_var, active):
        super(ConvLayer_hosvd, self).__init__()
        self.conv = wrap_convHOSVD_with_var_layer(nn.Conv2d(in_channels, out_channels, kernel_size), SVD_var, active)

    def forward(self, x):
        return self.conv(x)
criterion = nn.CrossEntropyLoss()
# Tạo một instance của lớp convolution
active = True
SVD_var = 0.8
radius = 2
####################################################
print("_____Normal conv_____")
conv_layer = ConvLayer(in_channels, out_channels=3, kernel_size=3)
# Forward pass
output = conv_layer(data)
loss = criterion(output.view(batch_size, -1), labels)
# Backward pass
loss.backward()
####################################################
print("_____ConvLayer_avg_____")
conv_layer_avg = ConvLayer_avg(in_channels, out_channels=3, kernel_size=3, radius=radius, active=active)
# Forward pass
output = conv_layer_avg(data)
loss = criterion(output.view(batch_size, -1), labels)
# Backward pass
loss.backward()
####################################################
print("_____ConvLayer_svd_____")
conv_layer_svd = ConvLayer_svd(in_channels, out_channels=3, kernel_size=3, SVD_var=SVD_var, active=active)
# Forward pass
output = conv_layer_svd(data)
loss = criterion(output.view(batch_size, -1), labels)
# Backward pass
loss.backward()
####################################################
print("_____ConvLayer_hosvd_____")
conv_layer_hosvd = ConvLayer_hosvd(in_channels, out_channels=3, kernel_size=3, SVD_var=SVD_var, active=active)
# Forward pass
output = conv_layer_hosvd(data)
loss = criterion(output.view(batch_size, -1), labels)
# Backward pass
loss.backward()
####################################################
print("_____Normal conv_____")
conv_layer = ConvLayer(in_channels, out_channels=3, kernel_size=3)
# Forward pass
output = conv_layer(data)
loss = criterion(output.view(batch_size, -1), labels)
# Backward pass
loss.backward()



