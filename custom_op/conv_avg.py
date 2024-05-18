import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, avg_pool2d
import torch.nn as nn
from math import ceil
import time

######################## Gradient filter ###################################
class Conv2dAvgOp(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:

        start_time = time.time()

        x, weight, bias, stride, dilation, padding, order, groups, current_index, forward_time, backward_time = args # This follow the order when passing variable in Conv2dAvgOp.apply() below
        x_h, x_w = x.shape[-2:]
        k_h, k_w = weight.shape[-2:]
        y = conv2d(x, weight, bias, stride, padding, dilation=dilation, groups=groups)
        h, w = y.shape[-2:]
        p_h, p_w = ceil(h / order), ceil(w / order)
        weight_sum = weight.sum(dim=(-1, -2))
        x_order_h, x_order_w = order * stride[0], order * stride[1]
        x_pad_h, x_pad_w = ceil(
            (p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)
        x_sum = avg_pool2d(x, kernel_size=(x_order_h, x_order_w),
                           stride=(x_order_h, x_order_w),
                           padding=(x_pad_h, x_pad_w), divisor_override=1)
        cfgs = th.tensor([bias is not None, groups != 1,
                          stride[0], stride[1],
                          x_pad_h, x_pad_w,
                          k_h, k_w,
                          x_h, x_w, order])
        ctx.save_for_backward(x_sum, weight_sum, cfgs, th.tensor([current_index]), backward_time)

        forward_time[current_index] = (time.time() - start_time)*1000
        # print("Forward time:", forward_time * 1000, " ms")
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:

        start_time = time.time()

        x_sum, weight_sum, cfgs, current_index, backward_time = ctx.saved_tensors
        has_bias, grouping,\
            s_h, s_w,\
            x_pad_h, x_pad_w,\
            k_h, k_w,\
            x_h, x_w, order = [int(c) for c in cfgs]
        n, c_in, p_h, p_w = x_sum.shape
        grad_y, = grad_outputs
        _, c_out, gy_h, gy_w = grad_y.shape
        grad_y_pad_h, grad_y_pad_w = ceil(
            (p_h * order - gy_h) / 2), ceil((p_w * order - gy_w) / 2)
        grad_y_avg = avg_pool2d(grad_y, kernel_size=order, stride=order,
                                padding=(grad_y_pad_h, grad_y_pad_w),
                                count_include_pad=False)
        if grouping:
            grad_x_sum = grad_y_avg * weight_sum.view(1, c_out, 1, 1)
            grad_w_sum = (x_sum * grad_y_avg).sum(dim=(0, 2, 3))
            grad_w = th.broadcast_to(grad_w_sum.view(
                c_out, 1, 1, 1), (c_out, 1, k_h, k_w)).clone()
        else:
            grad_x_sum = (
                weight_sum.t() @ grad_y_avg.flatten(start_dim=2)).view(n, c_in, p_h, p_w)
            gy = grad_y_avg.permute(1, 0, 2, 3).flatten(start_dim=1)
            gx = x_sum.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=-2)
            grad_w_sum = gy @ gx
            grad_w = th.broadcast_to(grad_w_sum.view(
                c_out, c_in, 1, 1), (c_out, c_in, k_h, k_w)).clone()
        grad_x = th.broadcast_to(grad_x_sum.view(n, c_in, p_h, p_w, 1, 1),
                                 (n, c_in, p_h, p_w, order * s_h, order * s_w))
        grad_x = grad_x.permute(0, 1, 2, 4, 3, 5).reshape(
            n, c_in, p_h * order * s_h, p_w * order * s_w)
        grad_x = grad_x[..., x_pad_h:x_pad_h + x_h, x_pad_w:x_pad_w + x_w]
        if has_bias:
            grad_b = grad_y.sum(dim=(0, 2, 3))
        else:
            grad_b = None


        backward_time[current_index[0]] = (time.time() - start_time)*1000
        # print("Backward time:", backward_time * 1000, " ms")

        return grad_x, grad_w, grad_b, None, None, None, None, None, None, None, None


class Conv2dAvg(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            order=4,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False,
            forward_time=0,
            current_index=0,
            backward_time=0
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        # assert padding[0] == kernel_size[0] // 2 and padding[1] == kernel_size[1] // 2
        super(Conv2dAvg, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding=padding,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.order = order
        self.forward_time = forward_time
        self.current_index = current_index
        self.backward_time = backward_time

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.activate:
            if self.dilation[0] == 1 or self.dilation[0] < self.order:
                y = Conv2dAvgOp.apply(x, self.weight, self.bias, self.stride, self.dilation,
                                      self.padding, self.order, self.groups, self.current_index, self.forward_time, self.backward_time)
            else:
                y = Conv2dDilatedOp.apply(x, self.weight, self.bias, self.stride, self.dilation,
                                          self.padding, self.order, self.groups)
        else:
            y = super().forward(x)
        return y





def wrap_conv_layer(conv, radius, active, current_index, forward_time, backward_time):
    new_conv = Conv2dAvg(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         order=radius,
                         activate=active,
                         current_index=current_index,
                         forward_time = forward_time,
                         backward_time = backward_time
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv
