import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d
import torch.nn as nn
import time


###### SVD by choosing principle components based on variance

# Cho 2 chiều
def truncated_svd(X, var=0.9, dim=0, driver=None):
    # dim là số chiều mà mình sẽ svd theo
    n_samples, n_features = th.prod(th.tensor(X.shape[:dim+1])), th.prod(th.tensor(X.shape[dim+1:]))
    X_reshaped = X.view(n_samples, n_features)
    U, S, Vt = th.linalg.svd(X_reshaped, full_matrices=False, driver=driver)
    total_variance = th.sum(S**2)

    explained_variance = th.cumsum(S**2, dim=0) / total_variance
    # k = (explained_variance >= var).nonzero()[0].item() + 1
    # print("explained_variance: ", explained_variance)
    # print("k: ", k)
    nonzero_indices = (explained_variance >= var).nonzero()
    if len(nonzero_indices) > 0:
        # Nếu có ít nhất một phần tử >= var
        k = nonzero_indices[0].item() + 1
    else:
        # Nếu không có phần tử nào >= var, gán k bằng vị trí của phần tử lớn nhất
        k = explained_variance.argmax().item() + 1
    return th.matmul(U[:, :k], th.diag_embed(S[:k])) , Vt[:k, :]

def restore_tensor(Uk_Sk, Vk_t, shape):
    reconstructed_matrix = th.matmul(Uk_Sk, Vk_t)
    shape = tuple(shape)
    return reconstructed_matrix.view(shape)

###############################################################
class Conv2dSVDop_with_var(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        input, weight, bias, stride, dilation, padding, groups, var, current_index, forward_time, backward_time = args

        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups) # Chỗ này như bình thường

        input_Uk_Sk, input_Vk_t = truncated_svd(input, var=var)
        ctx.save_for_backward(input_Uk_Sk, input_Vk_t, th.tensor(input.shape), weight, bias, th.tensor([current_index]), backward_time)

        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        forward_time[current_index] = (time.time() - start_time)*1000
        # print("Forward time:", forward_time * 1000, " ms")

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        start_time = time.time()

        input_Uk_Sk, input_Vk_t, input_shape, weight, bias, current_index, backward_time = ctx.saved_tensors
        input = restore_tensor(input_Uk_Sk, input_Vk_t, input_shape)

        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs

        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        backward_time[current_index[0]] = (time.time() - start_time)*1000
        # print("Backward time:", backward_time * 1000, " ms")

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None # Trả về gradient ứng với cái arg ở forward

class Conv2dSVD_with_var(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False,
            var=1,
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
        super(Conv2dSVD_with_var, self).__init__(in_channels=in_channels,
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
        self.var = var
        self.forward_time = forward_time
        self.current_index = current_index
        self.backward_time = backward_time

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.activate:
            y = Conv2dSVDop_with_var.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.var, self.current_index, self.forward_time, self.backward_time)
        else:
            y = super().forward(x)
        return y

def wrap_convSVD_with_var_layer(conv, SVD_var, active, current_index, forward_time, backward_time):
    new_conv = Conv2dSVD_with_var(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         var=SVD_var,
                         current_index=current_index,
                         forward_time = forward_time,
                         backward_time = backward_time
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv