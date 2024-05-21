import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d
import torch.nn as nn
import time
from typing import Tuple, List

###### HOSVD base on variance #############

def unfolding(n, A):
    shape = A.shape
    size = th.prod(th.tensor(shape))
    lsize = size // shape[n]
    sizelist = list(range(len(shape)))
    sizelist[n] = 0
    sizelist[0] = n
    return A.permute(sizelist).reshape(shape[n], lsize)


def truncated_svd(
    X: th.Tensor,
    k: int=1,
    n_iter: int = 2,
    n_oversamples: int = 8,
    var: float = 0.9
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    m, n = X.shape
    Q = th.randn(n, k + n_oversamples)
    Q = X @ Q

    Q, _ = th.linalg.qr(Q)

    # Power iterations
    for _ in range(n_iter):
        Q = (Q.t() @ X).t()
        Q, _ = th.linalg.qr(Q)
        Q = X @ Q
        Q, _ = th.linalg.qr(Q)

    QA = Q.t() @ X
    # Transpose QA to make it tall-skinny as MAGMA has optimisations for this
    # (USVt)t = VStUt
    Va, S, R = th.linalg.svd(QA.t(), full_matrices=False)
    U = Q @ R.t()

    # total_variance = th.sum(S**2)
    # explained_variance = th.cumsum(S**2, dim=0) / total_variance
    # # k = (explained_variance >= var).nonzero()[0].item() + 1
    # nonzero_indices = (explained_variance >= var).nonzero()
    # if len(nonzero_indices) > 0:
    #     # Nếu có ít nhất một phần tử >= var
    #     k = nonzero_indices[0].item() + 1
    # else:
    #     # Nếu không có phần tử nào >= var, gán k bằng vị trí của phần tử lớn nhất
    #     k = explained_variance.argmax().item() + 1


    return U[:, :k], S[:k], Va.t()[:k, :]

def modalsvd(n, A, var):
    nA = unfolding(n, A)
    return truncated_svd(X=nA, k=1)

def hosvd(A, var=0.9):
    S = A.clone()
    
    u0, _, _ = modalsvd(0, A, var)
    S = th.tensordot(S, u0, dims=([0], [0]))

    # u1, _, _ = modalsvd(1, A, var)
    # S = th.tensordot(S, u1, dims=([0], [0]))

    # u2, _, _ = modalsvd(2, A, var)
    # S = th.tensordot(S, u2, dims=([0], [0]))

    # u3, _, _ = modalsvd(3, A, var)
    # S = th.tensordot(S, u3, dims=([0], [0]))
    # return S, u0, u1, u2, u3
    return u0

def restore_hosvd(S, u0, u1, u2, u3):
    # Initialize the restored tensor
    restored_tensor = S.clone()

    # Multiply each mode of the restored tensor by the corresponding U matrix
    restored_tensor = th.tensordot(restored_tensor, u0.t(), dims=([0], [0]))
    restored_tensor = th.tensordot(restored_tensor, u1.t(), dims=([0], [0]))
    restored_tensor = th.tensordot(restored_tensor, u2.t(), dims=([0], [0]))
    restored_tensor = th.tensordot(restored_tensor, u3.t(), dims=([0], [0]))
    return restored_tensor

###############################################################
class Conv2dHOSVDop_one_component(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        input, weight, bias, stride, dilation, padding, groups, var, current_index, forward_time, backward_time = args

        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)


        # S, u0, u1, u2, u3 = hosvd(input, var=var)
        # ctx.save_for_backward(S, u0, u1, u2, u3, weight, bias, th.tensor([current_index]), backward_time)
        hosvd(input, var=var)

        ctx.save_for_backward(input, weight, bias, th.tensor([current_index]), backward_time)

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

        # S, u0, u1, u2, u3, weight, bias, current_index, backward_time = ctx.saved_tensors
        # input = restore_hosvd(S, u0, u1, u2, u3)
        input, weight, bias, current_index, backward_time = ctx.saved_tensors
        

         
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
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None

class Conv2dHOSVD_one_component(nn.Conv2d):
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
        super(Conv2dHOSVD_one_component, self).__init__(in_channels=in_channels,
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
            y = Conv2dHOSVDop_one_component.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.var, self.current_index, self.forward_time, self.backward_time)
        else:
            y = super().forward(x)
        return y

def wrap_convHOSVD_one_component_layer(conv, SVD_var, active, current_index, forward_time, backward_time):
    new_conv = Conv2dHOSVD_one_component(in_channels=conv.in_channels,
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