  return forward_call(*input, **kwargs)
  File "/mnt/diskb/penglong/dx/code/SR/Network.py", line 1130, in forward
    burst_feat = self.conv1(burst)
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1190, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/torch/nn/modules/container.py", line 204, in forward
    input = module(input)
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1190, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 463, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 459, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
