import torch
from tensorflow import keras
import einops as ein

def copy_dense(tf_dense, torch_dense, has_bias=True):
    tf_weight, tf_bias = tf_dense.get_weights()
    torch_dense.weight.data.copy_(torch.from_numpy(ein.rearrange(tf_weight, 'i o -> o i')))
    if has_bias: torch_dense.bias.data.copy_(torch.from_numpy(tf_bias))

def copy_conv(tf_conv, torch_conv):
    tf_weight = ein.rearrange(tf_conv.get_weights()[0], 'k i o -> o i k')
    torch_conv.weight.data.copy_(torch.from_numpy(tf_weight))

def copy_bn(tf_bn, torch_bn):
    tf_gamma, tf_beta, tf_ema_mean, tf_ema_var = map(torch.from_numpy, tf_bn.get_weights())
    torch_bn.weight.data.copy_(tf_gamma)
    torch_bn.bias.data.copy_(tf_beta)
    torch_bn.running_mean.data.copy_(tf_ema_mean)
    torch_bn.running_var.data.copy_(tf_ema_var)

def copy_tf_to_pytorch(tf_basenji, torch_basenji):
    # both models are nested sequential with the same layers
    tf_conv_layers = [module for module in tf_basenji.model.layers if isinstance(module, keras.layers.Conv1D)]
    torch_conv_layers = [module for module in torch_basenji.modules() if isinstance(module, torch.nn.Conv1d)]
    for tf_conv, torch_conv in zip(tf_conv_layers, torch_conv_layers):
        copy_conv(tf_conv, torch_conv)
    
    tf_bn_layers = [module for module in tf_basenji.model.layers if isinstance(module, keras.layers.BatchNormalization)]
    torch_bn_layers = [module for module in torch_basenji.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    for tf_bn, torch_bn in zip(tf_bn_layers, torch_bn_layers):
        copy_bn(tf_bn, torch_bn)
        
    tf_dense_layers = [module for module in tf_basenji.model.layers if isinstance(module, keras.layers.Dense)]
    torch_dense_layers = [module for module in torch_basenji.modules() if isinstance(module, torch.nn.Linear)]
    for tf_dense, torch_dense in zip(tf_dense_layers, torch_dense_layers):
        copy_dense(tf_dense, torch_dense)

    print('success')