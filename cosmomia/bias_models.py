import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping



def patchy_deterministic(params, delta, target_mean):
    alpha, rho_eps, epsilon = params
    delta = jnp.where(delta>=-1, delta, -1)
    #rho_eps = jnp.max(jnp.array([rho_eps, 0.]))
    alpha = jnp.max(jnp.array([alpha, 1]))
    
    #epsilon = jnp.max(jnp.array([alpha, 1]))
    delta_g = ((1 + delta)**alpha) * (jnp.exp(-((1+delta) / (rho_eps * target_mean))**jnp.abs(epsilon)))*jnp.e
    print(delta[jnp.isnan(delta_g)], params)
    
    delta_g_mean = delta_g.mean()
    norm = target_mean / delta_g_mean
    delta_g *= norm
    
    return delta_g


class ConvPoolCombo(eqx.Module):
    layers_down: list
    layers_up: list
    def __init__(self, key, in_channels, out_channels, conv_kernel, pool_kernel, pool_stride, activation = jax.nn.silu):
        key1, key2 = jax.random.split(key)
        self.layers_down = [
            eqx.nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel, key=key1, padding_mode = 'CIRCULAR', padding = 'SAME'),
            eqx.nn.AvgPool3d(kernel_size=pool_kernel, stride = pool_stride),
            activation,
        ]

        self.layers_up = [
            eqx.nn.ConvTranspose3d(out_channels, in_channels, kernel_size=pool_kernel, stride = pool_stride, key=key2, padding_mode = 'CIRCULAR', padding = 'SAME'),
            activation,
        ]
        
class UNET(eqx.Module):
    
    layers_down: list
    layers_up: list

    def __init__(self, key, n_channels, conv_kernels, pool_kernels, pool_strides):
        key1, key2 = jax.random.split(key)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        keys_down = jax.random.split(key1, len(n_channels))
        keys_up = jax.random.split(key2, len(n_channels))
        self.layers_down = []
        self.layers_up = []
        for i, (in_channel, out_channel, conv_kernel, pool_kernel, pool_stride) in enumerate(zip(n_channels[:-1], n_channels[1:], conv_kernels, pool_kernels, pool_strides)):
            self.layers_down.append(eqx.nn.Conv3d(in_channel, out_channel, kernel_size=conv_kernel, key=keys_down[i], padding_mode = 'CIRCULAR', padding = 'SAME'))
            self.layers_down.append(eqx.nn.AvgPool3d(kernel_size=pool_kernel, stride = pool_stride))
            self.layers_down.append(jax.nn.silu)
        
        
            self.layers_up.append(jax.nn.silu)
            self.layers_up.append(eqx.nn.ConvTranspose3d(out_channel, in_channel, kernel_size=pool_kernel, stride = pool_stride, key=keys_up[i], padding_mode = 'CIRCULAR', padding = 'SAME'))
        
        

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        intermediate = [x]
        for i, layer in enumerate(self.layers_down):
            x = layer(x)
            
            if i+1 % 3 == 0:
                intermediate.append(x)
        j = 1
        for i, layer in enumerate(self.layers_up[::-1]):
            x = layer(x)
            if i+1 % 2 ==0:
                #x = jnp.concatenate((x,intermediate[-(j)]), axis = 0)
                x += intermediate[-j]
                j+=1
        x += intermediate[0]
        return x



class CNN(eqx.Module):
    
    layers_down: list
    

    def __init__(self, key, n_channels, conv_kernels):
        key1, key2 = jax.random.split(key)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        keys_down = jax.random.split(key1, len(n_channels))
        keys_up = jax.random.split(key2, len(n_channels))
        self.layers_down = []
        
        for i, (in_channel, out_channel, conv_kernel) in enumerate(zip(n_channels[:-1], n_channels[1:], conv_kernels)):
            self.layers_down.append(eqx.nn.Conv3d(in_channel, out_channel, kernel_size=conv_kernel, key=keys_down[i], padding_mode = 'CIRCULAR', padding = 'SAME'))
            self.layers_down.append(jax.nn.silu)
        
        
        

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        xout = x.copy()
        for i, layer in enumerate(self.layers_down):
            xout = layer(xout)
        xout += x
            
        return xout



