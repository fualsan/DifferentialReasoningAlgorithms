import jax.numpy as jnp
import jax.random as jrd

import matplotlib.pyplot as plt


def binarize(x):
    return jnp.where(x>=0.5, 1, 0)


def calculate_metrics(x_pred, x_target, eps=1e-7):
    x_pred_binary = binarize(x_pred)
    
    acc = (jnp.sum(x_pred_binary == x_target) / len(x_pred_binary)) * 100.0
    
    tp = jnp.sum((x_pred_binary == 1.0) & (x_target == 1.0))
    tn = jnp.sum((x_pred_binary == 0.0) & (x_target == 0.0))
    fp = jnp.sum((x_pred_binary == 1.0) & (x_target == 0.0))
    fn = jnp.sum((x_pred_binary == 0.0) & (x_target == 1.0))

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2.0 * ((precision*recall)/(precision+recall+eps))    
    
    return {
        'Accuracy': acc,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F-1': f1
    }


def print_binary(x):
    for x_real, x_binary in zip(x.tolist(), binarize(x).tolist()):
        x_binary_str = 'True' if x_binary == 1 else 'False'
        print(f'{x_real:2.2f} --> {x_binary:1d} ({x_binary_str})')


def plot_norm_functions(x1, x2, norm_fn, norm_fn_name, save_name):
    X1, X2 = jnp.meshgrid(x1, x2)

    norm_output = norm_fn(X1, X2)

    fig = plt.figure(figsize=(8, 6))
    im = plt.imshow(
      norm_output,
      extent=(float(x1[0]), float(x1[-1]), float(x2[0]), float(x2[-1])),
      origin='lower',
      cmap='viridis',
      aspect='auto',
    )
    
    plt.colorbar(im, label='Truth')
    plt.xlabel(r'$X_1$')
    plt.ylabel(r'$X_2$')
    plt.title(f'2D Heatmap of {norm_fn_name}')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(
        f'{save_name}.png',
        bbox_inches='tight', 
        dpi=300
    )


def init_logical_array(array_shape, key, noise_scale=0.1):
    noise = jrd.normal(jrd.key(key), array_shape) * noise_scale
    # any shaped array centered around 0.5
    return 0.5 * jnp.ones(array_shape) + noise

