# %% [markdown]
# # Visualize a 1D neural network


# %% [markdown]
# Optimization and interactive control of a 1D -> 1D neural network function.

# %% [markdown]
# ## Preamble

# %%
# pylint: disable=missing-function-docstring, global-statement, redefined-outer-name

import contextlib
import math
from typing import Any

import IPython.display
import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import torch

# %%
SHOW_SLIDERS = True
VERBOSE = False
R = 10

# %%
# LAYER_SIZES = 3, 1
LAYER_SIZES = 2, 2, 1
# LAYER_SIZES = 3, 3, 1
# LAYER_SIZES = 4, 4, 1
# LAYER_SIZES = 4, 4, 4, 1  # It is best to set SHOW_SLIDERS=False.

# %% [markdown]
# ## Target functions

# %%
TARGETS = {
    'sine': lambda x: np.sin(math.tau * x),
    'cosine': lambda x: np.cos(math.tau * x),
    'triangle': lambda x: 1 - 2 * np.abs(x - 0.5),
    'half-sine': lambda x: np.sin(math.tau / 2 * x),
}


# %% [markdown]
# ## Activation functions


# %%
def relu(x):
  return torch.nn.functional.relu(x) if isinstance(x, torch.Tensor) else np.maximum(0, x)


def leaky_relu(x, negative_slope=0.1):
  if isinstance(x, torch.Tensor):
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
  return np.maximum(x * negative_slope, x)


def identity(x):
  return x


def sigmoid(x):
  return torch.sigmoid(x) if isinstance(x, torch.Tensor) else 1 / (1 + np.exp(-x))


# %%
ACTIVATIONS = {'relu': relu, 'leaky_relu': leaky_relu, 'identity': identity, 'sigmoid': sigmoid}


# %% [markdown]
# ## Neural network


# %%
# The dictionary `params` contains weights 'w{layer}{curr_index}{prev_index}' and
# biases 'b{layer}{index}', as either float values or (scalar) torch tensors.
def neural_network(x, params):
  prev_nodes = [x]
  for layer_index, layer_size in enumerate(LAYER_SIZES):
    is_last_layer = layer_index == len(LAYER_SIZES) - 1
    activation: Any = ACTIVATIONS[activation_selector.value] if not is_last_layer else identity
    nodes = []
    for node_index in range(layer_size):
      linear = sum(
          params[f'w{layer_index}{node_index}{prev_index}'] * prev_node
          for prev_index, prev_node in enumerate(prev_nodes)
      )
      node = activation(linear + params[f'b{layer_index}{node_index}'])
      nodes.append(node)
    prev_nodes = nodes
  (y,) = prev_nodes  # The last neural network layer must have a single node.
  return y


# %% [markdown]
# ## Params initialization


# %%
def get_reset_values():
  d: dict[str, float] = {}

  if LAYER_SIZES == (3, 1):
    d |= dict(w000=1.3, b00=-0.2, w010=3.0, b01=-2.1, w020=2.5, b02=-1.0)
    d |= dict(w100=7.0, w101=4.0, w102=-5.8, b10=-1.0)

  elif LAYER_SIZES == (2, 2, 1):
    # d |= dict(w000=-4.74, b00=2.05, w010=8.98, b01=0.14)
    # d |= dict(w100=-1.88, w101=-0.43, b10=2.84, w110=0.19, w111=-0.66, b11=1.97)
    # d |= dict(w200=-1.62, w201=-0.83, b20=0.66)
    d |= dict(w000=-3.45, b00=1.73, w010=5.27, b01=-2.44)
    d |= dict(w100=-2.21, w101=-0.27, b10=2.03, w110=0.51, w111=-0.66, b11=1.00)
    d |= dict(w200=-1.46, w201=-1.48, b20=3.08)

  else:
    rng = np.random.default_rng(0)  # Deterministic.
    prev_size = 1
    for layer_index, layer_size in enumerate(LAYER_SIZES):
      for node_index in range(layer_size):
        for prev_index in range(prev_size):
          d[f'w{layer_index}{node_index}{prev_index}'] = rng.random() * 2 - 1
        d[f'b{layer_index}{node_index}'] = rng.random() * 2 - 1
      prev_size = layer_size
    assert prev_size == 1

  return d


RESET_VALUES = get_reset_values()
params = RESET_VALUES


# %% [markdown]
# ## Neural network optimization


# %%
def fit_neural_network(num_steps=1500):
  tensor_params = {
      key: torch.tensor([value], dtype=torch.float32, requires_grad=True)
      for key, value in params.items()
  }
  variables = list(tensor_params.values())
  x_train = torch.linspace(0, 1, 100, dtype=torch.float32)
  y_train = TARGETS[target_selector.value](x_train)

  learning_rate = 0.01
  optimizer = torch.optim.Adam(variables, lr=learning_rate)
  for step in range(num_steps):
    optimizer.zero_grad()
    y_pred = neural_network(x_train, tensor_params)
    loss = torch.mean((y_train - y_pred) ** 2)
    loss.backward()
    optimizer.step()
    if VERBOSE and step % 100 == 0:
      print(f'Step {step}, Loss: {loss.item()}')  # See messages using "log" icon.

  update_params({key: tensor.detach().numpy().item() for key, tensor in tensor_params.items()})


# %% [markdown]
# ## User interface


# %%
def get_sliders():
  a = dict(step=0.01, continuous_update=True)
  return {
      key: ipywidgets.FloatSlider(value=value, min=value - R, max=value + R, description=key, **a)
      for key, value in params.items()
  }


target_selector = ipywidgets.RadioButtons(
    options=list(TARGETS), value='sine', description='Target function:'
)
activation_selector = ipywidgets.RadioButtons(
    options=list(ACTIVATIONS), value='leaky_relu', description='Activation function:'
)
sliders = get_sliders() if SHOW_SLIDERS else {}
center_button = ipywidgets.Button(description='Center slider ranges')
reset_button = ipywidgets.Button(description='Reset params')
randomize_button = ipywidgets.Button(description='Randomize params')
fit_button = ipywidgets.Button(description='Fit target')
output = ipywidgets.Output()
textarea = ipywidgets.Textarea(rows=len(LAYER_SIZES), layout={'width': '600px'}, disabled=True)


def update_plot():
  text_lines = []
  for layer_index in range(len(LAYER_SIZES)):
    layer_str = ', '.join(f'{k}={v:.2f}' for k, v in params.items() if int(k[1]) == layer_index)
    text_lines.append(f'dict({layer_str})')
  textarea.value = '\n'.join(text_lines)
  with output:
    output.clear_output(wait=True)
    _, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 1, 100)
    ax.plot(x, neural_network(x, params), label='neural network')
    ax.plot(x, TARGETS[target_selector.value](x), lw=0.5, label='target')
    ax.set(xlabel='x', ylabel='y=f(x)', ylim=(-2, 2))
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.show()


def update_params(new_params, center_sliders=True):
  global params
  params = new_params
  if SHOW_SLIDERS:
    with contextlib.ExitStack() as stack:
      for slider in reversed(sliders.values()):
        stack.enter_context(slider.hold_trait_notifications())
      for key, slider in sliders.items():
        value = params[key]
        slider.value = value  # Bug: https://github.com/jupyter-widgets/ipywidgets/issues/3824.
        if center_sliders:
          slider.min = value - R
          slider.max = value + R
  update_plot()


def on_slider_change():
  update_params({key: slider.value for key, slider in sliders.items()}, center_sliders=False)


def center_slider_range():
  for slider in sliders.values():
    value = slider.value
    slider.min, slider.max = value - R, value + R


def reset_parameters():
  update_params(RESET_VALUES)


def randomize_parameters():
  rng = np.random.default_rng()  # Non-deterministic.
  update_params({key: rng.random() * 2 - 1 for key in params})


def initialize_ui():
  target_selector.observe(lambda change: update_plot(), names='value')
  activation_selector.observe(lambda change: update_plot(), names='value')
  for slider in sliders.values():
    slider.observe(lambda change: on_slider_change(), 'value')
  center_button.on_click(lambda button: center_slider_range())
  # Due to a bug in slider value update, we may need to press a button more than once.
  reset_button.on_click(lambda button: reset_parameters())
  randomize_button.on_click(lambda button: randomize_parameters())
  fit_button.on_click(lambda button: fit_neural_network())

  reset_parameters()
  selectors = ipywidgets.HBox([target_selector, activation_selector])
  layers = [ipywidgets.Label('Weight and bias parameters in each network layer:')]
  for layer_index in range(len(LAYER_SIZES)):
    layer_sliders = [slider for key, slider in sliders.items() if int(key[1]) == layer_index]
    layers.append(ipywidgets.HBox([ipywidgets.Label(f'Layer {layer_index}:')] + layer_sliders))
  buttons = ipywidgets.HBox([center_button, reset_button, randomize_button, fit_button])
  rows = [selectors, *layers, buttons, output, textarea]
  IPython.display.display(ipywidgets.VBox(rows))


initialize_ui()

# %% [markdown]
# # End

# %% [markdown]
# <!-- For Emacs:
# Local Variables:
# fill-column: 100
# End:
# -->
