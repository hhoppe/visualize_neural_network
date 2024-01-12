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
import networkx
import numpy as np
import torch

# %%
SHOW_SLIDERS = True
VERBOSE = False
R = 10

# %%
# LAYER_SIZES = 1, 3, 1
LAYER_SIZES = 1, 2, 2, 1
# LAYER_SIZES = 1, 3, 3, 1
# LAYER_SIZES = 1, 4, 4, 1
# LAYER_SIZES = 1, 4, 4, 4, 1  # It is best to set SHOW_SLIDERS=False.

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
def neural_network(x, params, output_layer_index=10**9):
  assert len(x) == LAYER_SIZES[0]
  prev_nodes = x
  for layer_index, layer_size in enumerate(LAYER_SIZES[1 : output_layer_index + 1], 1):
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
  return prev_nodes


# %% [markdown]
# ## Params initialization


# %%
def get_reset_values():
  d: dict[str, float] = {}

  if LAYER_SIZES == (1, 3, 1):
    d |= dict(w100=1.3, b10=-0.2, w110=3.0, b11=-2.1, w120=2.5, b12=-1.0)
    d |= dict(w200=7.0, w201=4.0, w202=-5.8, b20=-1.0)

  elif LAYER_SIZES == (1, 2, 2, 1):
    d |= dict(w100=-3.45, b10=2.38, w110=6.88, b11=-2.76)
    d |= dict(w200=-1.40, w201=-0.59, b20=2.03, w210=-0.78, w211=-0.50, b21=1.48)
    d |= dict(w300=1.93, w301=-1.15, b30=0.17)

  else:
    rng = np.random.default_rng(0)  # Deterministic.
    for layer_index, layer_size in enumerate(LAYER_SIZES[1:], 1):
      for node_index in range(layer_size):
        for prev_index in range(LAYER_SIZES[layer_index - 1]):
          d[f'w{layer_index}{node_index}{prev_index}'] = rng.random() * 2 - 1
        d[f'b{layer_index}{node_index}'] = rng.random() * 2 - 1

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
    (y_pred,) = neural_network([x_train], tensor_params)
    loss = torch.mean((y_train - y_pred) ** 2)
    loss.backward()
    optimizer.step()
    if VERBOSE and step % 100 == 0:
      print(f'Step {step}, Loss: {loss.item()}')  # See messages using "log" icon.

  update_params({key: tensor.detach().numpy().item() for key, tensor in tensor_params.items()})


# %% [markdown]
# ## User interface


# %%
def display_network_graph():
  graph = networkx.DiGraph()
  pos = {}
  for layer_index, layer_size in enumerate(LAYER_SIZES):
    for node_index in range(layer_size):
      graph.add_node(node := f'n{layer_index}{node_index}')
      pos[node] = layer_index * 6, ((layer_size - 1) / 2 - node_index) * 4
      if layer_index > 0:
        for prev_index in range(LAYER_SIZES[layer_index - 1]):
          label = f'w{layer_index}{node_index}{prev_index}'
          graph.add_edge(f'n{layer_index - 1}{prev_index}', node, label=label)

  _, ax = plt.subplots(figsize=(12, 6), dpi=70)
  networkx.draw(graph, pos, ax=ax, node_size=800, node_color='lightblue', with_labels=True)
  edge_labels = networkx.get_edge_attributes(graph, 'label')
  networkx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=12)
  plt.show()


# %%
def create_sliders():
  a = dict(step=0.01, continuous_update=True)
  return {
      key: ipywidgets.FloatSlider(value=value, min=value - R, max=value + R, description=key, **a)
      for key, value in params.items()
  }


# %%
target_selector = ipywidgets.RadioButtons(
    options=list(TARGETS), value='sine', description='Target function:'
)
activation_selector = ipywidgets.RadioButtons(
    options=list(ACTIVATIONS), value='leaky_relu', description='Activation function:'
)
layer_selector = ipywidgets.RadioButtons(
    options=range(len(LAYER_SIZES)), value=len(LAYER_SIZES[1:]), description='Plot layer:'
)
sliders = create_sliders() if SHOW_SLIDERS else {}
center_button = ipywidgets.Button(description='Center slider ranges')
reset_button = ipywidgets.Button(description='Reset params')
randomize_button = ipywidgets.Button(description='Randomize params')
fit_button = ipywidgets.Button(description='Fit target')
output = ipywidgets.Output()
textarea = ipywidgets.Textarea(rows=len(LAYER_SIZES[1:]), layout={'width': '800px'}, disabled=True)


# %%
def update_plot():
  text_lines = []
  for layer_index in range(1, len(LAYER_SIZES)):
    layer_str = ', '.join(f'{k}={v:.2f}' for k, v in params.items() if int(k[1]) == layer_index)
    text_lines.append(f'dict({layer_str})')
  textarea.value = '\n'.join(text_lines)
  with output:
    output.clear_output(wait=True)
    _, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 1, 100)
    ax.plot(x, TARGETS[target_selector.value](x), lw=0.5, label='target')
    output_layer_index = layer_selector.value
    output_nodes = neural_network([x], params, output_layer_index=output_layer_index)
    for node_index, node in enumerate(output_nodes):
      ax.plot(x, node, label=f'node n{output_layer_index}{node_index}')
    ax.set(xlabel='x', ylabel='y=f(x)', ylim=(-2, 2))
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.show()


# %%
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


# %%
def display_main_ui():
  target_selector.observe(lambda change: update_plot(), names='value')
  activation_selector.observe(lambda change: update_plot(), names='value')
  layer_selector.observe(lambda change: update_plot(), names='value')
  for slider in sliders.values():
    slider.observe(lambda change: on_slider_change(), 'value')
  center_button.on_click(lambda button: center_slider_range())
  # Due to a bug in slider value update, we may need to press a button more than once.
  reset_button.on_click(lambda button: reset_parameters())
  randomize_button.on_click(lambda button: randomize_parameters())
  fit_button.on_click(lambda button: fit_neural_network())

  reset_parameters()
  selectors = ipywidgets.HBox([target_selector, activation_selector, layer_selector])
  layers = [ipywidgets.Label('Weight and bias parameters in each network layer:')]
  for layer_index in range(1, len(LAYER_SIZES)):
    layer_sliders = [slider for key, slider in sliders.items() if int(key[1]) == layer_index]
    layers.append(ipywidgets.HBox([ipywidgets.Label(f'Layer {layer_index}:')] + layer_sliders))
  buttons = ipywidgets.HBox([center_button, reset_button, randomize_button, fit_button])
  rows = [selectors, *layers, buttons, output, textarea]
  IPython.display.display(ipywidgets.VBox(rows))


# %%
display_network_graph()
display_main_ui()

# %% [markdown]
# # End

# %% [markdown]
# <!-- For Emacs:
# Local Variables:
# fill-column: 100
# End:
# -->
