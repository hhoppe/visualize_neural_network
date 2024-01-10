# %% [markdown]
# # Visualize a 1D neural network


# %%
def visualize_neural_network3(show_sliders=True, verbose=False, r=10):
  """Optimization and interactive control of a 1D -> 1D neural network function."""
  import contextlib
  import math
  import IPython.display
  import ipywidgets
  import matplotlib.pyplot as plt
  import numpy as np
  import torch

  def relu(x):
    return torch.nn.functional.relu(x) if isinstance(x, torch.Tensor) else np.maximum(0, x)

  def identity(x):
    return x

  def sigmoid(x):
    return torch.sigmoid(x) if isinstance(x, torch.Tensor) else 1 / (1 + np.exp(-x))

  # The dictionary `params` contains weights 'w{layer}{curr_index}{prev_index}' and
  # biases 'b{layer}{index}', as either float values or (scalar) torch tensors.
  def neural_network(x, layer_sizes, params):
    prev_nodes = [x]
    for layer_index, layer_size in enumerate(layer_sizes):
      is_last_layer = layer_index == len(layer_sizes) - 1
      activation = relu if not is_last_layer else identity if True else sigmoid
      nodes = []
      for node_index in range(layer_size):
        linear = sum(
            params[f'w{layer_index}{node_index}{prev_index}'] * prev_node
            for prev_index, prev_node in enumerate(prev_nodes)
        )
        node = activation(linear + params[f'b{layer_index}{node_index}'])
        nodes.append(node)
      prev_nodes = nodes
    (y,) = prev_nodes  # The last neural network layer should have a single node.
    return y

  # layer_sizes = 3, 1
  layer_sizes = 2, 2, 1
  # layer_sizes = 3, 3, 1
  # layer_sizes = 4, 4, 1
  # layer_sizes = 4, 4, 4, 1  # It is best to set show_sliders=False.
  initial_values: dict[str, float] = {}
  if layer_sizes == (3, 1):
    initial_values |= dict(w000=1.3, b00=-0.2, w010=3.0, b01=-2.1, w020=2.5, b02=-1.0)
    initial_values |= dict(w100=7.0, w101=4.0, w102=-5.8, b10=-1.0)
  elif layer_sizes == (2, 2, 1):
    initial_values |= dict(w000=-4.74, b00=2.05, w010=8.98, b01=0.14)
    initial_values |= dict(w100=-1.88, w101=-0.43, b10=2.84, w110=0.19, w111=-0.66, b11=1.97)
    initial_values |= dict(w200=-1.62, w201=-0.83, b20=0.66)
  else:
    rng = np.random.default_rng(0)  # Deterministic.
    prev_size = 1
    for layer_index, layer_size in enumerate(layer_sizes):
      for node_index in range(layer_size):
        for prev_index in range(prev_size):
          initial_values[f'w{layer_index}{node_index}{prev_index}'] = rng.random() * 2 - 1
        initial_values[f'b{layer_index}{node_index}'] = rng.random() * 2 - 1
      prev_size = layer_size
    assert prev_size == 1
  params = initial_values.copy()

  targets = {
      'sine': lambda x: np.sin(math.tau * x),
      'cosine': lambda x: np.cos(math.tau * x),
      'triangle': lambda x: 1 - 2 * np.abs(x - 0.5),
      'half-sine': lambda x: np.sin(math.tau / 2 * x),
  }

  target_selector = ipywidgets.RadioButtons(
      options=list(targets), value='sine', description='Target function:'
  )
  a = dict(step=0.01, continuous_update=True)
  sliders = {}
  if show_sliders:
    sliders = {
        key: ipywidgets.FloatSlider(value=value, min=value - r, max=value + r, description=key, **a)
        for key, value in params.items()
    }
  center_button = ipywidgets.Button(description='Center slider ranges')
  reset_button = ipywidgets.Button(description='Reset params')
  randomize_button = ipywidgets.Button(description='Randomize params')
  fit_button = ipywidgets.Button(description='Fit target')
  output = ipywidgets.Output()
  textarea = ipywidgets.Textarea(rows=len(layer_sizes), layout={'width': '600px'}, disabled=True)

  def update_plot():
    text_lines = []
    for layer_index in range(len(layer_sizes)):
      layer_str = ', '.join(f'{k}={v:.2f}' for k, v in params.items() if int(k[1]) == layer_index)
      text_lines.append(f'dict({layer_str})')
    textarea.value = '\n'.join(text_lines)
    with output:
      output.clear_output(wait=True)
      _, ax = plt.subplots(figsize=(10, 5))
      x = np.linspace(0, 1, 100)
      ax.plot(x, neural_network(x, layer_sizes, params), label='neural network')
      ax.plot(x, targets[target_selector.value](x), lw=0.5, label='target')
      ax.set(xlabel='x', ylabel='y=f(x)', ylim=(-2, 2))
      ax.grid(True)
      ax.legend(loc='upper right')
      plt.show()

  def update_params(new_params, center_sliders=True):
    nonlocal params
    params = new_params
    if show_sliders:
      with contextlib.ExitStack() as stack:
        for slider in reversed(sliders.values()):
          stack.enter_context(slider.hold_trait_notifications())
        for key, slider in sliders.items():
          value = params[key]
          slider.value = value  # Bug: https://github.com/jupyter-widgets/ipywidgets/issues/3824.
          if center_sliders:
            slider.min = value - r
            slider.max = value + r
    update_plot()

  def on_slider_change():
    update_params({key: slider.value for key, slider in sliders.items()}, center_sliders=False)

  def center_slider_range():
    for slider in sliders.values():
      value = slider.value
      slider.min, slider.max = value - r, value + r

  def reset_parameters():
    update_params(initial_values.copy())

  def randomize_parameters():
    rng = np.random.default_rng()  # Non-deterministic.
    update_params({key: rng.random() * 2 - 1 for key in params})

  def fit_neural_network(num_steps=1500):
    tensor_params = {
        key: torch.tensor([value], dtype=torch.float32, requires_grad=True)
        for key, value in params.items()
    }
    variables = list(tensor_params.values())
    x_train = torch.linspace(0, 1, 100, dtype=torch.float32)
    y_train = targets[target_selector.value](x_train)

    learning_rate = 0.01
    optimizer = torch.optim.Adam(variables, lr=learning_rate)
    for step in range(num_steps):
      optimizer.zero_grad()
      y_pred = neural_network(x_train, layer_sizes, tensor_params)
      loss = torch.mean((y_train - y_pred) ** 2)
      loss.backward()
      optimizer.step()
      if verbose and step % 100 == 0:
        print(f'Step {step}, Loss: {loss.item()}')  # See messages using "log" icon.

    update_params({key: tensor.detach().numpy().item() for key, tensor in tensor_params.items()})

  target_selector.observe(lambda change: update_plot(), names='value')
  for slider in sliders.values():
    slider.observe(lambda change: on_slider_change(), 'value')
  center_button.on_click(lambda button: center_slider_range())
  # Due to a bug in slider value update, we may need to press a button more than once.
  reset_button.on_click(lambda button: reset_parameters())
  randomize_button.on_click(lambda button: randomize_parameters())
  fit_button.on_click(lambda button: fit_neural_network())

  reset_parameters()
  layers = [ipywidgets.Label('Weight and bias parameters in each network layer:')]
  for layer_index in range(len(layer_sizes)):
    layer_sliders = [slider for key, slider in sliders.items() if int(key[1]) == layer_index]
    layers.append(ipywidgets.HBox([ipywidgets.Label(f'Layer {layer_index}:')] + layer_sliders))
  buttons = ipywidgets.HBox([center_button, reset_button, randomize_button, fit_button])
  rows = [target_selector, *layers, buttons, output, textarea]
  IPython.display.display(ipywidgets.VBox(rows))


visualize_neural_network3()

# %% [markdown]
# # End

# %% [markdown]
# <!-- For Emacs:
# Local Variables:
# fill-column: 100
# End:
# -->
