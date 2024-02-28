f:\my_softers\project_IDE\Anconda\envs\jp_layout_pytorch\lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
out-shape:torch.Size([32, 100, 200])
Help on class RNN in module torch.nn.modules.rnn:

class RNN(RNNBase)
 |  RNN(*args, **kwargs)
 |  

 |  .. math::
 |      h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})
 |  
 |  where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
 |  the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
 |  previous layer at time `t-1` or the initial hidden state at time `0`.
 |  If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.
 |  
 |  Args:
 |      input_size: The number of expected features in the input `x`
 |      hidden_size: The number of features in the hidden state `h`
 |      num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
 |          would mean stacking two RNNs together to form a `stacked RNN`,
 |          with the second RNN taking in outputs of the first RNN and
 |          computing the final results. Default: 1
 |      nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
 |      bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
 |          Default: ``True``
 |      batch_first: If ``True``, then the input and output tensors are provided
 |          as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
 |          Note that this does not apply to hidden or cell states. See the
 |          Inputs/Outputs sections below for details.  Default: ``False``
 |      dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
 |          RNN layer except the last layer, with dropout probability equal to
 |          :attr:`dropout`. Default: 0
 |      bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
 |  
 |  Inputs: input, h_0
 |      * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
 |        :math:`(L, N, H_{in})` when ``batch_first=False`` or
 |        :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
 |        the input sequence.  The input can also be a packed variable length sequence.
 |        See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
 |        :func:`torch.nn.utils.rnn.pack_sequence` for details.
 |      * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
 |        :math:`(D * \text{num\_layers}, N, H_{out})` containing the initial hidden
 |        state for the input sequence batch. Defaults to zeros if not provided.
 |  
 |      where:
 |  
 |      .. math::
 |          \begin{aligned}
 |              N ={} & \text{batch size} \\
 |              L ={} & \text{sequence length} \\
 |              D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
 |              H_{in} ={} & \text{input\_size} \\
 |              H_{out} ={} & \text{hidden\_size}
 |          \end{aligned}
 |  
 |  Outputs: output, h_n
 |      * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
 |        :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
 |        :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
 |        `(h_t)` from the last layer of the RNN, for each `t`. If a
 |        :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
 |        will also be a packed sequence.
 |      * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
 |        :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
 |        for each element in the batch.
 |  
 |  Attributes:
 |      weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
 |          of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
 |          `(hidden_size, num_directions * hidden_size)`
 |      weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
 |          of shape `(hidden_size, hidden_size)`
 |      bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
 |          of shape `(hidden_size)`
 |      bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
 |          of shape `(hidden_size)`
 |  
 |  .. note::
 |      All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
 |      where :math:`k = \frac{1}{\text{hidden\_size}}`
 |  
 |  .. note::
 |      For bidirectional RNNs, forward and backward are directions 0 and 1 respectively.
 |      Example of splitting the output layers when ``batch_first=False``:
 |      ``output.view(seq_len, batch, num_directions, hidden_size)``.
 |  
 |  .. note::
 |      ``batch_first`` argument is ignored for unbatched inputs.
 |  
 |  .. include:: ../cudnn_rnn_determinism.rst
 |  
 |  .. include:: ../cudnn_persistent_rnn.rst
 |  
 |  Examples::
 |  
 |      >>> rnn = nn.RNN(10, 20, 2)
 |      >>> input = torch.randn(5, 3, 10)
 |      >>> h0 = torch.randn(2, 3, 20)
 |      >>> output, hn = rnn(input, h0)
 |  
 |  Method resolution order:
 |      RNN
 |      RNNBase
 |      torch.nn.modules.module.Module
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self, *args, **kwargs)
 |      Initializes internal Module state, shared by both nn.Module and ScriptModule.
 |  
 |  forward(self, input, hx=None)
 |      Defines the computation performed at every call.
 |      
 |      Should be overridden by all subclasses.
 |      
 |      .. note::
 |          Although the recipe for forward pass needs to be defined within
 |          this function, one should call the :class:`Module` instance afterwards
 |          instead of this since the former takes care of running the
 |          registered hooks while the latter silently ignores them.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from RNNBase:
 |  
 |  __setattr__(self, attr, value)
 |      Implement setattr(self, name, value).
 |  
 |  __setstate__(self, d)
 |  
 |  check_forward_args(self, input: torch.Tensor, hidden: torch.Tensor, batch_sizes: Union[torch.Tensor, NoneType])
 |  
 |  check_hidden_size(self, hx: torch.Tensor, expected_hidden_size: Tuple[int, int, int], msg: str = 'Expected hidden size {}, got {}') -> None
 |  
 |  check_input(self, input: torch.Tensor, batch_sizes: Union[torch.Tensor, NoneType]) -> None
 |  
 |  extra_repr(self) -> str
 |      Set the extra representation of the module
 |      
 |      To print customized extra information, you should re-implement
 |      this method in your own modules. Both single-line and multi-line
 |      strings are acceptable.
 |  
 |  flatten_parameters(self) -> None
 |      Resets parameter data pointer so that they can use faster code paths.
 |      
 |      Right now, this works only if the module is on the GPU and cuDNN is enabled.
 |      Otherwise, it's a no-op.
 |  
 |  get_expected_hidden_size(self, input: torch.Tensor, batch_sizes: Union[torch.Tensor, NoneType]) -> Tuple[int, int, int]
 |  
 |  permute_hidden(self, hx: torch.Tensor, permutation: Union[torch.Tensor, NoneType])
 |  
 |  reset_parameters(self) -> None
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from RNNBase:
 |  
 |  all_weights
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from RNNBase:
 |  
 |  __annotations__ = {'batch_first': <class 'bool'>, 'bias': <class 'bool...
 |  
 |  __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'b...
 |  
 |  __jit_unused_properties__ = ['all_weights']
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from torch.nn.modules.module.Module:
 |  
 |  __call__ = _call_impl(self, *input, **kwargs)
 |  
 |  __delattr__(self, name)
 |      Implement delattr(self, name).
 |  
 |  __dir__(self)
 |      Default dir() implementation.
 |  
 |  __getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]
 |  
 |  __repr__(self)
 |      Return repr(self).
 |  
 |  add_module(self, name: str, module: Union[ForwardRef('Module'), NoneType]) -> None
 |      Adds a child module to the current module.
 |      
 |      The module can be accessed as an attribute using the given name.
 |      
 |      Args:
 |          name (str): name of the child module. The child module can be
 |              accessed from this module using the given name
 |          module (Module): child module to be added to the module.
 |  
 |  apply(self: ~T, fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T
 |      Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
 |      as well as self. Typical use includes initializing the parameters of a model
 |      (see also :ref:`nn-init-doc`).
 |      
 |      Args:
 |          fn (:class:`Module` -> None): function to be applied to each submodule
 |      
 |      Returns:
 |          Module: self
 |      
 |      Example::
 |      
 |          >>> @torch.no_grad()
 |          >>> def init_weights(m):
 |          >>>     print(m)
 |          >>>     if type(m) == nn.Linear:
 |          >>>         m.weight.fill_(1.0)
 |          >>>         print(m.weight)
 |          >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
 |          >>> net.apply(init_weights)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          Parameter containing:
 |          tensor([[1., 1.],
 |                  [1., 1.]], requires_grad=True)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          Parameter containing:
 |          tensor([[1., 1.],
 |                  [1., 1.]], requires_grad=True)
 |          Sequential(
 |            (0): Linear(in_features=2, out_features=2, bias=True)
 |            (1): Linear(in_features=2, out_features=2, bias=True)
 |          )
 |  
 |  bfloat16(self: ~T) -> ~T
 |      Casts all floating point parameters and buffers to ``bfloat16`` datatype.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Returns:
 |          Module: self
 |  
 |  buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]
 |      Returns an iterator over module buffers.
 |      
 |      Args:
 |          recurse (bool): if True, then yields buffers of this module
 |              and all submodules. Otherwise, yields only buffers that
 |              are direct members of this module.
 |      
 |      Yields:
 |          torch.Tensor: module buffer
 |      
 |      Example::
 |      
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> for buf in model.buffers():
 |          >>>     print(type(buf), buf.size())
 |          <class 'torch.Tensor'> (20L,)
 |          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
 |  
 |  children(self) -> Iterator[ForwardRef('Module')]
 |      Returns an iterator over immediate children modules.
 |      
 |      Yields:
 |          Module: a child module
 |  
 |  cpu(self: ~T) -> ~T
 |      Moves all model parameters and buffers to the CPU.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Returns:
 |          Module: self
 |  
 |  cuda(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T
 |      Moves all model parameters and buffers to the GPU.
 |      
 |      This also makes associated parameters and buffers different objects. So
 |      it should be called before constructing optimizer if the module will
 |      live on GPU while being optimized.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Args:
 |          device (int, optional): if specified, all parameters will be
 |              copied to that device
 |      
 |      Returns:
 |          Module: self
 |  
 |  double(self: ~T) -> ~T
 |      Casts all floating point parameters and buffers to ``double`` datatype.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Returns:
 |          Module: self
 |  
 |  eval(self: ~T) -> ~T
 |      Sets the module in evaluation mode.
 |      
 |      This has any effect only on certain modules. See documentations of
 |      particular modules for details of their behaviors in training/evaluation
 |      mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
 |      etc.
 |      
 |      This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.
 |      
 |      See :ref:`locally-disable-grad-doc` for a comparison between
 |      `.eval()` and several similar mechanisms that may be confused with it.
 |      
 |      Returns:
 |          Module: self
 |  
 |  float(self: ~T) -> ~T
 |      Casts all floating point parameters and buffers to ``float`` datatype.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Returns:
 |          Module: self
 |  
 |  get_buffer(self, target: str) -> 'Tensor'
 |      Returns the buffer given by ``target`` if it exists,
 |      otherwise throws an error.
 |      
 |      See the docstring for ``get_submodule`` for a more detailed
 |      explanation of this method's functionality as well as how to
 |      correctly specify ``target``.
 |      
 |      Args:
 |          target: The fully-qualified string name of the buffer
 |              to look for. (See ``get_submodule`` for how to specify a
 |              fully-qualified string.)
 |      
 |      Returns:
 |          torch.Tensor: The buffer referenced by ``target``
 |      
 |      Raises:
 |          AttributeError: If the target string references an invalid
 |              path or resolves to something that is not a
 |              buffer
 |  
 |  get_extra_state(self) -> Any
 |      Returns any extra state to include in the module's state_dict.
 |      Implement this and a corresponding :func:`set_extra_state` for your module
 |      if you need to store extra state. This function is called when building the
 |      module's `state_dict()`.
 |      
 |      Note that extra state should be pickleable to ensure working serialization
 |      of the state_dict. We only provide provide backwards compatibility guarantees
 |      for serializing Tensors; other objects may break backwards compatibility if
 |      their serialized pickled form changes.
 |      
 |      Returns:
 |          object: Any extra state to store in the module's state_dict
 |  
 |  get_parameter(self, target: str) -> 'Parameter'
 |      Returns the parameter given by ``target`` if it exists,
 |      otherwise throws an error.
 |      
 |      See the docstring for ``get_submodule`` for a more detailed
 |      explanation of this method's functionality as well as how to
 |      correctly specify ``target``.
 |      
 |      Args:
 |          target: The fully-qualified string name of the Parameter
 |              to look for. (See ``get_submodule`` for how to specify a
 |              fully-qualified string.)
 |      
 |      Returns:
 |          torch.nn.Parameter: The Parameter referenced by ``target``
 |      
 |      Raises:
 |          AttributeError: If the target string references an invalid
 |              path or resolves to something that is not an
 |              ``nn.Parameter``
 |  
 |  get_submodule(self, target: str) -> 'Module'
 |      Returns the submodule given by ``target`` if it exists,
 |      otherwise throws an error.
 |      
 |      For example, let's say you have an ``nn.Module`` ``A`` that
 |      looks like this:
 |      
 |      .. code-block:: text
 |      
 |          A(
 |              (net_b): Module(
 |                  (net_c): Module(
 |                      (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
 |                  )
 |                  (linear): Linear(in_features=100, out_features=200, bias=True)
 |              )
 |          )
 |      
 |      (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
 |      submodule ``net_b``, which itself has two submodules ``net_c``
 |      and ``linear``. ``net_c`` then has a submodule ``conv``.)
 |      
 |      To check whether or not we have the ``linear`` submodule, we
 |      would call ``get_submodule("net_b.linear")``. To check whether
 |      we have the ``conv`` submodule, we would call
 |      ``get_submodule("net_b.net_c.conv")``.
 |      
 |      The runtime of ``get_submodule`` is bounded by the degree
 |      of module nesting in ``target``. A query against
 |      ``named_modules`` achieves the same result, but it is O(N) in
 |      the number of transitive modules. So, for a simple check to see
 |      if some submodule exists, ``get_submodule`` should always be
 |      used.
 |      
 |      Args:
 |          target: The fully-qualified string name of the submodule
 |              to look for. (See above example for how to specify a
 |              fully-qualified string.)
 |      
 |      Returns:
 |          torch.nn.Module: The submodule referenced by ``target``
 |      
 |      Raises:
 |          AttributeError: If the target string references an invalid
 |              path or resolves to something that is not an
 |              ``nn.Module``
 |  
 |  half(self: ~T) -> ~T
 |      Casts all floating point parameters and buffers to ``half`` datatype.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Returns:
 |          Module: self
 |  
 |  ipu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T
 |      Moves all model parameters and buffers to the IPU.
 |      
 |      This also makes associated parameters and buffers different objects. So
 |      it should be called before constructing optimizer if the module will
 |      live on IPU while being optimized.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Arguments:
 |          device (int, optional): if specified, all parameters will be
 |              copied to that device
 |      
 |      Returns:
 |          Module: self
 |  
 |  load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True)
 |      Copies parameters and buffers from :attr:`state_dict` into
 |      this module and its descendants. If :attr:`strict` is ``True``, then
 |      the keys of :attr:`state_dict` must exactly match the keys returned
 |      by this module's :meth:`~torch.nn.Module.state_dict` function.
 |      
 |      Args:
 |          state_dict (dict): a dict containing parameters and
 |              persistent buffers.
 |          strict (bool, optional): whether to strictly enforce that the keys
 |              in :attr:`state_dict` match the keys returned by this module's
 |              :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
 |      
 |      Returns:
 |          ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
 |              * **missing_keys** is a list of str containing the missing keys
 |              * **unexpected_keys** is a list of str containing the unexpected keys
 |      
 |      Note:
 |          If a parameter or buffer is registered as ``None`` and its corresponding key
 |          exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
 |          ``RuntimeError``.
 |  
 |  modules(self) -> Iterator[ForwardRef('Module')]
 |      Returns an iterator over all modules in the network.
 |      
 |      Yields:
 |          Module: a module in the network
 |      
 |      Note:
 |          Duplicate modules are returned only once. In the following
 |          example, ``l`` will be returned only once.
 |      
 |      Example::
 |      
 |          >>> l = nn.Linear(2, 2)
 |          >>> net = nn.Sequential(l, l)
 |          >>> for idx, m in enumerate(net.modules()):
 |          ...     print(idx, '->', m)
 |      
 |          0 -> Sequential(
 |            (0): Linear(in_features=2, out_features=2, bias=True)
 |            (1): Linear(in_features=2, out_features=2, bias=True)
 |          )
 |          1 -> Linear(in_features=2, out_features=2, bias=True)
 |  
 |  named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]
 |      Returns an iterator over module buffers, yielding both the
 |      name of the buffer as well as the buffer itself.
 |      
 |      Args:
 |          prefix (str): prefix to prepend to all buffer names.
 |          recurse (bool): if True, then yields buffers of this module
 |              and all submodules. Otherwise, yields only buffers that
 |              are direct members of this module.
 |      
 |      Yields:
 |          (str, torch.Tensor): Tuple containing the name and buffer
 |      
 |      Example::
 |      
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> for name, buf in self.named_buffers():
 |          >>>    if name in ['running_var']:
 |          >>>        print(buf.size())
 |  
 |  named_children(self) -> Iterator[Tuple[str, ForwardRef('Module')]]
 |      Returns an iterator over immediate children modules, yielding both
 |      the name of the module as well as the module itself.
 |      
 |      Yields:
 |          (str, Module): Tuple containing a name and child module
 |      
 |      Example::
 |      
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> for name, module in model.named_children():
 |          >>>     if name in ['conv4', 'conv5']:
 |          >>>         print(module)
 |  
 |  named_modules(self, memo: Union[Set[ForwardRef('Module')], NoneType] = None, prefix: str = '', remove_duplicate: bool = True)
 |      Returns an iterator over all modules in the network, yielding
 |      both the name of the module as well as the module itself.
 |      
 |      Args:
 |          memo: a memo to store the set of modules already added to the result
 |          prefix: a prefix that will be added to the name of the module
 |          remove_duplicate: whether to remove the duplicated module instances in the result
 |              or not
 |      
 |      Yields:
 |          (str, Module): Tuple of name and module
 |      
 |      Note:
 |          Duplicate modules are returned only once. In the following
 |          example, ``l`` will be returned only once.
 |      
 |      Example::
 |      
 |          >>> l = nn.Linear(2, 2)
 |          >>> net = nn.Sequential(l, l)
 |          >>> for idx, m in enumerate(net.named_modules()):
 |          ...     print(idx, '->', m)
 |      
 |          0 -> ('', Sequential(
 |            (0): Linear(in_features=2, out_features=2, bias=True)
 |            (1): Linear(in_features=2, out_features=2, bias=True)
 |          ))
 |          1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
 |  
 |  named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.parameter.Parameter]]
 |      Returns an iterator over module parameters, yielding both the
 |      name of the parameter as well as the parameter itself.
 |      
 |      Args:
 |          prefix (str): prefix to prepend to all parameter names.
 |          recurse (bool): if True, then yields parameters of this module
 |              and all submodules. Otherwise, yields only parameters that
 |              are direct members of this module.
 |      
 |      Yields:
 |          (str, Parameter): Tuple containing the name and parameter
 |      
 |      Example::
 |      
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> for name, param in self.named_parameters():
 |          >>>    if name in ['bias']:
 |          >>>        print(param.size())
 |  
 |  parameters(self, recurse: bool = True) -> Iterator[torch.nn.parameter.Parameter]
 |      Returns an iterator over module parameters.
 |      
 |      This is typically passed to an optimizer.
 |      
 |      Args:
 |          recurse (bool): if True, then yields parameters of this module
 |              and all submodules. Otherwise, yields only parameters that
 |              are direct members of this module.
 |      
 |      Yields:
 |          Parameter: module parameter
 |      
 |      Example::
 |      
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> for param in model.parameters():
 |          >>>     print(type(param), param.size())
 |          <class 'torch.Tensor'> (20L,)
 |          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
 |  
 |  register_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, torch.Tensor]]) -> torch.utils.hooks.RemovableHandle
 |      Registers a backward hook on the module.
 |      
 |      This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
 |      the behavior of this function will change in future versions.
 |      
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |  
 |  register_buffer(self, name: str, tensor: Union[torch.Tensor, NoneType], persistent: bool = True) -> None
 |      Adds a buffer to the module.
 |      
 |      This is typically used to register a buffer that should not to be
 |      considered a model parameter. For example, BatchNorm's ``running_mean``
 |      is not a parameter, but is part of the module's state. Buffers, by
 |      default, are persistent and will be saved alongside parameters. This
 |      behavior can be changed by setting :attr:`persistent` to ``False``. The
 |      only difference between a persistent buffer and a non-persistent buffer
 |      is that the latter will not be a part of this module's
 |      :attr:`state_dict`.
 |      
 |      Buffers can be accessed as attributes using given names.
 |      
 |      Args:
 |          name (str): name of the buffer. The buffer can be accessed
 |              from this module using the given name
 |          tensor (Tensor or None): buffer to be registered. If ``None``, then operations
 |              that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
 |              the buffer is **not** included in the module's :attr:`state_dict`.
 |          persistent (bool): whether the buffer is part of this module's
 |              :attr:`state_dict`.
 |      
 |      Example::
 |      
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> self.register_buffer('running_mean', torch.zeros(num_features))
 |  
 |  register_forward_hook(self, hook: Callable[..., NoneType]) -> torch.utils.hooks.RemovableHandle
 |      Registers a forward hook on the module.
 |      
 |      The hook will be called every time after :func:`forward` has computed an output.
 |      It should have the following signature::
 |      
 |          hook(module, input, output) -> None or modified output
 |      
 |      The input contains only the positional arguments given to the module.
 |      Keyword arguments won't be passed to the hooks and only to the ``forward``.
 |      The hook can modify the output. It can modify the input inplace but
 |      it will not have effect on forward since this is called after
 |      :func:`forward` is called.
 |      
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |  
 |  register_forward_pre_hook(self, hook: Callable[..., NoneType]) -> torch.utils.hooks.RemovableHandle
 |      Registers a forward pre-hook on the module.
 |      
 |      The hook will be called every time before :func:`forward` is invoked.
 |      It should have the following signature::
 |      
 |          hook(module, input) -> None or modified input
 |      
 |      The input contains only the positional arguments given to the module.
 |      Keyword arguments won't be passed to the hooks and only to the ``forward``.
 |      The hook can modify the input. User can either return a tuple or a
 |      single modified value in the hook. We will wrap the value into a tuple
 |      if a single value is returned(unless that value is already a tuple).
 |      
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |  
 |  register_full_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, torch.Tensor]]) -> torch.utils.hooks.RemovableHandle
 |      Registers a backward hook on the module.
 |      
 |      The hook will be called every time the gradients with respect to a module
 |      are computed, i.e. the hook will execute if and only if the gradients with
 |      respect to module outputs are computed. The hook should have the following
 |      signature::
 |      
 |          hook(module, grad_input, grad_output) -> tuple(Tensor) or None
 |      
 |      The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
 |      with respect to the inputs and outputs respectively. The hook should
 |      not modify its arguments, but it can optionally return a new gradient with
 |      respect to the input that will be used in place of :attr:`grad_input` in
 |      subsequent computations. :attr:`grad_input` will only correspond to the inputs given
 |      as positional arguments and all kwarg arguments are ignored. Entries
 |      in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
 |      arguments.
 |      
 |      For technical reasons, when this hook is applied to a Module, its forward function will
 |      receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
 |      of each Tensor returned by the Module's forward function.
 |      
 |      .. warning ::
 |          Modifying inputs or outputs inplace is not allowed when using backward hooks and
 |          will raise an error.
 |      
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |  
 |  register_load_state_dict_post_hook(self, hook)
 |      Registers a post hook to be run after module's ``load_state_dict``
 |      is called.
 |      
 |      It should have the following signature::
 |          hook(module, incompatible_keys) -> None
 |      
 |      The ``module`` argument is the current module that this hook is registered
 |      on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
 |      of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
 |      is a ``list`` of ``str`` containing the missing keys and
 |      ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.
 |      
 |      The given incompatible_keys can be modified inplace if needed.
 |      
 |      Note that the checks performed when calling :func:`load_state_dict` with
 |      ``strict=True`` are affected by modifications the hook makes to
 |      ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
 |      set of keys will result in an error being thrown when ``strict=True``, and
 |      clearning out both missing and unexpected keys will avoid an error.
 |      
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |  
 |  register_module(self, name: str, module: Union[ForwardRef('Module'), NoneType]) -> None
 |      Alias for :func:`add_module`.
 |  
 |  register_parameter(self, name: str, param: Union[torch.nn.parameter.Parameter, NoneType]) -> None
 |      Adds a parameter to the module.
 |      
 |      The parameter can be accessed as an attribute using given name.
 |      
 |      Args:
 |          name (str): name of the parameter. The parameter can be accessed
 |              from this module using the given name
 |          param (Parameter or None): parameter to be added to the module. If
 |              ``None``, then operations that run on parameters, such as :attr:`cuda`,
 |              are ignored. If ``None``, the parameter is **not** included in the
 |              module's :attr:`state_dict`.
 |  
 |  requires_grad_(self: ~T, requires_grad: bool = True) -> ~T
 |      Change if autograd should record operations on parameters in this
 |      module.
 |      
 |      This method sets the parameters' :attr:`requires_grad` attributes
 |      in-place.
 |      
 |      This method is helpful for freezing part of the module for finetuning
 |      or training parts of a model individually (e.g., GAN training).
 |      
 |      See :ref:`locally-disable-grad-doc` for a comparison between
 |      `.requires_grad_()` and several similar mechanisms that may be confused with it.
 |      
 |      Args:
 |          requires_grad (bool): whether autograd should record operations on
 |                                parameters in this module. Default: ``True``.
 |      
 |      Returns:
 |          Module: self
 |  
 |  set_extra_state(self, state: Any)
 |      This function is called from :func:`load_state_dict` to handle any extra state
 |      found within the `state_dict`. Implement this function and a corresponding
 |      :func:`get_extra_state` for your module if you need to store extra state within its
 |      `state_dict`.
 |      
 |      Args:
 |          state (dict): Extra state from the `state_dict`
 |  
 |  share_memory(self: ~T) -> ~T
 |      See :meth:`torch.Tensor.share_memory_`
 |  
 |  state_dict(self, *args, destination=None, prefix='', keep_vars=False)
 |      Returns a dictionary containing references to the whole state of the module.
 |      
 |      Both parameters and persistent buffers (e.g. running averages) are
 |      included. Keys are corresponding parameter and buffer names.
 |      Parameters and buffers set to ``None`` are not included.
 |      
 |      .. note::
 |          The returned object is a shallow copy. It contains references
 |          to the module's parameters and buffers.
 |      
 |      .. warning::
 |          Currently ``state_dict()`` also accepts positional arguments for
 |          ``destination``, ``prefix`` and ``keep_vars`` in order. However,
 |          this is being deprecated and keyword arguments will be enforced in
 |          future releases.
 |      
 |      .. warning::
 |          Please avoid the use of argument ``destination`` as it is not
 |          designed for end-users.
 |      
 |      Args:
 |          destination (dict, optional): If provided, the state of module will
 |              be updated into the dict and the same object is returned.
 |              Otherwise, an ``OrderedDict`` will be created and returned.
 |              Default: ``None``.
 |          prefix (str, optional): a prefix added to parameter and buffer
 |              names to compose the keys in state_dict. Default: ``''``.
 |          keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
 |              returned in the state dict are detached from autograd. If it's
 |              set to ``True``, detaching will not be performed.
 |              Default: ``False``.
 |      
 |      Returns:
 |          dict:
 |              a dictionary containing a whole state of the module
 |      
 |      Example::
 |      
 |          >>> # xdoctest: +SKIP("undefined vars")
 |          >>> module.state_dict().keys()
 |          ['bias', 'weight']
 |  
 |  to(self, *args, **kwargs)
 |      Moves and/or casts the parameters and buffers.
 |      
 |      This can be called as
 |      
 |      .. function:: to(device=None, dtype=None, non_blocking=False)
 |         :noindex:
 |      
 |      .. function:: to(dtype, non_blocking=False)
 |         :noindex:
 |      
 |      .. function:: to(tensor, non_blocking=False)
 |         :noindex:
 |      
 |      .. function:: to(memory_format=torch.channels_last)
 |         :noindex:
 |      
 |      Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
 |      floating point or complex :attr:`dtype`\ s. In addition, this method will
 |      only cast the floating point or complex parameters and buffers to :attr:`dtype`
 |      (if given). The integral parameters and buffers will be moved
 |      :attr:`device`, if that is given, but with dtypes unchanged. When
 |      :attr:`non_blocking` is set, it tries to convert/move asynchronously
 |      with respect to the host if possible, e.g., moving CPU Tensors with
 |      pinned memory to CUDA devices.
 |      
 |      See below for examples.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Args:
 |          device (:class:`torch.device`): the desired device of the parameters
 |              and buffers in this module
 |          dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
 |              the parameters and buffers in this module
 |          tensor (torch.Tensor): Tensor whose dtype and device are the desired
 |              dtype and device for all parameters and buffers in this module
 |          memory_format (:class:`torch.memory_format`): the desired memory
 |              format for 4D parameters and buffers in this module (keyword
 |              only argument)
 |      
 |      Returns:
 |          Module: self
 |      
 |      Examples::
 |      
 |          >>> # xdoctest: +IGNORE_WANT("non-deterministic")
 |          >>> linear = nn.Linear(2, 2)
 |          >>> linear.weight
 |          Parameter containing:
 |          tensor([[ 0.1913, -0.3420],
 |                  [-0.5113, -0.2325]])
 |          >>> linear.to(torch.double)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          >>> linear.weight
 |          Parameter containing:
 |          tensor([[ 0.1913, -0.3420],
 |                  [-0.5113, -0.2325]], dtype=torch.float64)
 |          >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
 |          >>> gpu1 = torch.device("cuda:1")
 |          >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          >>> linear.weight
 |          Parameter containing:
 |          tensor([[ 0.1914, -0.3420],
 |                  [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
 |          >>> cpu = torch.device("cpu")
 |          >>> linear.to(cpu)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          >>> linear.weight
 |          Parameter containing:
 |          tensor([[ 0.1914, -0.3420],
 |                  [-0.5112, -0.2324]], dtype=torch.float16)
 |      
 |          >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
 |          >>> linear.weight
 |          Parameter containing:
 |          tensor([[ 0.3741+0.j,  0.2382+0.j],
 |                  [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
 |          >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
 |          tensor([[0.6122+0.j, 0.1150+0.j],
 |                  [0.6122+0.j, 0.1150+0.j],
 |                  [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
 |  
 |  to_empty(self: ~T, *, device: Union[str, torch.device]) -> ~T
 |      Moves the parameters and buffers to the specified device without copying storage.
 |      
 |      Args:
 |          device (:class:`torch.device`): The desired device of the parameters
 |              and buffers in this module.
 |      
 |      Returns:
 |          Module: self
 |  
 |  train(self: ~T, mode: bool = True) -> ~T
 |      Sets the module in training mode.
 |      
 |      This has any effect only on certain modules. See documentations of
 |      particular modules for details of their behaviors in training/evaluation
 |      mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
 |      etc.
 |      
 |      Args:
 |          mode (bool): whether to set training mode (``True``) or evaluation
 |                       mode (``False``). Default: ``True``.
 |      
 |      Returns:
 |          Module: self
 |  
 |  type(self: ~T, dst_type: Union[torch.dtype, str]) -> ~T
 |      Casts all parameters and buffers to :attr:`dst_type`.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Args:
 |          dst_type (type or string): the desired type
 |      
 |      Returns:
 |          Module: self
 |  
 |  xpu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T
 |      Moves all model parameters and buffers to the XPU.
 |      
 |      This also makes associated parameters and buffers different objects. So
 |      it should be called before constructing optimizer if the module will
 |      live on XPU while being optimized.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Arguments:
 |          device (int, optional): if specified, all parameters will be
 |              copied to that device
 |      
 |      Returns:
 |          Module: self
 |  
 |  zero_grad(self, set_to_none: bool = False) -> None
 |      Sets gradients of all model parameters to zero. See similar function
 |      under :class:`torch.optim.Optimizer` for more context.
 |      
 |      Args:
 |          set_to_none (bool): instead of setting to zero, set the grads to None.
 |              See :meth:`torch.optim.Optimizer.zero_grad` for details.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from torch.nn.modules.module.Module:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from torch.nn.modules.module.Module:
 |  
 |  T_destination = ~T_destination
 |  
 |  dump_patches = False
