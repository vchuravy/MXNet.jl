#=doc
Models
======

The model API provides convenient high-level interface to do training and predicting on
a network described using the symbolic API.
=#

#=doc
.. class:: AbstractModel

   The abstract super type of all models in MXNet.jl.
=#
abstract AbstractModel

#=doc
.. class:: FeedForward

   The feedforward model provides convenient interface to train and predict on
   feedforward architectures like multi-layer MLP, ConvNets, etc. There is no
   explicitly handling of *time index*, but it is relatively easy to implement
   unrolled RNN / LSTM under this framework (**TODO**: add example). For models
   that handles sequential data explicitly, please use **TODO**...
=#
type FeedForward <: AbstractModel
  arch        :: SymbolicNode
  ctx         :: Vector{Context}

  arg_params  :: Dict{Base.Symbol, NDArray}
  aux_params  :: Dict{Base.Symbol, NDArray}

  pred_exec   :: Union{Executor, Void}

  # leave the rest fields undefined
  FeedForward(arch :: SymbolicNode, ctx :: Vector{Context}) = new(arch, ctx)
end

"""Get a split of `batch_size` into `n_split` pieces for data parallelization. Returns a vector
    of length `n_split`, with each entry a `UnitRange{Int}` indicating the slice index for that
    piece.
"""
function _split_inputs(batch_size :: Int, n_split :: Int)
  @assert(batch_size >= n_split)
  per_split = floor(Int, batch_size / n_split)
  counts    = Base.zeros(Int, n_split)+per_split
  extra     = batch_size - sum(counts)
  counts[1:extra] += 1

  cum = [0, cumsum(counts)...]
  idx = [cum[i-1]+1:cum[i] for i = 2:length(cum)]
  return idx
end

#=doc
.. function:: FeedForward(arch :: SymbolicNode, ctx)

   :param arch: the architecture of the network constructed using the symbolic API.
   :param ctx: the devices on which this model should do computation. It could be a single :class:`Context`
               or a list of :class:`Context` objects. In the latter case, data parallelization will be used
               for training. If no context is provided, the default context ``cpu()`` will be used.
=#
function FeedForward(arch :: SymbolicNode; context :: Union{Context, Vector{Context}, Void} = nothing)
  if isa(context, Void)
    context = [Context(CPU)]
  elseif isa(context, Context)
    context = [context]
  end
  FeedForward(arch, context)
end

#=doc
.. function:: init_model(self, initializer; overwrite=false, input_shapes...)

   Initialize the weights in the model.

   This method will be called automatically when training a model. So there is usually no
   need to call this method unless one needs to inspect a model with only randomly initialized
   weights.

   :param FeedForward self: the model to be initialized.
   :param AbstractInitializer initializer: an initializer describing how the weights should be initialized.
   :param Bool overwrite: keyword argument, force initialization even when weights already exists.
   :param input_shapes: the shape of all data and label inputs to this model, given as keyword arguments.
                        For example, ``data=(28,28,1,100), label=(100,)``.
=#
function init_model(self :: FeedForward, initializer :: Union{AbstractInitializer, Dict}; overwrite::Bool=false, input_shapes...)
  # all arg names, including data, label, and parameters
  arg_names    = list_arguments(self.arch)

  input_names  = [x[1] for x in input_shapes]

  param_names = setdiff(arg_names, input_names)
  aux_names   = list_auxiliary_states(self.arch)

  arg_defined = true
  aux_defined = true

  arg_shapes, out_shapes, aux_shapes = infer_shape(self.arch; input_shapes...)
  if !isdefined(self, :arg_params)
    param_name_shapes = filter(x -> in(x[1],param_names), zip(arg_names, arg_shapes))
    self.arg_params = Dict([name => empty(shape) for (name,shape) in param_name_shapes])
    arg_defined = false
  end
  if !isdefined(self, :aux_params)
    self.aux_params = Dict([name => empty(shape) for (name,shape) in zip(aux_names,aux_shapes)])
    aux_defined = false
  end

  # initialize the contents of the parameters
  if !arg_defined || overwrite
    _initialize!(initializer, self.arg_params)
  end

  if !aux_defined || overwrite
    _initialize!(initializer, self.aux_params)
  end

  return (arg_names, param_names, aux_names)
end

function _initialize!(initializer :: AbstractInitializer, params)
  for (k, v) in params
    initializer(k, v)
  end
end

function _initialize!(initializer :: Dict, params)
  for (k, v) in params
    if haskey(initializer, k)
      initializer[k](k, v)
    else
      initializer[:default](k, v)
    end
  end
end

function _setup_predictor(self :: FeedForward, overwrite :: Bool=false; data_shapes...)
  if !isdefined(self, :pred_exec) || isa(self.pred_exec, Void) || overwrite
    if !isdefined(self, :arg_params) || !isdefined(self, :aux_params)
      @assert(false, "Model weights not defined, please init or train the model, or load from file")
    end

    # the predictor use only the first device
    self.pred_exec = simple_bind(self.arch, self.ctx[1]; grad_req=GRAD_NOP, data_shapes...)
    copy_params_from(self.pred_exec, self.arg_params, self.aux_params)
  else
    # make sure the new setup is compatible with the existing one
    for (d_name, d_shape) in data_shapes
      @assert(d_shape == size(self.pred_exec.arg_dict[d_name]),
              "Shape of $d_name mismatch with existing predictor, use overwrite=true overwrite existing predictor")
    end
  end
end

#=doc
.. function::
   predict(self, data; overwrite=false, callback=nothing)

   Predict using an existing model. The model should be already initialized, or trained or loaded from
   a checkpoint. There is an overloaded function that allows to pass the callback as the first argument,
   so it is possible to do

   .. code-block:: julia

      predict(model, data) do batch_output
        # consume or write batch_output to file
      end

   :param FeedForward self: the model.
   :param AbstractDataProvider data: the data to perform prediction on.
   :param Bool overwrite: an :class:`Executor` is initialized the first time predict is called. The memory
                          allocation of the :class:`Executor` depends on the mini-batch size of the test
                          data provider. If you call predict twice with data provider of the same batch-size,
                          then the executor can be potentially be re-used. So, if ``overwrite`` is false,
                          we will try to re-use, and raise an error if batch-size changed. If ``overwrite``
                          is true (the default), a new :class:`Executor` will be created to replace the old one.

   .. note::

      Prediction is computationally much less costly than training, so the bottleneck sometimes becomes the IO
      for copying mini-batches of data. Since there is no concern about convergence in prediction, it is better
      to set the mini-batch size as large as possible (limited by your device memory) if prediction speed is a
      concern.

      For the same reason, currently prediction will only use the first device even if multiple devices are
      provided to construct the model.

   .. note::

      If you perform further after prediction. The weights are not automatically synchronized if ``overwrite``
      is set to false and the old predictor is re-used. In this case
      setting ``overwrite`` to true (the default) will re-initialize the predictor the next time you call
      predict and synchronize the weights again.

   :seealso: :func:`train`, :func:`fit`, :func:`init_model`, :func:`load_checkpoint`
=#
function predict(callback :: Function, self :: FeedForward, data :: AbstractDataProvider; overwrite :: Bool = true)
  predict(self, data; overwrite = overwrite, callback=callback)
end
function predict(self :: FeedForward, data :: AbstractDataProvider; overwrite::Bool=true, callback::Union{Function,Void}=nothing)
  data_shapes = provide_data(data)
  data_names  = [x[1] for x in data_shapes]
  _setup_predictor(self, overwrite; data_shapes...)

  batch_size  = get_batch_size(data)
  data_arrays =  [self.pred_exec.arg_dict[name] for name in data_names]
  output_list = [Array{MX_float}[] for i=1:length(self.pred_exec.outputs)]
  for batch in eachbatch(data)
    load_data!(data, batch, data_arrays)
    forward(self.pred_exec, is_train=false)
    if isa(callback, Void)
      # no callback, accumulate the data and return at the end
      for (o_list, o_nd) in zip(output_list, self.pred_exec.outputs)
        push!(o_list, copy(slice(o_nd, 1:count_samples(data, batch))))
      end
    else
      outputs = self.pred_exec.outputs
      if length(outputs) == 1
        outputs = outputs[1]
      end
      callback(outputs)
    end
  end

  if !isa(callback, Void)
    # callback exists, do not accumulate data
    return nothing
  end

  if isempty(output_list)
    # maybe model does not have outputs
    return nothing
  end
  if isempty(output_list[1])
    # maybe no output because data is empty
    return length(output_list) == 1 ? output_list[1] : output_list
  end

  # concatenate along mini-batches
  output_arrays = [cat(ndims(x[1]), x...) for x in output_list]
  if length(output_arrays) == 1
    # only 1 output, return it directly, instead of a list
    output_arrays = output_arrays[1]
  end
  return output_arrays
end

function _init_model(self :: FeedForward, data :: AbstractDataProvider, initializer :: Union{AbstractInitializer, Dict}, overwrite :: Bool)
  init_model(self, initializer; overwrite=overwrite, [provide_data(data)..., provide_label(data)...]...)
end

function _create_kvstore(kv_type :: Base.Symbol, num_device :: Int, arg_params :: Dict{Base.Symbol,NDArray})
  if num_device == 1 && !ismatch(r"dist", string(kv_type))
    kv = nothing
  else
    if kv_type == :local
      max_size = maximum([prod(size(param)) for (k,param) in arg_params])
      if max_size < 1024 * 1024 * 16
        kv_type = :local_update_cpu
      else
        kv_type = :local_allreduce_cpu
      end
      info("Auto-select kvstore type = $kv_type")
    end
    kv = KVStore(kv_type)
  end

  update_on_kvstore = true
  if isa(kv, Void) || ismatch(r"local_allreduce", string(get_type(kv)))
    update_on_kvstore = false
  end

  return (kv, update_on_kvstore)
end

@defstruct TrainingOptions Any (
  initializer :: Union{AbstractInitializer, Dict} = UniformInitializer(0.01),
  n_epoch     :: Int = 10,
  eval_data   :: Union{Void, AbstractDataProvider} = nothing,
  eval_metric :: AbstractEvalMetric = Accuracy(),
  kvstore     :: Union{Base.Symbol, KVStore} = :local,
  force_init  :: Bool = false,
  callbacks   :: Vector{AbstractCallback} = AbstractCallback[],
)

function _invoke_callbacks(self::FeedForward, callbacks::Vector{AbstractCallback},
                           state::OptimizationState, type_filter::Type)
  map(callbacks) do cb
    if isa(cb, type_filter)
      if type_filter == AbstractEpochCallback
        # epoch callback have extra access to the model object
        cb(self, state)
      else
        cb(state)
      end
    end
  end
end

#=doc
.. function:: train(model :: FeedForward, ...)

   Alias to :func:`fit`.
=#
function train(self :: FeedForward, optimizer :: AbstractOptimizer, data :: AbstractDataProvider; kwargs...)
  fit(self, optimizer, data; kwargs...)
end

#=doc
.. function:: fit(model :: FeedForward, optimizer, data; kwargs...)

   Train the ``model`` on ``data`` with the ``optimizer``.

   :param FeedForward model: the model to be trained.
   :param AbstractOptimizer optimizer: the optimization algorithm to use.
   :param AbstractDataProvider data: the training data provider.
   :param Int n_epoch: default 10, the number of full data-passes to run.
   :param AbstractDataProvider eval_data: keyword argument, default ``nothing``. The data provider for
          the validation set.
   :param AbstractEvalMetric eval_metric: keyword argument, default ``Accuracy()``. The metric used
          to evaluate the training performance. If ``eval_data`` is provided, the same metric is also
          calculated on the validation set.
   :param kvstore: keyword argument, default ``:local``. The key-value store used to synchronize gradients
          and parameters when multiple devices are used for training.
   :type kvstore: :class:`KVStore` or ``Base.Symbol``
   :param AbstractInitializer initializer: keyword argument, default ``UniformInitializer(0.01)``.
   :param Bool force_init: keyword argument, default false. By default, the random initialization using the
          provided ``initializer`` will be skipped if the model weights already exists, maybe from a previous
          call to :func:`train` or an explicit call to :func:`init_model` or :func:`load_checkpoint`. When
          this option is set, it will always do random initialization at the begining of training.
   :param callbacks: keyword argument, default ``[]``. Callbacks to be invoked at each epoch or mini-batch,
          see :class:`AbstractCallback`.
   :type callbacks: ``Vector{AbstractCallback}``
=#
function fit(self :: FeedForward, optimizer :: AbstractOptimizer, data :: AbstractDataProvider; kwargs...)
  opts = TrainingOptions(; kwargs...)

  info("Start training on $(self.ctx)")

  batch_size  = get_batch_size(data)
  num_dev     = length(self.ctx)
  slices      = _split_inputs(batch_size, num_dev)

  # initialize parameters
  info("Initializing parameters...")
  arg_names, param_names, aux_names = _init_model(self, data, opts.initializer, opts.force_init)

  # setup kvstore
  kvstore = opts.kvstore
  if isa(kvstore, Base.Symbol)
    info("Creating KVStore...")
    kvstore, update_on_kvstore = _create_kvstore(kvstore, length(self.ctx), self.arg_params)
  end

  train_execs = Array(Executor, num_dev)
  for i = 1:num_dev
    data_shapes = [k => tuple(v[1:end-1]...,length(slices[i])) for (k,v) in provide_data(data)]
    label_shapes = [k => tuple(v[1:end-1]...,length(slices[i])) for (k,v) in provide_label(data)]
    train_execs[i] = simple_bind(self.arch, self.ctx[i]; grad_req=GRAD_WRITE, data_shapes..., label_shapes...)

    copy_params_from(train_execs[i], self.arg_params, self.aux_params)
  end

  # set up input data structures
  data_names   = [x[1] for x in provide_data(data)]
  label_names  = [x[1] for x in provide_label(data)]

  data_arrays  = [SlicedNDArray[(slices[i], exec.arg_dict[name]) for (i,exec) in enumerate(train_execs)]
                  for name in data_names]
  label_arrays = [SlicedNDArray[(slices[i], exec.arg_dict[name]) for (i,exec) in enumerate(train_execs)]
                  for name in label_names]

  param_idx    = filter(i -> in(arg_names[i], param_names), 1:length(arg_names))

  param_arrays = [NDArray[exec.arg_arrays[i] for exec in train_execs] for i in param_idx]
  grad_arrays  = [NDArray[exec.grad_arrays[i] for exec in train_execs] for i in param_idx]
  aux_arrays   = [NDArray[exec.aux_arrays[i] for exec in train_execs] for i = 1:length(aux_names)]

  op_state = OptimizationState(batch_size)
  optimizer.state = op_state

  if !update_on_kvstore
    updater = get_updater(optimizer)
  end

  if !isa(kvstore, Void)
    if update_on_kvstore
      set_optimizer(kvstore, optimizer)
    end

    info("Initializing KVStore...")
    # init kv with gradients
    for idx = 1:length(param_arrays)
      param_on_devs = param_arrays[idx]

      init!(kvstore, idx, self.arg_params[param_names[idx]])

      if update_on_kvstore
        # pull weights back
        pull!(kvstore, idx, param_on_devs, priority=-idx)
      end
    end
  end

  # set up output and labels in CPU for evaluation metric
  output_shapes = [tuple(size(x)[1:end-1]...,batch_size) for x in train_execs[1].outputs]
  cpu_dev = Context(CPU)
  cpu_output_arrays = [empty(shape, cpu_dev) for shape in output_shapes]
  cpu_label_arrays  = [empty(shape, cpu_dev) for (name,shape) in provide_label(data)]

  # invoke callbacks on epoch 0
  _invoke_callbacks(self, opts.callbacks, op_state, AbstractEpochCallback)

  info("Start training...")
  for i_epoch = 1:opts.n_epoch
    time_start = time()
    reset!(opts.eval_metric)

    op_state.curr_epoch = i_epoch
    op_state.curr_batch = 0

    # invoke callbacks on iteration 0
    _invoke_callbacks(self, opts.callbacks, op_state, AbstractBatchCallback)

    for batch in eachbatch(data)
      load_data!(data, batch, data_arrays)
      load_label!(data, batch, label_arrays)

      # forward and backward
      for (texec, islice) in zip(train_execs, slices)
        forward(texec, is_train=true)

        # copy outputs into cpu ndarray, for evaluation metric
        for (cpu_out, dev_out) in zip(cpu_output_arrays, texec.outputs)
          copy!(slice(cpu_out, islice), dev_out)
        end

        backward(texec)
      end

      op_state.curr_iter  += 1
      op_state.curr_batch += 1
      optimizer.state = op_state

      # update parameters
      for idx = 1:length(param_names)
        # gradient synchronization
        if !isa(kvstore, Void)
          # push gradient, priority is negative index
          push!(kvstore, idx, grad_arrays[idx], priority=-idx)
          if update_on_kvstore
            # pull back the weights
            pull!(kvstore, idx, param_arrays[idx], priority=-idx)
          else
            # pull back the sum-ed gradients, to the same locations
            pull!(kvstore, idx, grad_arrays[idx], priority=-idx)
          end
        end

        if !update_on_kvstore
          # manual updating
          for i_dev = 1:num_dev
            # create a fake index, so that the updater create states
            # for different param AND different devices, TODO(mli)
            # use a better solution later
            fake_idx = idx * num_dev + i_dev
            updater(fake_idx, grad_arrays[idx][i_dev], param_arrays[idx][i_dev])
          end
        end
      end

      # invoke callbacks after finishing each iteration
      _invoke_callbacks(self, opts.callbacks, op_state, AbstractBatchCallback)

      # update evaluation metric on training set
      load_label!(data, batch, cpu_label_arrays)
      update!(opts.eval_metric, cpu_label_arrays, cpu_output_arrays)
    end # end of one epoch

    time_stop = time()
    info(format("== Epoch {1:0>3d} ==========", i_epoch))
    info("## Training summary")
    for (name, value) in get(opts.eval_metric)
      info(format("{1:>18s} = {2:.4f}", string(name), value))
    end
    info(format("{1:>18s} = {2:.4f} seconds", "time", time_stop-time_start))

    # evaluation on validation set
    if !isa(opts.eval_data, Void)
      # because we are re-using the memory allocated for the training network,
      # the batch_size of the validation dataset must be the same as the training
      # batch_size
      @assert(get_batch_size(opts.eval_data) == batch_size)

      reset!(opts.eval_metric)
      for batch in eachbatch(opts.eval_data)
        load_data!(opts.eval_data, batch, data_arrays)

        # forward and backward
        for (texec, islice) in zip(train_execs, slices)
          forward(texec, is_train=true)

          # copy outputs into cpu ndarray, for evaluation metric
          for (cpu_out, dev_out) in zip(cpu_output_arrays, texec.outputs)
            copy!(slice(cpu_out, islice), dev_out)
          end
        end
        load_label!(opts.eval_data, batch, cpu_label_arrays)
        update!(opts.eval_metric, cpu_label_arrays, cpu_output_arrays)
      end

      info("## Validation summary")
      for (name, value) in get(opts.eval_metric)
        info(format("{1:>18s} = {2:.4f}", string(name), value))
      end
    end

    if i_epoch == opts.n_epoch || any(x->isa(x, AbstractEpochCallback), opts.callbacks)
      # copy data back to cpu
      for (name, weights) in zip(param_names, param_arrays)
        # average parameters across devices
        weight = +([copy(w, cpu()) for w in weights]...) / length(weights)
        copy!(self.arg_params[name], weight)
      end
      for (name, aux_devs) in zip(aux_names, aux_arrays)
        aux_avg = +([copy(aux, cpu()) for aux in aux_devs]...) / length(aux_devs)
        copy!(self.aux_params[name], aux_avg)
      end
    end
    _invoke_callbacks(self, opts.callbacks, op_state, AbstractEpochCallback)
  end # end of all epochs
end

function save_checkpoint(self :: FeedForward, prefix :: AbstractString, state :: OptimizationState)
  save_checkpoint(self.arch, self.arg_params, self.aux_params, prefix, state.curr_epoch)
end
function save_checkpoint(sym :: SymbolicNode, arg_params :: Dict{Base.Symbol, NDArray},
                         aux_params :: Dict{Base.Symbol, NDArray}, prefix :: AbstractString, epoch :: Int)
  save("$prefix-symbol.json", sym)
  save_dict = merge(Dict([symbol("arg:$k") => v for (k,v) in arg_params]),
                    Dict([symbol("aux:$k") => v for (k,v) in aux_params]))
  save_filename = format("{1}-{2:04d}.params", prefix, epoch)
  save(save_filename, save_dict)
  info("Saved checkpoint to '$save_filename'")
end

function load_checkpoint(prefix :: AbstractString, epoch :: Int)
  arch       = load("$prefix-symbol.json", SymbolicNode)
  saved_dict = load(format("{1}-{2:04d}.params", prefix, epoch), NDArray)
  arg_params = Dict{Base.Symbol, NDArray}()
  aux_params = Dict{Base.Symbol, NDArray}()
  for (k,v) in saved_dict
    tp, name = split(string(k), ':')
    name = symbol(name)
    if tp == "arg"
      arg_params[name] = v
    else
      aux_params[name] = v
    end
  end

  return (arch, arg_params, aux_params)
end

function load_checkpoint(prefix :: AbstractString, epoch :: Int, ::Type{FeedForward})
  arch, arg_params, aux_params = load_checkpoint(prefix, epoch)
  model = FeedForward(arch)
  model.arg_params = arg_params
  model.aux_params = aux_params
  return model
end

function load_checkpoint(self :: FeedForward, prefix :: AbstractString, epoch :: Int;
                         overwrite :: Bool = true, allow_different_arch :: Bool = false)
  if isdefined(self, :arg_params) && isdefined(self, :aux_params) && !overwrite
    info("model weights already exists, skip loading... (call with overwrite=true if needed)")
    return self
  end

  arch, arg_params, aux_params = load_checkpoint(prefix, epoch)
  if !allow_different_arch
    # TODO: is there better way to compare two symbols
    @assert(to_json(self.arch) == to_json(arch), "Cannot load from a checkpoint with different network architecture")
  end
  self.arg_params = arg_params
  self.aux_params = aux_params
  return self
end

