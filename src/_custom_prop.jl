immutable CustomOpPropInfo
  list_arguments :: Ptr{Void} 
  list_outputs :: Ptr{Void}
  infer_shape :: Ptr{Void}
  declare_backward_dependency :: Ptr{Void}
  create_operator :: Ptr{Void}
  list_auxiliary_states :: Ptr{Void}
  delete :: Ptr{Void}
  p_list_arguments :: Ptr{Void} 
  p_list_outputs :: Ptr{Void}
  p_infer_shape :: Ptr{Void}
  p_declare_backward_dependency :: Ptr{Void}
  p_create_operator :: Ptr{Void}
  p_list_auxiliary_states :: Ptr{Void}
  p_delete :: Ptr{Void}
  function CustomOpPropInfo(op :: CustomOpProp)
    payload = pointer_from_objref(op)
    c_infer_shape = cfunction(_infer_shape_entry, Bool, (Cint, Ptr{Ptr{Void}}, Ptr{Cint}, Ptr{Void}))
    c_list_outputs = cfunction(_list_outputs_entry, Bool, (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Void}))
    c_list_arguments = cfunction(_list_arguments_entry, Bool, (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Void}))
    c_list_auxiliary_states = cfunction(_list_auxiliary_states_entry, Bool, (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Void}))
    c_declare_backward_dependency = cfunction(_declare_backward_dependency_entry, Bool, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Void}))
    c_delete = cfunction(_delete_entry, Void, (Ptr{Void},))

    new(c_list_arguments, c_list_outputs, c_infer_shape, c_declare_backwards_dependency, c_list_auxiliary_states, c_delete,
        payload,          payload,        payload,       payload,                        payload,                 payload)
  end
end

const __prop_pinned_memory = WeakKeyDict{CustomOpProp, Vector{Any}}()
function _pin!(op :: CustomOpProp, x :: ANY)
  xs = get(__prop_pinned_memory, op, Any[])
  push!(xs, x)
  __prop_pinned_memory[op] = xs
end

function _finalizer(op :: CustomOpProp)
  if haskey(__prop_pinned_memory)
    delete!(__prop_pinned_memory, op)
  end
end

function _delete_entry(payload :: Ptr{Void})
  # Figure out what to do here. This is to keep this part of the memory alive
end

function _infer_shape_entry(num_tensor, tensor_dims, tensor_shapes, payload)
  try
    op = unsafe_pointer_to_objref(payload) :: CustomOpProp
    n_in = length(list_arguments(op))
    n_out = length(list_outputs(op))
    n_aux = length(list_auxiliary_states())

    @assert num_tensor == n_in + n_out + n_aux

    shapes = Vector{Cuint}[]
    # copy and revert input shapes
    for i in 1:n_in
      # Get size of array and create julia arry
      ndims = unsafe_load(tensor_dims, i)
      shape = zeros(Cuint, ndims)
      tshape = unsafe_load(tensor_shapes, i)
      for j in 1:ndims
        shape[j] = unsafe_load(tshapes, ndims-j + 1)
      end
      push!(shapes, shape)
    end

    ret = infer_shape(op, shapes)
    if length(ret) == 2
      ishapes, oshapes = ret
      ashapes = Cuint[]
    elseif lenght(ret) == 3
      ishapes, oshapes, ashapes = ret
    else
       error("infer_shape must return 2 or 3 lists.")
    end

    @assert length(ishapes) == n_in
    @assert length(oshapes) == n_out
    @assert length(ashapes) == n_aux

    # We now have to reverse the arrays again
    # We can't perform a inplace operation in case the arrays share memory
    rshapes = Vector{Cuint}
    for shape in ishapes
      push!(rshapes, reverse(shape))
    end
    for shape in oshapes
      push!(rshapes, reverse(shape))
    end
    for shape in ashapes
      push!(rshapes, reverse(shape))
    end

    _pin!(op, rshapes)

    for i in 1:num_tensors
      unsafe_store!(tensor_shapes, pointer(rshapes[i]), i)
      unsafe_store!(tensor_dims, length(rshapes[i]), i)
    end
  catch err
    println(STDERR, "Error in infer_shape: ")
    showerror(STDERR, err)
    return false
  end
  return true
end

function _list_arguments_entry(data :: Ptr{Ptr{Ptr{Cchar}}}, payload :: Ptr{Void})
  try
    op = unsafe_pointer_to_objref(payload) :: CustomOpProp
    arguments = list_arguments(op)
    _pin!(op, arguments)
    ptrs = Ptr{Cchar}[Base.unsafe_convert(Ptr{Cchar}, s) for s in arguments]
    _pin!(op, ptrs)
    push!(ptrs, C_NULL)
    unsafe_store!(data, pointer(ptrs), 1)
  catch err
    println(STDERR, "Error in list_arguments: ")
    showerror(STDERR, err)
    return false
  end
  return true
end

function _list_outputs_entry(data :: Ptr{Ptr{Ptr{Cchar}}}, payload :: Ptr{Void})
  try
    op = unsafe_pointer_to_objref(payload) :: CustomOpProp
    outputs = list_outputs(op)
    _pin!(op, outputs)
    ptrs = Ptr{Cchar}[Base.unsafe_convert(Ptr{Cchar}, s) for s in outputs]
    _pin!(op, ptrs)
    push!(ptrs, C_NULL)
    unsafe_store!(data, pointer(ptrs), 1)
  catch err
    println(STDERR, "Error in list_outputs: ")
    showerror(STDERR, err)
    return false
  end
  return true
end

function _list_auxiliary_states_entry(data :: Ptr{Ptr{Ptr{Cchar}}}, payload :: Ptr{Void})
  try
    op = unsafe_pointer_to_objref(payload) :: CustomOpProp
    aux = list_auxiliary_states(op)
    _pin!(op, aux)
    ptrs = Ptr{Cchar}[Base.unsafe_convert(Ptr{Cchar}, s) for s in aux]
    _pin!(op, ptrs)
    push!(ptrs, C_NULL)
    unsafe_store!(data, pointer(ptrs), 1)
  catch err
    println(STDERR, "Error in list_auxiliary_states: ")
    showerror(STDERR, err)
    return false
  end
  return true
end

function _declare_backward_dependency(_out_grad :: Ptr{Cint},
                                      _in_data :: Ptr{Cint},
                                      _out_data :: Ptr{Cint}
                                      num_dep :: Ptr{Cint},
                                      deps :: Ptr{Ptr{Cint}},
                                      payload :: Ptr{Void})
  try
    op = unsafe_pointer_to_objref(payload) :: CustomOpProp
    out_grad = unsafe_wrap(Array, _out_grad, length(list_outputs(op)))
    in_data = unsafe_wrap(Array, _in_data, length(list_arguments(op)))
    out_data = unsafe_wrap(Array, _out_data, length(list_outputs(op)))

    rdeps = convert(Vector{Cint}, declare_backward_dependency(op, out_grad, in_data, out_data))
    _pin!(op, rdeps)

    unsafe_store!(num_dep, length(rdeps), 1)
    unsafe_store!(deps, pointer(rdeps), 1)
  catch err
    println(STDERR, "Error in declare_backward_dependency: ")
    showerror(STDERR, err)
    return false
  end
  return true
end

