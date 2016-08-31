immutable CustomOpInfo
  forward :: Ptr{Void}
  backward :: Ptr{Void}
  delete :: Ptr{Void}
  p_forward :: Ptr{Void}
  p_backward :: Ptr{Void}
  p_delete :: Ptr{Void}
end
 
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
end

function infer_shape_entry(num_tensor, tensor_dims, tensor_shapes, payload)
  try
    op = unsafe_pointer_to_objref(payload) :: Operator    n_in = length(list_arguments(op))
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

    # link memory lifetime of rshapes and op
    # TODO

    for i in 1:num_tensors
      unsafe_store!(tensor_shapes, pointer(rshapes[i]), i)
      unsafe_store!(tensor_dims, length(rshapes[i]), i)
    end
  catch error
    println(STDERR, "Error in infer_shape: ")
    showerror(STDERR, error)
    return false
  end
  return true
end

end


