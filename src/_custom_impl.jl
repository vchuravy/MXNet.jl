import RawMutex

immutable CustomOpInfo
  forward :: Ptr{Void}
  backward :: Ptr{Void}
  delete :: Ptr{Void}
  p_forward :: Ptr{Void}
  p_backward :: Ptr{Void}
  p_delete :: Ptr{Void}

  function CustomOpInfo(op :: Operator)
     c_wrapper_fb = cfunction(_wrapper_fb, Bool, (Cint, Ptr{Ptr{Void}}, Ptr{Ptr{Cint}}, Ptr{Void}))
     p_f = _create_entry(op, _forward_entry)
     p_b = _create_entry(op, _backward_entry)
     new(c_wrapper_fb, c_wrapper_fb, C_NULL, p_f, p_b, C_NULL)
  end
end

const __op_pinned_memory{Operator, Vector{Any}}()
function _pin!(op :: Operator, x :: ANY)
  xs = get(__op_pinned_memory, op, Any[])
  push!(xs, x)
  __op_pinned_memory[op] = xs
end

function _finalizer(op :: Operator)
  if haskey(__op_pinned_memory)
    delete!(__op_pinned_memory, op)
  end
end

##
# Forward and backward can be called from different threads in libmxnet and
# so we need to take special care in handling these callbacks correctly in
# the julia runtime.

immutable _FB
  handle :: Ptr{Void}
  m_entry :: RawMutex.Mutex
  size :: Cint
  data :: Ptr{Ptr{Void}}
  tags :: Ptr{Cint}
end

_FB(handle :: Ptr{Void}, m_entry) = _FB(handle,m_entry, 0, 0, 0)
@assert isbits(_FB)

# This function is called async and because the Julia runtime is not thread safe, we are
# very limited in the things we can do. Using a immutable that is a bitstype we can pass,
# return values to the handling tasks.
function _wrapper_fb(size :: Cint, data :: Ptr{Ptr{Void}}, tags :: Ptr{Cint}, payload :: Ptr{Void})
  # Load the libuv async handle
  ptr = convert(Ptr{_FB}, payload)
  handle = unsafe_load(ptr, 1).handle
  m_entry = unsafe_load(ptr, 1).m_entry

  # lock the hard part
  RawMutex.lock(m_entry)

  # Create result
  val = _FB(handle, m_entry, b_exit, size, data, tags)
  unsafe_store!(ptr, val, 1)

  ccall(:uv_async_send, Void, (Ptr{Void},), handle)

  return true # Better solution?
end

function _forward_entry(op :: Operator, payload :: _FB)
  info("Forward entry function")
end

function _backward_entry(op :: Operator, payload :: _FB)
  info("Backward entry function")
end

function _create_entry(op:: Operator, _entry :: Function)
  cond = Base.AsyncCondition()
  m_entry = RawMutex.create_mutex()

  ref = Ref(_FB(Base.unsafe_convert(Ptr{Void}, cond), m_entry))
  ptr = Base.unsafe_convert(Ptr{Void}, ref)

  task = @schedule begin
    try
      while true
         wait(cond) # Do we need to replace the AsyncCondition?
         _entry(op, ref[])
         RawMutex.unlock(m_entry)
      end
    catch err
      @show err
      rethrow()
    finally
      Base.close(cond)
      RawMutex.close_mutex(m_enrty)
    end
  end
  _pin!(op, task)
  return ptr
end
