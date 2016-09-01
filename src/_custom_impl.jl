import RawMutex

immutable CustomOpInfo
  forward :: Ptr{Void}
  backward :: Ptr{Void}
  delete :: Ptr{Void}
  p_forward :: Ptr{Void}
  p_backward :: Ptr{Void}
  p_delete :: Ptr{Void}
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
