def get_thread_id(dim: int, _builder=None):
    return _builder.cuda_create_get_thread_id(dim)
