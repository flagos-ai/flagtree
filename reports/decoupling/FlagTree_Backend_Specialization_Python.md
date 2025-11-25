# FlagTree Backend Specialization 统一设计（Python）

## 接口
提供两种接口，spec 接口特化函数实现，spec_func 接口特化函数定义。由于调用了当前活动驱动类中的成员，只能在活动后端发现并激活后使用，因此一般来说只能用于一个局部作用域内。如果用在 py 文件的全局作用域且该文件在启动初期被 import，则会报错。

- python/triton/runtime/driver.py
```python
# flagtree backend specialization
def spec(function_name: str, *args, **kwargs):
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func(*args, **kwargs)
    return None
```
```python
# flagtree backend func specialization
def spec_func(function_name: str):
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func
    return None
```