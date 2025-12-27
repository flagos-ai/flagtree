[中文](../cn/CONTRIBUTING.md)

## Running tests

After installation, you can run tests. For NVIDIA, the command is:

```shell
cd python/test
python3 -m pytest -s
```

For other backends, the command is:

```shell
cd third_party/<backend>/python/test
python3 -m pytest -s
python3 test_xxx.py
```
