> [!NOTE]  
> We are using [pytest](https://docs.pytest.org/en/stable/) for testing


# Running tests

0. Install pytest
```
pip install pytest
```

1. Run tests
```
pytest tests
```
> [!IMPORTANT]  
> pytest automatically finds files named test_*.py or *_test.py (we are taking the former as a convention)
> (note that * cannot contain underscore)


# Creating tests
1. Create new file in tests/ with name test_*.py where * is a label for the module you are testing
2. Organize your test following the pytest conventions e.g.:

```
# content of test_sample.py
def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 5
```

3. Run tests locally before pushing them not to clog up the queue
4. Create PR for your changes and make sure that newly added test was invoked on the CI

> [!NOTE]  
> CI is trigerred only for PRs tageting master