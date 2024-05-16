# Packaging phconvert

## PyPI Distribution

Install tools:

```
pip install wheel twine
```

Build source distribution:

```
python setup.py sdist
```

Build universal wheel:

```
python -m build
```

Upload distributions:

```
twine upload dist/*
```

## Conda distribution

Form phconvert source folder:

```
conda build conda.recipe --python=3.9
```

