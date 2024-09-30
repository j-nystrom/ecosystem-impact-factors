# Ecosystem impact factors
Using InVEST models to quantify and value the impact on ecosystem services from
raw material production.

## Getting started

### 1. Clone the repository

To get started, create a project folder on your local machine and clone the
repository:

```bash
mkdir <folder_name>
cd <folder_name>
git clone https://github.com/j-nystrom/ecosystem-impact-factors.git
```

### 2. Set up a virtual environment

To manage the dependencies of the project, it's recommended that you use a
virtual environment. To create a ``conda`` environment with a ``Python``
installation:

```bash
conda create --name <env_name> python=3.11
conda activate <env_name>
```

If you are using VSCode or another IDE, check that the correct Python
interpreter is used. In VSCode, Open Command Palette (Cmd+Shift+P), search for
"Python: Select Interpreter", and select the virtual environment you just
created.

To install all dependencies, run this from the root of the project:
```bash
conda config --add channels conda-forge
conda install --file requirements.txt
```

#### Enable pre-commit hooks

The pre-commit package is installed as part of ``requirements.txt``. To enable
hooks to run automatically on commit, install and verify that it works
```bash
pre-commit install
ls .git/hooks/
pre-commit run --all-files
```

### 3. Configure the PYTHONPATH

To enable imports of project modules into other parts of the project like this,

```python
from core.data.data_processing import create_site_coord_geometries
```

 we need to set the ``PYTHONPATH`` to recognize the project folder structure.
 Create a file named ``set_path.sh`` (for Linux / macOS) or ``set_path.bat``
 (for Windows) in the following directory within your Conda environment:

```bash
<conda_env_path>/etc/conda/activate.d/
```

Replace ``<conda_env_path>`` with the actual path to your Conda environment,
which you can find by running ``conda info --envs``.

In the file you created, add the content below.

Linux / macOS:
```bash
#!/bin/sh
export PYTHONPATH="<path_to_your_project>:$PYTHONPATH"
```

Windows:
```bash
@echo off
set PYTHONPATH=<path_to_your_project>;%PYTHONPATH%
```

Replace ``<path_to_your_project>`` with the actual path to the project's root
directory.

Finally, we need to unset the path when deactivating the environment. Create an
 ``unset_path.sh`` (Linux / macOS) or ``unset_path.bat`` (Windows) in the
 equivalent deactivation folder

```bash
<conda_env_path>/etc/conda/deactivate.d/
```

with the content below.

Linux / macOS:
```bash
#!/bin/sh
unset PYTHONPATH
```

Windows:
```bash
@echo off
set PYTHONPATH=
```
