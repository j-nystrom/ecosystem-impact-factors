# Ecosystem impact factor
Using InVEST models to quantify and value the impact on ecosystem services from raw material production.

## Getting started

### 1. Clone the repository

To get started, create a project folder on your local machine and clone the repository:

```bash
mkdir <folder_name>
cd <folder_name>
git clone https://github.com/j-nystrom/nature-impact-factors.git
```

### 2. Set up a virtual environment

To manage the dependencies of the project, it's recommended that you use a virtual environment. To create a ``conda`` environment with a ``Python`` installation:

```bash
conda create --name <env_name> python=3.12
conda activate <env_name>
```

To install all dependencies, run this from the root of the project:

```bash
conda config --add channels conda-forge
conda install --file requirements.txt
```
