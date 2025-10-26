# XCS Student Code Repository
This repository contains all code for your assignment!
The build tools in this repo can be used to run the autograder locally or
compile a LaTeX submission.

## Development environment

We provide two development environments: (i) local and (ii) Colab environment. We recommend using a local environment with Azure cloud for GPU access.

For the local environment, you need to use either `uv` or `conda` to set up your Python environment. For assignment notebooks that require **GPU**, you can use the Azure cloud VM (virtual machine) provided to you. The advantages of setting up your own environment are: (i) you can use your preferred IDE such as VSCode, and (ii) you can run basic test cases locally to help with debugging.

For the Colab environment, you don't need to set up a Python environment since it's already configured, which can be convenient. For assignment notebooks that require **GPU**, you can change the Colab runtime type to GPU, though you may need to subscribe to Colab Pro. The main advantage of using Colab is the ease of environment setup. The disadvantages are: (i) editing and debugging code can be less convenient, and (ii) you may not be able to run basic test cases.

GPU usage options are summarized as follows based on your environment setup choice:
- Azure VM is available for those who chose the local environment setup
- Colab with GPU runtime is available for those who chose the Colab environment (Note: Colab Pro subscription may be required)

The followings describes how to setup the local environments:

### Local + `uv` (preferred)
For assignment development, it is best if you work on a local environment with `uv` as your Python package manager. We also have legacy support for `conda`. If you wish to rely on `conda` [please follow setup instructions here](#option-2-using-conda-legacy).
- [How to setup `uv`?](#option-1-using-uv-recommended)
- [How to run the autograder?](#running-the-autograder-locally)

### Colab + `uv`
If your prefer Colab as your development environment, we have a tutorial video that will help you get started. For the Python package manager in the Colab environment, we offer support for both `uv` and `conda` with again preference for `uv`. 
<!-- TODO: ADD LINK TO RECORDED VIDEO SESSION SETTING UP COLAB ENV -->
- [How to setup Colab environment?]()

## Setting Up Virtual Environment

Here is how you setup your local environment. For using Google Colab, you may skip this section.
There are two ways to set up and manage the Python environment for this project:

### Option 1: Using uv (Recommended)
We have introduced [uv](https://docs.astral.sh/uv/) for a modern, faster environment management experience. For more detailed setup instructions, please refer to [the uv setup guide](docs/uv_setup.md).

This workflow uses:
- `pyproject.toml` to define base dependencies
- `requirements.txt` for CPU and MPS (Apple GPU) systems
- `requirements.cuda.txt` for CUDA (Nvidia GPUs)-enabled systems

#### Installation Steps
1. Run the installation script:
    ```bash
    source install.sh
    ```
    This will:
    - Create a virtual environment in the root directory named `.venv`
    - Configure OS compatible python version in `.python-version`
    - Sync dependencies from `pyproject.toml`
    - Automatically install either CPU or CUDA requirements depending on your hardware
2. Activate the environment
    ```bash
    source .venv/bin/activate
    ```

    > [!IMPORTANT]  
    > For every new terminal session you will need to activate your virtual environment. 

3. Deactivate
    ```bash
    deactivate
    ```

You can check if your virtual environment is ready for use by running `which python` and ensuring that the path returned is coming from within your `.venv/bin` directory. 

### Option 2: Using conda (Legacy)
If you prefer using [Conda](https://anaconda.org/anaconda/conda), please walk through the
[Anaconda Setup for XCS Courses](https://github.com/scpd-proed/General_Handouts/blob/master/Anaconda_Setup.pdf) to familiarize yourself with the coding environment. You can create the environment from the provided  `environment.yml` and/or `environment_cuda.yml` file located in the `/src` directory:

```bash
cd src
conda env create -f environment.yml
# if GPU/CUDA support available
conda env create -f environment_cuda.yml
conda activate <env_name>
```

Replace `<env_name>` with the name specified in the `environment.yml` file.

Deactivate the environment at any time with:
```bash
conda deactivate
```

## What should I submit?

Take a look at the problem set PDF for *Submission Instructions* section.

### Running the autograder locally

All assignment code is in the `src/` subirectory. Please only make changes between the lines containing
`### START CODE HERE ###` and `### END CODE HERE ###`.

The unit tests in `src/grader.py` will be used to autograde your submission.
Run the autograder locally using the following terminal command within the
`src/` subdirectory:

```bash
$ python grader.py
```

There are two types of unit tests used by our autograders:
- `basic`:  These unit tests will verify only that your code runs without
  errors on obvious test cases.

- `hidden`: These unit tests will verify that your code produces correct
  results on complex inputs and tricky corner cases.  In the student version of
  `src/grader.py`, only the setup and inputs to these unit tests are provided.
  When you run the autograder locally, these test cases will run, but the
  results will not be verified by the autograder.

For debugging purposes, a single unit test can be run locally.  For example, you
can run the test case `3a-0-basic` using the following terminal command within
the `src/` subdirectory:
```
(XCS_ENV) $ python grader.py 3a-0-basic
```
