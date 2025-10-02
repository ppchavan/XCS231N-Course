# 🧪 Setting Up the Python Environment with `uv`

This repository can be used without the `conda` toolchain by relying on [`uv`](https://docs.astral.sh/uv/) — a fast, modern Python package manager written in Rust. `uv` is a drop-in replacement for `pip` and `virtualenv`, offering significantly faster dependency resolution and installation.

`uv` supports creating virtual environments and syncing dependencies from both `pyproject.toml` and `requirements.txt` files.  
To replicate the course's original `environment.yml`, we provide:

- `pyproject.toml` — shared dependency list
- `requirements.txt` for CPU and MPS (Apple GPU) systems
- `requirements.cuda.txt` for CUDA (Nvidia GPUs)-enabled systems

These ensure consistent environments whether you're using `conda` or `uv`.

## Installing uv

Follow the uv installation guide for you preferred way to install [`uv`](https://docs.astral.sh/uv/getting-started/installation).

# 📦 Creating the Environment

From the root of this repository, run the following to create and populate the environment:

```bash
source install.sh
```

This will:
- Create a virtual environment `(.venv)`
- Detect whether your system supports GPU
- Install either `requirements.cuda.txt` or `requirements.txt`
- Sync shared dependencies from `pyproject.toml`

To activate the environment later:

Linux/Mac

```bash
source .venv/bin/activate
```

Windows (using Git Bash)

```bash
source .venv/Scripts/activate
```

> [!IMPORTANT]  
> For every new terminal session you will need to activate your virtual environment. 

To deactivate:
```bash
deactivate
```

## 🔄 Refreshing dependencies

If you want a fresh virtual environment, you can re-install all necessary dependencies with the `-r` or `--refresh` command as so

```bash
source install.sh -r
```

## Using uv with the Autograder

The autograder workflow expects the same Python packages used in your local environment. As long as your `.venv` environment is active and up-to-date, there’s no need to use conda — uv fully supports this workflow out of the box.