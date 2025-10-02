# Defines helper functions for autograder

import json

import numpy as np
import torch


def assert_allclose(a, b, err_msg="", rtol=1e-5, atol=1e-7, squeeze=True):
    """
    Inputs:
    - a: np.ndarray. First array to compare.
    - b: np.ndarray. Second array to compare.
    - err_msg: optional, str. Error message to print on failure.
    - rtol: optional, float. Relative tolerance; see documentation for np.allclose.
        Defaults to 1e-5.
    - atol: optional, float. Absolute tolerance; see documentation for np.allclose.
        Defaults to 1e-7.
    - squeeze: optional, bool. Squeeze inputs before comparing. Defaults to True.
    """

    # By default, let's squeeze to ignore errors related to row/column vector
    # convention, etc ¯\_(ツ)_/¯
    a_orig, b_orig = a, b
    if squeeze and type(a) in (np.ndarray, torch.Tensor):
        a = a.squeeze()
    if squeeze and type(b) in (np.ndarray, torch.Tensor):
        b = b.squeeze()

    exception = None
    try:
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=err_msg)
    except Exception as e:
        exception = e

    if exception is not None:
        # If the test fails, try to give students some useful error messages
        assert (a is None) == (b is None), "Comparison failed! Unexpected 'None' value."
        assert type(a) == type(b), "Comparison type error! {} doesn't match {}.".format(
            type(a).__name__, type(b).__name__
        )
        assert (
            a.shape == b.shape
        ), "Comparison shape error! {} doesn't match {}.".format(
            a_orig.shape, b_orig.shape
        )
        assert (
            a.dtype == b.dtype
        ), "Comparison datatype error! {} doesn't match {}.".format(a.dtype, b.dtype)

        # Resort to original error message
        raise exception


def text_in_cell(ipynb_path, metadata):
    """
    Get the output texts from the CODE cell with metadata == metadata in the provided
    .ipynb file.

    Inputs:
    - ipynb_path: str. Path to the .ipynb file.
    - metadata: str. Contains cell identifier.

    Returns:
    - str. All output texts in the cell with the specified metadata.
    """

    with open(ipynb_path) as f:
        ipynb = json.load(f)

    # Pick out the target cell
    code_cells = [cell for cell in ipynb["cells"] if cell["cell_type"] == "code"]
    success = False
    for code_cell in code_cells:
        if "test" in code_cell["metadata"].keys():
            if code_cell["metadata"]["test"] == metadata:
                tg_cell = code_cell
                success = True
                break

    # Couldn't find cell
    error_file = ipynb_path.split("/")[-1]
    if not success:
        raise ValueError(
            "Corrupted notebook metadata: you may have accidentally deleted or"
            " modified a critical code cell. \n"
            f"Please try copying your {error_file} changes back into the "
            "skeleton code, rerunning, and resubmitting."
        )

    if tg_cell["outputs"]:
        # When there are multiple output blocks, combine all stdout and return
        stdout = []
        for output in tg_cell["outputs"]:
            if "name" in output and output["name"] == "stdout":
                if type(output["text"]) is list:
                    stdout += output["text"]
                elif type(output["text"]) is str:
                    # Hack to fix one student's output...
                    # For some reason, their output comes as a single string instead of
                    # a list of strings
                    stdout.extend([s for s in output["text"].split("\n")])
                else:
                    raise ValueError(
                        "Error processing code cell output in {error_file}!"
                    )

        # Strip out empty lines -- these can appear inconsistently
        stdout = [x.strip() for x in stdout if len(x.strip()) > 0]
        return stdout

    # Error message if nothing was returned
    raise ValueError(
        f"Missing code cell output in {error_file}. Make sure all code cells "
        "in your submission have been run."
    )


def if_text_in_py(py_path, string):
    """
    Check if the provided .py file contains the specified string.

    Inputs:
    - py_path: str. Path to the .py file.
    - string: str. The text to be checked.

    Returns:
    - bool. True if the .py file contains string, False otherwise.
    """

    with open(py_path) as f:
        py = f.readlines()

    for line in py:
        line = line.lstrip()  # Remove unrelevant leading characters
        if string in line and line[0] != "#":  # string is not in a comment
            return True

    return False
