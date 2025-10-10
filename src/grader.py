#!/usr/bin/env python3
import inspect
import unittest
import argparse
from graderUtil import graded, CourseTestRunner, GradedTestCase
from subprocess import Popen
from subprocess import DEVNULL, STDOUT, check_call
import torch
import numpy as np
import os
import time
from types import FunctionType
import re

from autograde_utils import if_text_in_py, text_in_cell, assert_allclose

import submission
from submission.xcs231n.gradient_check import eval_numerical_gradient

#########
# TESTS #
#########

# TO Fold to utility
def _assert_allclose(testcase, x, y, msg="", atol=1e-5, rtol=1e-7):
    """
    Helper to compare two torch / numpy objects inside the unittest framework.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    testcase.assertTrue(
        np.allclose(x, y, atol=atol, rtol=rtol),
        msg=msg,
    )


def _to_tensor(*args):
    return [torch.from_numpy(arg) for arg in args]

# BN
class Test_1(GradedTestCase):
    """
    Batch-/Layer-normalisation & FullyConnectedNet tests
    (adapted to the new autograder infrastructure)
    """

    def setUp(self):
        # Student implementation ------------------------------------------------
        self.sub_layers = submission.xcs231n.layers
        self.FCNsub = (
            submission.xcs231n.classifiers.fc_net.FullyConnectedNet
        )

        # Reference implementation (if available) ------------------------------
        self.sol_layers = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.xcs231n.layers
        )
        self.FCNsol = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.xcs231n.classifiers.fc_net.FullyConnectedNet,
        )

        # Helper ---------------------------------------------------------------
        self._rtol, self._atol = 1e-5, 1e-8

    # --------------------------------------------------------------------- #
    #                         Batch-Norm – forward                          #
    # --------------------------------------------------------------------- #
    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_6(self):
        """1-6-basic: batchnorm_backward_alt speed-up ≥ 1.2×"""

        def _bench():
            np.random.seed(231)
            N, D = 100, 500
            x = 5 * np.random.randn(N, D) + 12
            gamma, beta = np.random.randn(D), np.random.randn(D)
            dout = np.random.randn(N, D)
            bn_param = {"mode": "train"}
            _, cache = self.sub_layers.batchnorm_forward(x, gamma, beta, bn_param)

            t0 = time.time()
            _ = self.sub_layers.batchnorm_backward(dout, cache)
            t1 = time.time()
            _ = self.sub_layers.batchnorm_backward_alt(dout, cache)
            t2 = time.time()
            return (t1 - t0) / (t2 - t1 + 1e-12)

        speedups = [_bench() for _ in range(30)]
        self.assertGreaterEqual(
            np.mean(speedups), 1.1, msg="Mean speed-up < 1.1×"
        )
        self.assertGreaterEqual(
            np.max(speedups), 1.2, msg="Max speed-up < 1.2×"
        )

    # --------------------------------------------------------------------- #
    #               Batch-Norm + FC-Net: loss & gradients                   #
    # --------------------------------------------------------------------- #
    ### BEGIN_HIDE ###
    ### END_HIDE ###

# DROPOUT

MEAN = 2.0
SEED = 231


class Test_2(GradedTestCase):
    """
    Assignment-wide index ‘2’.  All three tests below belong to this logical
    group.
    """

    # ---------------------------------------------------------------------#
    # set-up: import student / reference implementations once and for all
    # ---------------------------------------------------------------------#
    def setUp(self):
        # Student modules
        self.sub_layers  = submission.xcs231n.layers
        self.sub_fc_net  = submission.xcs231n.classifiers.fc_net

        # Reference solution (if available – falls back to student code on
        # the autograder-side if only one code-base is present).
        self.sol_layers = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.xcs231n.layers,
        )

        # Ensure deterministic runs whenever we rely on randomness
        np.random.seed(SEED)

    # ------------------------------------------------------------------ #
    # 2-0-basic  →  Drop-out *forward* pass
    # ------------------------------------------------------------------ #
    @graded()          # visible / public
    def test_0(self):
        """2-0-basic: dropout forward"""

        x = np.random.randn(500, 500) + MEAN

        for p in [0.25, 0.5, 0.75]:
            # ---- training mode ------------------------------------------------
            out_train, _ = self.sub_layers.dropout_forward(
                x, {"p": p, "mode": "train", "seed": SEED}
            )

            # mean of the output should be close to mean of the input
            self.assertTrue(
                np.isclose(out_train.mean(), MEAN, atol=1e-2),
                msg=f"Train-time output mean not close to input mean for p={p}",
            )

            # fraction of zeros should be close to (1-p)
            frac_zeros = (out_train == 0).mean()
            self.assertTrue(
                np.isclose(frac_zeros, 1 - p, atol=1e-2),
                msg=f"Fraction of dropped units incorrect for p={p}",
            )

            # ---- test / inference mode ----------------------------------------
            out_test, _ = self.sub_layers.dropout_forward(
                x, {"p": p, "mode": "test", "seed": SEED}
            )

            # in test mode the layer should behave like the identity
            self.assertTrue(
                np.allclose(out_test, x),
                msg="dropout_forward in test mode should return the input unchanged",
            )
            self.assertAlmostEqual(out_test.mean(), MEAN, places=2)

    # ------------------------------------------------------------------ #
    # 2-1-basic  →  Drop-out *backward* pass
    # ------------------------------------------------------------------ #
    ### BEGIN_HIDE ###
    ### END_HIDE ###

    # ------------------------------------------------------------------ #
    # 2-2-basic  →  Fully-connected net + dropout, gradient check
    # ------------------------------------------------------------------ #
    @graded()
    def test_2(self):
        """2-2-basic: FC-Net with dropout – gradient check"""

        np.random.seed(231231)          # different seed for this test

        # Tiny network / dataset
        N, D, H1, H2, C = 2, 15, 20, 30, 10
        X = np.random.randn(N, D)
        y = np.random.randint(C, size=N)

        def rel_error(x, y):
            return np.max(np.abs(x - y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))

        for keep_ratio in [1, 0.75, 0.25]:
            model = self.sub_fc_net.FullyConnectedNet(
                [H1, H2],
                input_dim=D,
                num_classes=C,
                weight_scale=5e-2,
                dtype=np.float64,
                dropout_keep_ratio=keep_ratio,
                seed=123,
            )

            # Forward / backward from the student model
            loss, grads = model.loss(X, y)

            # ------------------------------------------------------------------#
            # Make sure dropout is *actually* used when keep_ratio != 1
            # ------------------------------------------------------------------#
            if keep_ratio != 1:
                model.dropout_param["seed"] = 124
                loss_second, _ = model.loss(X, y)
                model.dropout_param["seed"] = 123
                self.assertNotEqual(
                    loss,
                    loss_second,
                    msg="Drop-out appears to have no effect – implementation missing?",
                )

            # ------------------------------------------------------------------#
            # Numerical gradient check for every parameter
            # ------------------------------------------------------------------#
            for name in sorted(grads):
                f     = lambda _: model.loss(X, y)[0]
                grad_num = eval_numerical_gradient(f, model.params[name],  verbose=False, h=1e-5)

                self.assertLessEqual(
                    rel_error(grad_num, grads[name]),
                    5e-3,
                    msg=f"Gradient check failed for param '{name}' (dropout={keep_ratio})",
                )

# # CONV

# -----------------------------------------------------------------------------  
# Convolution-layer forward & backward
# -----------------------------------------------------------------------------
### BEGIN_HIDE ###
### END_HIDE ###

# -----------------------------------------------------------------------------  
#  Max-pool forward & backward
# -----------------------------------------------------------------------------

### BEGIN_HIDE ###
### END_HIDE ###

# -----------------------------------------------------------------------------  
# Three-layer ConvNet loss & gradient checks
# -----------------------------------------------------------------------------
class Test_3c(GradedTestCase):

    def setUp(self):
        # Modules
        self.cnn = submission.cnn
        self.sol_cnn = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.xcs231n.classifiers.cnn
        )

        # Build identical models (copy parameters afterwards)
        self.model     = self.cnn.ThreeLayerConvNet()
        self.model_sol = self.sol_cnn.ThreeLayerConvNet()

        # Copy parameters so both networks share weights / bias
        self.model_sol.params = {k: v.copy() for k, v in self.model.params.items()}

        np.random.seed(123123)
        N = 3
        X = np.random.randn(N, 3, 32, 32)
        y = np.random.randint(10, size=N)

        self.loss,  self.grads  = self.model.loss(X, y)
        self.loss_sol, self.grads_sol = self.model_sol.loss(X, y)

        # Repeat with regularization
        self.model.reg = self.model_sol.reg = 0.5
        self.loss_reg,  self.grads_reg  = self.model.loss(X, y)
        self.loss_reg_sol, self.grads_reg_sol = self.model_sol.loss(X, y)

    # -----------------------------------------------------------------
    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_1(self):
        """3c-1-basic: bias initialisation ≈ 0"""
        for b_key in ["b1", "b2", "b3"]:
            self.assertLessEqual(np.sum(np.abs(self.model.params[b_key])), 1e-8)

    @graded()
    def test_2(self):
        """3c-2-basic: weight initialisation non-trivial"""
        for w_key in ["W1", "W2", "W3"]:
            self.assertGreaterEqual(np.sum(np.abs(self.model.params[w_key])), 1e-3)

    ### BEGIN_HIDE ###
    ### END_HIDE ###

# -----------------------------------------------------------------------------  
# Spatial Batch Normalisation
# -----------------------------------------------------------------------------
class Test_3d(GradedTestCase):

    def setUp(self):
        self.layers     = submission.layers
        self.sol_layers = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.xcs231n.layers
        )
        np.random.seed(231231)

        # Train-mode warm-up to align running mean / var
        N, C, H, W = 10, 4, 11, 12
        bn_param        = {"mode": "train"}
        bn_param_ref    = {"mode": "train"}
        self.gamma      = np.random.randn(C)
        self.beta       = np.random.randn(C)
        gamma_ref       = self.gamma.copy()
        beta_ref        = self.beta.copy()

        for _ in range(50):
            x_tmp = 2.3 * np.random.randn(N, C, H, W) + 13
            self.layers.spatial_batchnorm_forward(x_tmp, self.gamma, self.beta, bn_param)
            self.sol_layers.spatial_batchnorm_forward(x_tmp, gamma_ref, beta_ref, bn_param_ref)

        # Store variables for tests
        self.gamma_ref, self.beta_ref = gamma_ref, beta_ref

        # Forward (test mode)
        bn_param["mode"] = bn_param_ref["mode"] = "test"
        x = 2.3 * np.random.randn(N, C, H, W) + 13
        self.out, _     = self.layers.spatial_batchnorm_forward(x, self.gamma, self.beta, bn_param)
        self.out_ref, _ = self.sol_layers.spatial_batchnorm_forward(x, gamma_ref, beta_ref, bn_param_ref)

        # Backward (train mode)
        bn_param["mode"] = bn_param_ref["mode"] = "train"
        x = 2.3 * np.random.randn(N, C, H, W) + 13
        dout = np.random.randn(N, C, H, W)

        _, cache     = self.layers.spatial_batchnorm_forward(x, self.gamma, self.beta, bn_param)
        _, cache_ref = self.sol_layers.spatial_batchnorm_forward(x, gamma_ref, beta_ref, bn_param_ref)

        self.grads     = self.layers.spatial_batchnorm_backward(dout, cache)
        self.grads_ref = self.sol_layers.spatial_batchnorm_backward(dout, cache_ref)

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_1(self):
        """3d-1-basic: running beta"""
        _assert_allclose(self, self.beta, self.beta_ref)

    ### BEGIN_HIDE ###
    ### END_HIDE ###

# -----------------------------------------------------------------------------  
# Spatial Group Normalisation
# -----------------------------------------------------------------------------

### BEGIN_HIDE ###
### END_HIDE ###

class Test_4(GradedTestCase):

    # ---------------------------------------------------------------
    # set-up
    # ---------------------------------------------------------------
    def setUp(self):
        # Path to the student’s notebook (relative to this test file)
        self.notebook_path = os.path.join(
            os.path.dirname(__file__),
            "./submission/PyTorch.ipynb",
        )

    # ---------------------------------------------------------------
    # 0 - Bare-bones network: output shape
    # ---------------------------------------------------------------
    @graded()
    def test_0(self):
        """4-0-basic: barebones output shape"""

        block_text = text_in_cell(self.notebook_path, "barebones_output_shape")[-1]
        self.assertTrue(
            "torch.Size([64, 10])" in block_text,
            msg="The bare-bones network should produce an output of shape (64, 10).",
        )

    # ---------------------------------------------------------------
    # 1 - Bare-bones network: accuracy
    # ---------------------------------------------------------------
    @graded()
    def test_1(self):
        """4-1-basic: barebones accuracy"""

        block_text = text_in_cell(self.notebook_path, "barebones_accuracy")
        # the printed string typically ends with something like  'accuracy: 43.2 %'
        accuracy = float(block_text[-1].split(" ")[-1][1:-3])
        self.assertGreaterEqual(
            accuracy,
            42.0,
            msg=f"Bare-bones accuracy {accuracy:.1f}% is below the required 42.0%.",
        )

    # ---------------------------------------------------------------
    # 2 - nn.Module network: output shape
    # ---------------------------------------------------------------
    @graded()
    def test_2(self):
        """4-2-basic: module output shape"""

        block_text = text_in_cell(self.notebook_path, "module_output_shape")[-1]
        self.assertTrue(
            "torch.Size([64, 10])" in block_text,
            msg="The nn.Module network should produce an output of shape (64, 10).",
        )

    # ---------------------------------------------------------------
    # 3 - nn.Module network: accuracy
    # ---------------------------------------------------------------
    @graded()
    def test_3(self):
        """4-3-basic: module accuracy"""

        block_text = text_in_cell(self.notebook_path, "module_accuracy")
        accuracy = float(block_text[-1].split(" ")[-1][1:-2])
        self.assertGreaterEqual(
            accuracy,
            45.0,
            msg=f"Module network accuracy {accuracy:.1f}% is below the required 45.0%.",
        )

    # ---------------------------------------------------------------
    # 4 - Sequential model: accuracy
    # ---------------------------------------------------------------
    @graded()
    def test_4(self):
        """4-4-basic: sequential accuracy"""

        block_text = text_in_cell(self.notebook_path, "sequential_accuracy")
        accuracy = float(block_text[-1].split(" ")[-1][1:-2])
        self.assertGreaterEqual(
            accuracy,
            54.0,
            msg=f"Sequential model accuracy {accuracy:.1f}% is below the required 54.0%.",
        )

    # ---------------------------------------------------------------
    # 5 - Open-ended model: accuracy
    # ---------------------------------------------------------------
    @graded()
    def test_5(self):
        """4-5-basic: open-ended accuracy"""

        block_text = text_in_cell(self.notebook_path, "open_ended_accuracy")
        accuracy = float(block_text[-1].split(" ")[-1][1:-2])
        self.assertGreaterEqual(
            accuracy,
            70.0,
            msg=f"Open-ended model accuracy {accuracy:.1f}% is below the required 70.0%.",
        )
        

class Test_5(GradedTestCase):
    """
    The number prefix ‘2’ is chosen so that this file is executed after the
    KNN tests that use prefix ‘1’.
    """

    def setUp(self):
        # Student implementation
        self.rnn_layers = submission.rnn_layers_pytorch
        self.CaptioningRNN = submission.rnn_pytorch.CaptioningRNN

        # Reference implementation (if available in the autograder container)
        self.soln_rnn_layers = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.xcs231n.rnn_layers_pytorch
        )
        self.SolnCaptioningRNN = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.xcs231n.classifiers.rnn_pytorch.CaptioningRNN
        )

        # A seed so that student / reference see exactly the same random numbers
        np.random.seed(231)

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    # --------------------------------------------------------------------- #
    # 4. Over-fit check (visible)                                           #
    # --------------------------------------------------------------------- #
    @graded()  # visible
    def test_4(self):
        """5-4-basic: over-fit tiny data (final training loss)"""

        msg_parse_error = (
            "Can't parse final loss. Please do not alter the code block that "
            "prints the final training loss in the notebook."
        )
        threshold = 0.1

        notebook_path = os.path.join(
            os.path.dirname(__file__), "./submission/RNN_Captioning_pytorch.ipynb"
        )
        # Utility to extract cell text – available in the autograder repo
        from autograde_utils import text_in_cell

        block_text = text_in_cell(notebook_path, "rnn_final_training_loss")
        match = re.search(r"Final loss:\s*(.*)", " ".join(block_text))
        self.assertIsNotNone(match, msg=msg_parse_error)

        try:
            final_training_loss = float(match.group(1))
        except ValueError:
            self.fail(msg=msg_parse_error)

        self.assertLess(
            final_training_loss,
            threshold,
            msg=(
                "Expected the model to over-fit the tiny data - "
                f"got final loss {final_training_loss:.4f} (threshold {threshold})."
            ),
        )

def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
