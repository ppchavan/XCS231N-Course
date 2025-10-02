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
    @graded()
    def test_0(self):
        """1-0-basic: batchnorm train-time forward"""
        np.random.seed(231)
        N, D1, D2, D3 = 200, 50, 60, 3
        X = np.random.randn(N, D1)
        W1 = np.random.randn(D1, D2)
        W2 = np.random.randn(D2, D3)
        a = np.maximum(0, X.dot(W1)).dot(W2)

        gamma = np.asarray([1.0, 2.0, 3.0])
        beta = np.asarray([11.0, 12.0, 13.0])

        exp, _ = self.sol_layers.batchnorm_forward(a, gamma, beta, {"mode": "train"})
        act, _ = self.sub_layers.batchnorm_forward(a, gamma, beta, {"mode": "train"})
        self.assertTrue(
            np.allclose(exp, act, rtol=self._rtol, atol=self._atol),
            msg="Train-time batch-norm forward pass incorrect",
        )

    @graded()
    def test_1(self):
        """1-1-basic: batchnorm test-time forward (running statistics)"""

        def _run(fwd_fn: FunctionType):
            np.random.seed(231)
            N, D1, D2, D3 = 200, 50, 60, 3
            W1, W2 = np.random.randn(D1, D2), np.random.randn(D2, D3)
            bn_param = {"mode": "train"}
            gamma, beta = np.ones(D3), np.zeros(D3)

            # run a handful of train-time passes to warm running stats
            for _ in range(50):
                X = np.random.randn(N, D1)
                a = np.maximum(0, X.dot(W1)).dot(W2)
                fwd_fn(a, gamma, beta, bn_param)

            # switch to test-time
            bn_param["mode"] = "test"
            X = np.random.randn(N, D1)
            a = np.maximum(0, X.dot(W1)).dot(W2)
            out, _ = fwd_fn(a, gamma, beta, bn_param)
            return out

        exp = _run(self.sol_layers.batchnorm_forward)
        act = _run(self.sub_layers.batchnorm_forward)
        self.assertTrue(
            np.allclose(exp, act, rtol=self._rtol, atol=self._atol),
            msg="Test-time batch-norm forward pass incorrect",
        )

    # --------------------------------------------------------------------- #
    #                   Batch-Norm – backward (individual)                  #
    # --------------------------------------------------------------------- #
    @graded()
    def test_2(self):
        """1-2-basic: batchnorm dx"""
        np.random.seed(231)
        N, D = 4, 5
        x = 5 * np.random.randn(N, D) + 12
        gamma, beta = np.random.randn(D), np.random.randn(D)
        dout = np.random.randn(N, D)
        bn_param = {"mode": "train"}

        _, cache = self.sol_layers.batchnorm_forward(x, gamma, beta, bn_param)
        dx_exp, _, _ = self.sol_layers.batchnorm_backward(dout, cache)

        _, cache = self.sub_layers.batchnorm_forward(x, gamma, beta, bn_param)
        dx_act, _, _ = self.sub_layers.batchnorm_backward(dout, cache)

        self.assertTrue(
            np.allclose(dx_exp, dx_act, rtol=self._rtol, atol=self._atol),
            msg="batchnorm_backward: dx incorrect",
        )

    @graded()
    def test_3(self):
        """1-3-basic: batchnorm dgamma"""
        np.random.seed(231)
        N, D = 4, 5
        x = 5 * np.random.randn(N, D) + 12
        gamma, beta = np.random.randn(D), np.random.randn(D)
        dout = np.random.randn(N, D)
        bn_param = {"mode": "train"}

        _, cache = self.sol_layers.batchnorm_forward(x, gamma, beta, bn_param)
        _, dgamma_exp, _ = self.sol_layers.batchnorm_backward(dout, cache)

        _, cache = self.sub_layers.batchnorm_forward(x, gamma, beta, bn_param)
        _, dgamma_act, _ = self.sub_layers.batchnorm_backward(dout, cache)

        self.assertTrue(
            np.allclose(dgamma_exp, dgamma_act, rtol=self._rtol, atol=self._atol),
            msg="batchnorm_backward: dgamma incorrect",
        )

    @graded()
    def test_4(self):
        """1-4-basic: batchnorm dbeta"""
        np.random.seed(231)
        N, D = 4, 5
        x = 5 * np.random.randn(N, D) + 12
        gamma, beta = np.random.randn(D), np.random.randn(D)
        dout = np.random.randn(N, D)
        bn_param = {"mode": "train"}

        _, cache = self.sol_layers.batchnorm_forward(x, gamma, beta, bn_param)
        _, _, dbeta_exp = self.sol_layers.batchnorm_backward(dout, cache)

        _, cache = self.sub_layers.batchnorm_forward(x, gamma, beta, bn_param)
        _, _, dbeta_act = self.sub_layers.batchnorm_backward(dout, cache)

        self.assertTrue(
            np.allclose(dbeta_exp, dbeta_act, rtol=self._rtol, atol=self._atol),
            msg="batchnorm_backward: dbeta incorrect",
        )

    # --------------------------------------------------------------------- #
    #           Batch-Norm – alternative backward & speed-up                #
    # --------------------------------------------------------------------- #
    @graded()
    def test_5(self):
        """1-5-basic: batchnorm_backward_alt gradients"""
        np.random.seed(231)
        N, D = 100, 500
        x = 5 * np.random.randn(N, D) + 12
        gamma, beta = np.random.randn(D), np.random.randn(D)
        dout = np.random.randn(N, D)
        bn_param = {"mode": "train"}

        _, cache = self.sol_layers.batchnorm_forward(x, gamma, beta, bn_param)
        grads_exp = self.sol_layers.batchnorm_backward_alt(dout, cache)

        _, cache = self.sub_layers.batchnorm_forward(x, gamma, beta, bn_param)
        grads_act = self.sub_layers.batchnorm_backward_alt(dout, cache)

        for g_e, g_a in zip(grads_exp, grads_act):
            self.assertTrue(
                np.allclose(g_e, g_a, rtol=self._rtol, atol=self._atol),
                msg="Alternative batch-norm backward pass incorrect",
            )

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
    @graded()
    def test_7(self):
        """1-7-basic: FullyConnectedNet loss / grads with batch-norm"""
        params_shared = None  # reused so both impls use identical weights

        def _run(Cls):
            nonlocal params_shared
            np.random.seed(231)
            N, D, H1, H2, C = 2, 15, 20, 30, 10
            X = np.random.randn(N, D)
            y = np.random.randint(C, size=N)
            reg = 3.14
            model = Cls(
                [H1, H2],
                input_dim=D,
                num_classes=C,
                reg=reg,
                weight_scale=5e-2,
                dtype=np.float64,
                normalization="batchnorm",
            )
            if params_shared is None:
                params_shared = model.params
            else:
                model.params = params_shared
            return model.loss(X, y)

        loss_exp, grads_exp = _run(self.FCNsol)
        loss_act, grads_act = _run(self.FCNsub)

        self.assertTrue(
            np.allclose(loss_exp, loss_act, rtol=self._rtol, atol=self._atol),
            msg="FC-Net loss with batch-norm incorrect",
        )
        for k in grads_exp:
            self.assertTrue(
                np.allclose(
                    grads_exp[k], grads_act[k], rtol=self._rtol, atol=self._atol
                ),
                msg=f"FC-Net gradient for {k} incorrect (batch-norm enabled)",
            )

    # --------------------------------------------------------------------- #
    #                    Layer-Norm – forward & backward                    #
    # --------------------------------------------------------------------- #
    @graded()
    def test_8(self):
        """1-8-basic: layernorm train-time forward"""
        np.random.seed(231)
        N, D1, D2, D3 = 4, 50, 60, 3
        X = np.random.randn(N, D1)
        W1, W2 = np.random.randn(D1, D2), np.random.randn(D2, D3)
        a = np.maximum(0, X.dot(W1)).dot(W2)
        gamma = np.asarray([3.0, 3.0, 3.0])
        beta = np.asarray([5.0, 5.0, 5.0])

        exp, _ = self.sol_layers.layernorm_forward(a, gamma, beta, {"mode": "train"})
        act, _ = self.sub_layers.layernorm_forward(a, gamma, beta, {"mode": "train"})
        self.assertTrue(
            np.allclose(exp, act, rtol=self._rtol, atol=self._atol),
            msg="layernorm forward pass incorrect",
        )

    @graded()
    def test_9(self):
        """1-9-basic: layernorm backward gradients"""
        np.random.seed(231)
        N, D = 100, 500
        x = 5 * np.random.randn(N, D) + 12
        gamma, beta = np.random.randn(D), np.random.randn(D)
        dout = np.random.randn(N, D)
        ln_param = {"mode": "train"}

        _, cache = self.sol_layers.layernorm_forward(x, gamma, beta, ln_param)
        grads_exp = self.sol_layers.layernorm_backward(dout, cache)

        _, cache = self.sub_layers.layernorm_forward(x, gamma, beta, ln_param)
        grads_act = self.sub_layers.layernorm_backward(dout, cache)

        for g_e, g_a in zip(grads_exp, grads_act):
            self.assertTrue(
                np.allclose(g_e, g_a, rtol=self._rtol, atol=self._atol),
                msg="layernorm backward pass incorrect",
            )

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
    @graded()          # visible / public
    def test_1(self):
        """2-1-basic: dropout backward"""

        x    = np.random.randn(10, 10) + 10.0
        dout = np.random.randn(*x.shape)
        dp   = {"p": 0.2, "mode": "train", "seed": SEED}

        # Forward passes
        _, cache_sol = self.sol_layers.dropout_forward(x, dp)
        _, cache_sub = self.sub_layers.dropout_forward(x, dp)

        # ------------------------------------------------------------------ #
        # The student implementation might scale the mask differently from the
        # reference solution.  To compare the *backward* pass on equal footing
        # we patch the student cache so that both caches share the same mask.
        # ------------------------------------------------------------------ #
        patched_cache_sub = (
            cache_sub[0],                             # dropout_param
            cache_sol[1] / np.max(cache_sol[1]) * np.max(cache_sub[1]),  # mask
        )

        # Backward passes
        dx_sol = self.sol_layers.dropout_backward(dout, cache_sol)
        dx_sub = self.sub_layers.dropout_backward(dout, patched_cache_sub)

        self.assertTrue(
            np.allclose(dx_sol, dx_sub, atol=1e-6),
            msg="dropout_backward gradients do not match reference implementation",
        )

    # ------------------------------------------------------------------ #
    # 2-2-hidden  →  Fully-connected net + dropout, gradient check
    # ------------------------------------------------------------------ #
    @graded(is_hidden=True)
    def test_2(self):
        """2-2-hidden: FC-Net with dropout – gradient check"""

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
class Test_3a(GradedTestCase):
    """
    Convolution forward / backward tests.
    """

    def setUp(self):
        # Student implementation
        self.layers = submission.layers

        # Reference implementation (solution)
        self.sol_layers = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.xcs231n.layers
        )

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    @graded()
    def test_0(self):
        """3a-0-basic: conv_forward stride=2 pad=1"""

        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)

        conv_param = {"stride": 2, "pad": 1}
        out, _ = self.layers.conv_forward_naive(x, w, b, conv_param)
        out_sol, _ = self.sol_layers.conv_forward_naive(x, w, b, conv_param)

        _assert_allclose(self, out, out_sol)

    @graded()
    def test_1(self):
        """3a-1-basic: conv_forward stride=1 pad=2"""

        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)

        conv_param = {"stride": 1, "pad": 2}
        out, _ = self.layers.conv_forward_naive(x, w, b, conv_param)
        out_sol, _ = self.sol_layers.conv_forward_naive(x, w, b, conv_param)

        _assert_allclose(self, out, out_sol)

    # ---------------------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------------------
    @graded(is_hidden=True)
    def test_2(self):
        """3a-2-hidden: conv_backward small 3×3 kernel"""

        np.random.seed(231231)
        x = np.random.randn(4, 3, 5, 5)
        w = np.random.randn(2, 3, 3, 3)
        b = np.random.randn(2)
        dout = np.random.randn(4, 2, 5, 5)
        conv_param = {"stride": 1, "pad": 1}

        _, cache = self.layers.conv_forward_naive(x, w, b, conv_param)
        _, cache_sol = self.sol_layers.conv_forward_naive(x, w, b, conv_param)

        dx, dw, db = self.layers.conv_backward_naive(dout, cache)
        dx_sol, dw_sol, db_sol = self.sol_layers.conv_backward_naive(dout, cache_sol)

        _assert_allclose(self, dx, dx_sol)
        _assert_allclose(self, dw, dw_sol)
        _assert_allclose(self, db, db_sol)

    @graded(is_hidden=True)
    def test_3(self):
        """3a-3-hidden: conv_backward larger 5×5 kernel"""

        np.random.seed(231231)
        x = np.random.randn(4, 3, 5, 5)
        w = np.random.randn(2, 3, 5, 5)
        b = np.random.randn(2)
        dout = np.random.randn(4, 2, 5, 5)
        conv_param = {"stride": 1, "pad": 2}

        _, cache = self.layers.conv_forward_naive(x, w, b, conv_param)
        _, cache_sol = self.sol_layers.conv_forward_naive(x, w, b, conv_param)

        dx, dw, db = self.layers.conv_backward_naive(dout, cache)
        dx_sol, dw_sol, db_sol = self.sol_layers.conv_backward_naive(dout, cache_sol)

        _assert_allclose(self, dx, dx_sol)
        _assert_allclose(self, dw, dw_sol)
        _assert_allclose(self, db, db_sol)


# -----------------------------------------------------------------------------  
#  Max-pool forward & backward
# -----------------------------------------------------------------------------
class Test_3b(GradedTestCase):
    """
    Max-pool forward / backward tests.
    """

    def setUp(self):
        # Student & solution layers modules
        self.layers = submission.layers
        self.sol_layers = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.xcs231n.layers
        )

    @graded()
    def test_0(self):
        """3b-0-basic: max_pool_forward 2×2 stride=2"""

        x_shape = (2, 3, 24, 24)
        x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        out, _ = self.layers.max_pool_forward_naive(x, pool_param)
        out_sol, _ = self.sol_layers.max_pool_forward_naive(x, pool_param)

        _assert_allclose(self, out, out_sol)

    @graded(is_hidden=True)
    def test_1(self):
        """3b-1-hidden: max_pool_backward 2×2 stride=2"""

        np.random.seed(231231)
        x = np.random.randn(3, 2, 24, 24)
        dout = np.random.randn(3, 2, 12, 12)
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        _, cache = self.layers.max_pool_forward_naive(x, pool_param)
        _, cache_sol = self.sol_layers.max_pool_forward_naive(x, pool_param)

        dx = self.layers.max_pool_backward_naive(dout, cache)
        dx_sol = self.sol_layers.max_pool_backward_naive(dout, cache_sol)

        _assert_allclose(self, dx, dx_sol)


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
    @graded()
    def test_0(self):
        """3c-0-basic: parameter shapes identical"""
        self.assertEqual(set(self.model.params.keys()),
                         set(self.model_sol.params.keys()))
        for k in self.model.params:
            self.assertEqual(self.model.params[k].shape,
                             self.model_sol.params[k].shape)

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

    @graded()
    def test_3(self):
        """3c-3-basic: loss (reg=0)"""
        _assert_allclose(self, self.loss, self.loss_sol)

    @graded()
    def test_4(self):
        """3c-4-basic: loss (reg>0)"""
        _assert_allclose(self, self.loss_reg, self.loss_reg_sol)

    @graded(is_hidden=True)
    def test_5(self):
        """3c-5-hidden: gradients (reg=0)"""
        for k in self.grads:
            _assert_allclose(self, self.grads[k], self.grads_sol[k], atol=1e-7, rtol=1e-5)

    @graded(is_hidden=True)
    def test_6(self):
        """3c-6-hidden: gradients (reg>0)"""
        for k in self.grads_reg:
            _assert_allclose(self, self.grads_reg[k], self.grads_reg_sol[k], atol=1e-7, rtol=1e-5)


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

    @graded()
    def test_0(self):
        """3d-0-basic: running gamma"""
        _assert_allclose(self, self.gamma, self.gamma_ref)

    @graded()
    def test_1(self):
        """3d-1-basic: running beta"""
        _assert_allclose(self, self.beta, self.beta_ref)

    @graded()
    def test_2(self):
        """3d-2-basic: forward (test-mode)"""
        _assert_allclose(self, self.out, self.out_ref)

    @graded(is_hidden=True)
    def test_3(self):
        """3d-3-hidden: backward dx"""
        dx, dgamma, dbeta = self.grads
        dx_ref, _, _      = self.grads_ref
        _assert_allclose(self, dx, dx_ref)

    @graded(is_hidden=True)
    def test_4(self):
        """3d-4-hidden: backward dgamma"""
        _, dgamma, _    = self.grads
        _, dgamma_ref, _ = self.grads_ref
        _assert_allclose(self, dgamma, dgamma_ref)

    @graded(is_hidden=True)
    def test_5(self):
        """3d-5-hidden: backward dbeta"""
        _, _, dbeta = self.grads
        _, _, dbeta_ref = self.grads_ref
        _assert_allclose(self, dbeta, dbeta_ref)


# -----------------------------------------------------------------------------  
# Spatial Group Normalisation
# -----------------------------------------------------------------------------
class Test_3e(GradedTestCase):

    def setUp(self):
        self.layers     = submission.layers
        self.sol_layers = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.xcs231n.layers
        )
        np.random.seed(231231)

        N, C, H, W = 2, 6, 4, 5
        self.G = 2
        self.x = 4 * np.random.randn(N, C, H, W) + 10
        self.gamma = np.random.randn(1, C, 1, 1)
        self.beta  = np.random.randn(1, C, 1, 1)
        self.bn_param = {"mode": "train"}

        self.out,  self.cache  = self.layers.spatial_groupnorm_forward(
            self.x, self.gamma, self.beta, self.G, self.bn_param
        )
        self.out_ref, self.cache_ref = self.sol_layers.spatial_groupnorm_forward(
            self.x, self.gamma, self.beta, self.G, self.bn_param
        )

        # Backward
        dout = np.random.randn(*self.x.shape)
        self.grads     = self.layers.spatial_groupnorm_backward(dout, self.cache)
        self.grads_ref = self.sol_layers.spatial_groupnorm_backward(dout, self.cache_ref)

    @graded()
    def test_0(self):
        """3e-0-basic: forward"""
        # Squeeze to ignore dangling singleton dims in some student impls
        _assert_allclose(self, self.out.squeeze(), self.out_ref.squeeze())

    @graded(is_hidden=True)
    def test_1(self):
        """3e-1-hidden: backward dx"""
        dx, _, _ = self.grads
        dx_ref, _, _ = self.grads_ref
        _assert_allclose(self, dx, dx_ref)

    @graded(is_hidden=True)
    def test_2(self):
        """3e-2-hidden: backward dgamma"""
        _, dgamma, _ = self.grads
        _, dgamma_ref, _ = self.grads_ref
        _assert_allclose(self, dgamma, dgamma_ref)

    @graded(is_hidden=True)
    def test_3(self):
        """3e-3-hidden: backward dbeta"""
        _, _, dbeta = self.grads
        _, _, dbeta_ref = self.grads_ref
        _assert_allclose(self, dbeta, dbeta_ref)




# # PTH

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
        
# # RNN

# """
# Autograder tests for the RNN assignment.
# Converted to the new Gradescope-style testing environment.
# """
# import os
# import re
# import numpy as np
# import torch

# from gradescope_utils.autograder_utils.decorators import graded
# from public_tests import GradedTestCase   # The base class offered in the new infra
# import submission                         # The student’s submission root


# ----------------------------------------------------------------------------- #
# Small helpers                                                                 #
# ----------------------------------------------------------------------------- #




# ----------------------------------------------------------------------------- #
# Main test-suite                                                               #
# ----------------------------------------------------------------------------- #
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

    # --------------------------------------------------------------------- #
    # 0. rnn_step_forward                                                   #
    # --------------------------------------------------------------------- #
    @graded()
    def test_0(self):
        """5-0-basic: rnn_step_forward"""

        N, D, H = 400, 32, 16
        x = np.linspace(-0.4, 0.7, num=N * D).reshape(N, D)
        prev_h = np.linspace(-0.2, 0.5, num=N * H).reshape(N, H)
        Wx = np.linspace(-0.1, 0.9, num=D * H).reshape(D, H)
        Wh = np.linspace(-0.3, 0.7, num=H * H).reshape(H, H)
        b = np.linspace(-0.2, 0.4, num=H)

        x, prev_h, Wx, Wh, b = _to_tensor(x, prev_h, Wx, Wh, b)

        next_h = self.rnn_layers.rnn_step_forward(x, prev_h, Wx, Wh, b)
        sol_next_h = self.soln_rnn_layers.rnn_step_forward(x, prev_h, Wx, Wh, b)

        _assert_allclose(
            self,
            next_h,
            sol_next_h,
            msg="rnn_step_forward: computed next_h doesn't match reference.",
        )

    # --------------------------------------------------------------------- #
    # 1. rnn_forward                                                        #
    # --------------------------------------------------------------------- #
    @graded(is_hidden=True)
    def test_1(self):
        """5-1-hidden: rnn_forward"""

        N, T, D, H = 10, 11, 12, 13
        x = np.linspace(-0.1, 0.3, num=N * T * D).reshape(N, T, D)
        h0 = np.linspace(-0.3, 0.1, num=N * H).reshape(N, H)
        Wx = np.linspace(-0.2, 0.4, num=D * H).reshape(D, H)
        Wh = np.linspace(-0.4, 0.1, num=H * H).reshape(H, H)
        b = np.linspace(-0.7, 0.1, num=H)

        x, h0, Wx, Wh, b = _to_tensor(x, h0, Wx, Wh, b)

        h = self.rnn_layers.rnn_forward(x, h0, Wx, Wh, b)
        sol_h = self.soln_rnn_layers.rnn_forward(x, h0, Wx, Wh, b)

        _assert_allclose(
            self, h, sol_h, msg="rnn_forward: entire sequence h is incorrect."
        )

    # --------------------------------------------------------------------- #
    # 2. word_embedding_forward                                             #
    # --------------------------------------------------------------------- #
    @graded(is_hidden=True)
    def test_2(self):
        """5-2-hidden: word_embedding_forward"""

        N, T, V, D = 4, 8, 10, 6
        x = np.asarray([[0, 3, 1, 2, 9], [2, 1, 0, 3, 9]])
        W = np.linspace(0, 1, num=V * D).reshape(V, D)

        x, W = _to_tensor(x, W)

        out, _ = self.rnn_layers.word_embedding_forward(x, W)
        sol_out, _ = self.soln_rnn_layers.word_embedding_forward(x, W)

        _assert_allclose(
            self, out, sol_out, msg="word_embedding_forward output mismatch."
        )

    # --------------------------------------------------------------------- #
    # 3. End-to-end captioning loss                                         #
    # --------------------------------------------------------------------- #
    @graded(is_hidden=True)
    def test_3(self):
        """5-3-hidden: captioning_RNN_loss"""

        N, D, Wdim, H = 20, 30, 40, 50
        word_to_idx = {"<NULL>": 0, "cat": 2, "dog": 3}
        V = len(word_to_idx)
        T = 26

        def _compute_loss(CaptioningCls):
            np.random.seed(231)
            model = CaptioningCls(
                word_to_idx,
                input_dim=D,
                wordvec_dim=Wdim,
                hidden_dim=H,
                cell_type="rnn",
                dtype=torch.float32,
            )

            # Initialising every parameter to a deterministic sequence
            for k, v in model.params.items():
                model.params[k] = torch.from_numpy(
                    np.linspace(-1.4, 1.3, num=v.numel()).reshape(*v.shape)
                ).float()

            features = np.linspace(-1.5, 0.3, num=N * D).reshape(N, D)
            captions = (np.arange(N * T) % V).reshape(N, T)
            features, captions = _to_tensor(features, captions)
            
            # Ensure tensors have the same dtype as model parameters
            features = features.float()
            captions = captions.long()  # captions should be long for indexing

            return model.loss(features, captions)

        student_loss = _compute_loss(self.CaptioningRNN)
        sol_loss = _compute_loss(self.SolnCaptioningRNN)
        _assert_allclose(
            self,
            student_loss,
            sol_loss,
            msg="CaptioningRNN.loss output does not match reference.",
            atol=1e-4,
            rtol=1e-4,
        )

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
