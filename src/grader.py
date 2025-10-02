#!/usr/bin/env python3
import inspect
import unittest
import argparse
from graderUtil import graded, CourseTestRunner, GradedTestCase
from subprocess import Popen
from subprocess import DEVNULL, STDOUT, check_call
import torch
import os
from types import FunctionType
import re
import os
import re
import unittest
import numpy as np
from PIL import Image


from autograde_utils import if_text_in_py, text_in_cell, assert_allclose

import submission
from submission.xcs231n.gradient_check import eval_numerical_gradient

#########
# TESTS #
#########
# tests/test_transformer.py
import re

import autograde_utils                          # <-- helper supplied by infra


# ---------------------------------------------------------------------------
# Helpers collected once in setUp so we do not repeat long import lines later
# ---------------------------------------------------------------------------
class Test_1(GradedTestCase):
    """
    Transformer / ViT tests
    """

    def setUp(self):
        """
        Collect submission / solution modules that are re-used in many tests.
        """
        # --------------------------------------------------------------------
        # modules that come from the submission ------------------------------
        # --------------------------------------------------------------------
        #     submission.xcs231n.transformer_layers         -> layer building blocks
        #     submission.xcs231n.classifiers.captioning     -> captioning transformer
        #     submission.xcs231n.classifiers.transformer    -> vision transformer
        #
        #     (The `xcs231n` prefix is used in the new autograder repository.)
        # --------------------------------------------------------------------
        self.sub_layers = submission.xcs231n.transformer_layers

        self.CaptioningTransformer = (
            submission.xcs231n.classifiers.transformer.CaptioningTransformer
        )
        self.VisionTransformer = (
            submission.xcs231n.classifiers.transformer.VisionTransformer
        )

        # --------------------------------------------------------------------
        # counterpart coming from the reference solution (if available)
        # --------------------------------------------------------------------
        self.sol_layers = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.xcs231n.transformer_layers,
        )
        self.SolnCaptioningTransformer = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.xcs231n.classifiers.transformer.CaptioningTransformer,
        )
        self.SolnVisionTransformer = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.xcs231n.classifiers.transformer.VisionTransformer,
        )

        # handy constants that several tests re-use
        self.N, self.T, self.D = 4, 5, 16

    @graded()
    def test_0(self):
        """1-0-basic: multi-headed attention (PyTorch)"""
        N, T, D = self.N, self.T, self.D

        torch.manual_seed(123)
        data        = torch.randn(N, T, D)
        other_data  = torch.randn(N, 1, D)
        mask        = (torch.randn(T, T) < 0.5)

        # identical weight initialisation for both implementations
        torch.manual_seed(123)
        sol_attn = self.sol_layers.MultiHeadAttention(D, num_heads=4, dropout=0.1)
        torch.manual_seed(123)
        sub_attn = self.sub_layers.MultiHeadAttention(D, num_heads=4, dropout=0.1)

        # --------------- self-attention -------------------------------------
        with torch.no_grad():
            torch.manual_seed(123)
            sol_out = sol_attn(query=data, key=data, value=data)
            torch.manual_seed(123)
            sub_out = sub_attn(query=data, key=data, value=data)

        autograde_utils.assert_allclose(
            sol_out, sub_out, err_msg="self-attention output mismatch"
        )

        # --------------- masked self-attention ------------------------------
        with torch.no_grad():
            torch.manual_seed(123)
            sol_out = sol_attn(query=data, key=data, value=data, attn_mask=mask)
            torch.manual_seed(123)
            sub_out = sub_attn(query=data, key=data, value=data, attn_mask=mask)

        autograde_utils.assert_allclose(
            sol_out, sub_out, err_msg="masked self-attention output mismatch"
        )

        # --------------- mixed query / key-value ----------------------------
        with torch.no_grad():
            torch.manual_seed(123)
            sol_out = sol_attn(query=data, key=other_data, value=other_data)
            torch.manual_seed(123)
            sub_out = sub_attn(query=data, key=other_data, value=other_data)

        autograde_utils.assert_allclose(
            sol_out, sub_out, err_msg="cross-attention output mismatch"
        )

    @graded()
    def test_1(self):
        """1-1-basic: positional encoding"""
        N, T, D = self.N, self.T, self.D

        torch.manual_seed(1234)
        x = torch.randn(N, T, D)

        torch.manual_seed(1234)
        sol_pe = self.sol_layers.PositionalEncoding(D)
        torch.manual_seed(1234)
        sub_pe = self.sub_layers.PositionalEncoding(D)

        torch.manual_seed(1234)
        sol_out = sol_pe(x)
        torch.manual_seed(1234)
        sub_out = sub_pe(x)

        autograde_utils.assert_allclose(
            sol_out, sub_out, err_msg="positional encoding output mismatch"
        )

    @graded(is_hidden=True)
    def test_2(self):
        """1-2-hidden: CaptioningTransformer forward pass"""
        torch.manual_seed(1234)
        np.random.seed(231)

        N, D, W = 20, 30, 40
        T              = 26
        word_to_idx    = {"<NULL>": 0, "cat": 2, "dog": 3}
        V              = len(word_to_idx)

        def _run(model_cls):
            torch.manual_seed(1234)
            np.random.seed(231)

            model = model_cls(
                word_to_idx,
                input_dim=D,
                wordvec_dim=W,
                num_heads=2,
                num_layers=2,
                max_length=30,
            )
            # deterministic parameter values
            for p in model.parameters():
                p.data = torch.tensor(
                    np.linspace(-1.4, 1.3, num=p.numel()).reshape(*p.shape)
                )

            features  = torch.tensor(
                np.linspace(-1.5, 0.3, num=N * D).reshape(N, D)
            )
            captions  = torch.tensor(
                (np.arange(N * T) % V).reshape(N, T)
            )

            return model(features, captions).detach().numpy()

        expected = _run(self.SolnCaptioningTransformer)
        student  = _run(self.CaptioningTransformer)

        autograde_utils.assert_allclose(
            expected,
            student,
            err_msg="CaptioningTransformer forward mismatch",
        )

    @graded(is_hidden=True)
    def test_3(self):
        """1-3-hidden: over-fit training loss (from notebook)"""
        msg_parse = (
            "Can't parse final loss – please keep the print line that shows "
            "'Final loss:' in the tagged notebook cell."
        )

        threshold     = 0.05
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            "./submission/Transformer_Captioning.ipynb",
        )
        block_text = autograde_utils.text_in_cell(
            notebook_path, "transformer_final_training_loss"
        )
        match = re.search(r"Final loss:\s*(.*)", " ".join(block_text))
        self.assertIsNotNone(match, msg=msg_parse)

        try:
            final_loss = float(match.group(1))
        except ValueError:
            self.fail(msg=msg_parse)

        self.assertLess(
            final_loss,
            threshold,
            msg="Final training loss should be below 0.05 when over-fitting",
        )

    @graded(is_hidden=True)
    def test_4(self):
        """1-4-hidden: CaptioningTransformer sampling"""
        torch.manual_seed(1234)
        np.random.seed(231)

        word_to_idx = {
            "<NULL>": 0, "<START>": 1, "<END>": 2, "cat": 3, "dog": 4, "bird": 5
        }
        D, W = 30, 40

        model = self.CaptioningTransformer(
            word_to_idx,
            input_dim=D,
            wordvec_dim=W,
            num_heads=2,
            num_layers=2,
            max_length=30,
        )
        for p in model.parameters():
            p.data = torch.tensor(
                np.linspace(-1.4, 1.3, num=p.numel()).reshape(*p.shape)
            )

        features = torch.tensor(
            np.linspace(-1.5, 0.3, num=D).reshape(1, D)
        )

        captions = model.sample(features, max_length=10)

        # ---- basic sanity checks -------------------------------------------
        self.assertIsInstance(captions, np.ndarray, "sample() must return np.ndarray")
        self.assertEqual(captions.ndim, 2, "sample() output should be 2-D")
        self.assertEqual(captions.shape[0], 1, "batch size should stay 1")
        self.assertGreater(captions.shape[1], 0, "generated caption is empty")
        self.assertLessEqual(captions.shape[1], 10, "generated caption too long")

        self.assertTrue(
            np.all(captions >= 0) and np.all(captions < len(word_to_idx)),
            "caption indices out of vocabulary range",
        )

    @graded(is_hidden=True)
    def test_5(self):
        """1-5-hidden: PatchEmbedding"""
        torch.manual_seed(234)

        img_size     = 32
        patch_size   = 8
        in_channels  = 3
        embed_dim    = 128
        batch_size   = 4

        x = torch.randn(batch_size, in_channels, img_size, img_size)

        torch.manual_seed(234)
        sol_patch = self.sol_layers.PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        torch.manual_seed(234)
        sub_patch = self.sub_layers.PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )

        with torch.no_grad():
            sol_out = sol_patch(x)
            sub_out = sub_patch(x)

        autograde_utils.assert_allclose(
            sol_out,
            sub_out,
            atol=1e-4,
            rtol=1e-4,
            err_msg="PatchEmbedding output mismatch",
        )

    @graded(is_hidden=True)
    def test_6(self):
        """1-6-hidden: TransformerEncoderLayer"""
        torch.manual_seed(345)

        input_dim       = 128
        num_heads       = 4
        dim_ff          = 256
        dropout         = 0.1
        batch_size      = 8
        seq_len         = 16

        src  = torch.randn(batch_size, seq_len, input_dim)
        mask = (torch.randn(seq_len, seq_len) < 0.5)

        torch.manual_seed(345)
        sol_enc = self.sol_layers.TransformerEncoderLayer(
            input_dim, num_heads, dim_ff, dropout
        )
        torch.manual_seed(345)
        sub_enc = self.sub_layers.TransformerEncoderLayer(
            input_dim, num_heads, dim_ff, dropout
        )

        with torch.no_grad():
            torch.manual_seed(345)
            sol_out = sol_enc(src, src_mask=mask)
            torch.manual_seed(345)
            sub_out = sub_enc(src, src_mask=mask)

        autograde_utils.assert_allclose(
            sol_out,
            sub_out,
            atol=1e-6,
            rtol=1e-6,
            err_msg="TransformerEncoderLayer output mismatch",
        )

    @graded(is_hidden=True)
    def test_7(self):
        """1-7-hidden: VisionTransformer forward"""
        torch.manual_seed(456)

        img_size        = 32
        patch_size      = 8
        in_channels     = 3
        embed_dim       = 128
        num_layers      = 2
        num_heads       = 4
        dim_ff          = 256
        num_classes     = 10
        dropout         = 0.1
        batch_size      = 4

        x = torch.randn(batch_size, in_channels, img_size, img_size)

        torch.manual_seed(456)
        sol_vit = self.SolnVisionTransformer(
            img_size, patch_size, in_channels, embed_dim,
            num_layers, num_heads, dim_ff, num_classes, dropout
        )
        torch.manual_seed(456)
        sub_vit = self.VisionTransformer(
            img_size, patch_size, in_channels, embed_dim,
            num_layers, num_heads, dim_ff, num_classes, dropout
        )

        with torch.no_grad():
            torch.manual_seed(456)
            sol_out = sol_vit(x)
            torch.manual_seed(456)
            sub_out = sub_vit(x)

        autograde_utils.assert_allclose(
            sol_out,
            sub_out,
            atol=1e-4,
            rtol=1e-4,
            err_msg="VisionTransformer output mismatch",
        )

    @graded(is_hidden=True)
    def test_8(self):
        """1-8-hidden: ViT – over-fit one batch accuracy"""
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            "./submission/Transformer_Captioning.ipynb",
        )

        block = autograde_utils.text_in_cell(notebook_path, "vit_overfit_accuracy")
        match = re.search(
            r"Overfitting ViT on one batch\. Top-1 accuracy: ([\d\.]+)",
            " ".join(block),
        )
        self.assertIsNotNone(match, "Could not find over-fit accuracy print-out")

        acc = float(match.group(1))
        self.assertAlmostEqual(
            acc, 1.0, places=1,
            msg="ViT should reach ~1.0 acc when over-fitting a single batch",
        )

    @graded(is_hidden=True)
    def test_9(self):
        """1-9-hidden: ViT – CIFAR-10 test accuracy"""
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            "./submission/Transformer_Captioning.ipynb",
        )

        block = autograde_utils.text_in_cell(notebook_path, "vit_test_accuracy")
        match = re.search(
            r"Accuracy on test set: ([\d\.]+)", " ".join(block)
        )
        self.assertIsNotNone(match, "Could not find test accuracy print-out")

        acc = float(match.group(1))
        self.assertGreater(
            acc, 0.4,
            msg="ViT should achieve at least 0.4 accuracy on CIFAR-10 test split",
        )


# ----------------------------------------------------------------------------- #
#  Self-Supervised PyTorch tests – new environment
# ----------------------------------------------------------------------------- #

class Test_2(GradedTestCase):
    """
    Test-set 2  (self-supervised learning)
    """

    # ------------------------------------------------------------------ #
    # set-up – executed once per individual test
    # ------------------------------------------------------------------ #
    def setUp(self):
        """
        1.  Make student / reference modules available under the same names that
            the original file used (sub_cl, sol_cl, …).
        2.  Put frequently used constants on 'self' to avoid re-declaring them.
        """
        # student implementation
        self.sub_cl = submission.xcs231n.simclr.contrastive_loss
        self.sub_du = submission.xcs231n.simclr.data_utils

        # reference / solution implementation
        self.sol_cl = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.xcs231n.simclr.contrastive_loss
        )
        self.sol_du = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.xcs231n.simclr.data_utils,
        )

        # handy constants (same as in original tests)
        self.N = 6
        self.D = 16

    # ------------------------------------------------------------------ #
    # individual graded tests
    # ------------------------------------------------------------------ #

    @graded()
    def test_0(self):
        """2-0-basic: sim (similarity)"""

        torch.manual_seed(123)
        data1 = torch.randn(self.D)
        data2 = torch.randn(self.D)

        sol_sim = self.sol_cl.sim(data1, data2)
        sub_sim = self.sub_cl.sim(data1, data2)

        autograde_utils.assert_allclose(
            sol_sim, sub_sim, "sim (similarity) output does not match."
        )

    @graded()
    def test_1(self):
        """2-1-basic: simclr_loss_naive"""

        torch.manual_seed(1234)
        data_left  = torch.randn((self.N, self.D))
        data_right = torch.randn((self.N, self.D))
        tau        = torch.randn(1).item()

        sol_out = self.sol_cl.simclr_loss_naive(data_left, data_right, tau)
        sub_out = self.sub_cl.simclr_loss_naive(data_left, data_right, tau)

        autograde_utils.assert_allclose(
            sol_out, sub_out, "simclr_loss_naive output does not match."
        )

    @graded(is_hidden=True)
    def test_2(self):
        """2-2-hidden: sim_positive_pairs"""

        torch.manual_seed(1235)
        data_left  = torch.randn((self.N, self.D))
        data_right = torch.randn((self.N, self.D))

        sol_out = self.sol_cl.sim_positive_pairs(data_left, data_right)
        sub_out = self.sub_cl.sim_positive_pairs(data_left, data_right)

        autograde_utils.assert_allclose(
            sol_out, sub_out, "sim_positive_pairs output does not match."
        )

    @graded(is_hidden=True)
    def test_3(self):
        """2-3-hidden: compute_sim_matrix"""

        torch.manual_seed(1236)
        data = torch.randn((2 * self.N, self.D))

        sol_out = self.sol_cl.compute_sim_matrix(data)
        sub_out = self.sub_cl.compute_sim_matrix(data)

        autograde_utils.assert_allclose(
            sol_out, sub_out, "compute_sim_matrix output does not match."
        )

    @graded(is_hidden=True)
    def test_4(self):
        """2-4-hidden: simclr_loss_vectorized"""

        device = torch.device("cpu")
        torch.manual_seed(1234)
        data_left  = torch.randn((self.N, self.D))
        data_right = torch.randn((self.N, self.D))
        tau        = torch.randn(1).item()

        sol_out = self.sol_cl.simclr_loss_vectorized(
            data_left, data_right, tau, device=device
        )
        sub_out = self.sub_cl.simclr_loss_vectorized(
            data_left, data_right, tau, device=device
        )

        autograde_utils.assert_allclose(
            sol_out, sub_out, "simclr_loss_vectorized output does not match."
        )

    @graded(is_hidden=True)
    def test_5(self):
        """2-5-hidden: compute_train_transform"""

        torch.manual_seed(1237)
        np.random.seed(1237)

        img_array = np.random.randint(
            0, high=255, size=(32, 32, 3), dtype=np.uint8
        )
        img = Image.fromarray(img_array)

        sol_transform = self.sol_du.compute_train_transform(seed=1237)
        sub_transform = self.sub_du.compute_train_transform(seed=1237)

        torch.manual_seed(1237)
        sol_out = sol_transform(img)
        torch.manual_seed(1237)
        sub_out = sub_transform(img)

        autograde_utils.assert_allclose(
            sol_out, sub_out, "compute_train_transform output does not match."
        )

    @graded(is_hidden=True)
    def test_6(self):
        """2-6-hidden: self-supervised best accuracy"""

        msg_parse_error = (
            "Can't parse best accuracy. Please do not alter the "
            "code block which prints the best accuracy."
        )
        threshold = 0.70
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            "./submission/Self_Supervised_Learning.ipynb",
        )
        block_text = autograde_utils.text_in_cell(
            notebook_path, "ssl_best_accuracy"
        )
        match = re.search(
            r"Best top-1 accuracy with self-supervised learning:\s*(.*)",
            " ".join(block_text),
        )
        self.assertIsNotNone(match, msg=msg_parse_error)
        try:
            final_best_acc = float(match.group(1))
        except ValueError:
            self.fail(msg=msg_parse_error)

        self.assertGreaterEqual(
            final_best_acc,
            threshold,
            msg="Self-supervised best top-1 test accuracy failed.",
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
