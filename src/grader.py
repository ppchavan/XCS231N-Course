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

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_3(self):
        """1-3-basic: over-fit training loss (from notebook)"""
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

    @graded()
    def test_4(self):
        """1-4-basic: CaptioningTransformer sampling"""
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
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_8(self):
        """1-8-basic: ViT – over-fit one batch accuracy"""
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

    @graded()
    def test_9(self):
        """1-9-basic: ViT – CIFAR-10 test accuracy"""
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
    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_6(self):
        """2-6-basic: self-supervised best accuracy"""

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
