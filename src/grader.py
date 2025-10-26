#!/usr/bin/env python3
import inspect
import unittest
import argparse
from graderUtil import graded, CourseTestRunner, GradedTestCase
from subprocess import Popen
from subprocess import DEVNULL, STDOUT, check_call
import torch
import numpy as np
import pickle
import os

from autograde_utils import if_text_in_py, text_in_cell, assert_allclose

# import student submission
import submission

#########
# TESTS #
#########


class Test_1(GradedTestCase):

    def setUp(self):

        self.test_knn = submission.classifiers.k_nearest_neighbor.KNearestNeighbor()
        self.sol_knn = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.xcs231n.classifiers.k_nearest_neighbor.KNearestNeighbor(),
        )

        np.random.seed(231)

        # train
        self.X_train = np.random.randn(50, 10)
        self.y_train = np.random.randint(5, size=(50))
        self.test_knn.train(self.X_train, self.y_train)
        self.sol_knn.train(self.X_train, self.y_train)

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_1(self):
        """1-1-basic: no scipy.linalg"""

        knn_path = os.path.join(
            os.path.dirname(__file__),
            "./submission/xcs231n/classifiers/k_nearest_neighbor.py",
        )
        if_function = if_text_in_py(knn_path, "scipy") or if_text_in_py(
            knn_path, "linalg"
        )
        self.assertTrue(
            not if_function, msg="Do not use scipy or linalg anywhere in KNN!"
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_4(self):
        """1-4-basic: no loops vs two loops speedup"""

        out_lst = text_in_cell(
            os.path.join(os.path.dirname(__file__), "./submission/knn.ipynb"),
            "no_loop",
        )
        out_twoloop = out_lst[0]
        out_noloop = out_lst[2]

        time_twoloop = float(out_twoloop.split(" ")[-2])
        time_noloop = float(out_noloop.split(" ")[-2])

        # Correct if the speedup is more than 5x
        speedup = time_twoloop / time_noloop
        self.assertTrue(speedup > 5)

    @graded()
    def test_5(self):
        """1-5-basic: cross validation"""

        cv_output = text_in_cell(
            os.path.join(os.path.dirname(__file__), "./submission/knn.ipynb"),
            "cross_validation",
        )
        acc = float(cv_output[0].split(":")[-1])
        self.assertGreater(acc, 0.28)


class Test_2(GradedTestCase):

    def setUp(self):

        self.W = np.random.randn(50, 10) * 0.0001
        self.X = np.random.randn(100, 50) * 256 - 128  # random images
        self.y = np.random.randint(10, size=(100))  # 100 images

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_2(self):
        """2-2-basic: softmax validation accuracy"""

        validation_output = text_in_cell(
            os.path.join(os.path.dirname(__file__), "./submission/softmax.ipynb"),
            "validate",
        )  # all the output text, a list
        validation_acc = float(validation_output[-1].
                               split(" ")[-1])
        return self.assertGreaterEqual(validation_acc, 0.31) and self.assertLess(
            validation_acc, 0.45, msg="Softmax validation accuracy not within the expected range"
        )

    @graded()
    def test_3(self):
        """2-3-basic: best softmax validation accuracy"""

        validation_output = text_in_cell(
            os.path.join(os.path.dirname(__file__), "./submission/softmax.ipynb"),
            "tuning",
        )  # all the output text, a list
        validation_acc = float(validation_output[-1].split(" ")[-1])
        return self.assertGreaterEqual(validation_acc, 0.35, msg="Best softmax validation accuracy not above the expected value")

    @graded()
    def test_4(self):
        """2-4-basic: best softmax test accuracy"""

        test_output = text_in_cell(
            os.path.join(os.path.dirname(__file__), "./submission/softmax.ipynb"),
            "test",
        )  # all the output text, a list
        validation_acc = float(test_output[-1].split(" ")[-1])
        return self.assertGreaterEqual(validation_acc, 0.35, msg="Best softmax test accuracy not above the expected value")


class Test_3(GradedTestCase):

    def setUp(self):

        # Two Layer Network

        np.random.seed(231)
        N, D, H, C = 3, 5, 50, 7
        X = np.random.randn(N, D)
        y = np.random.randint(C, size=N)
        self.std = 1e-3

        self.model_sub = submission.fc_net.TwoLayerNet(
            input_dim=D, hidden_dim=H, num_classes=C, weight_scale=self.std
        )

        self.model_sol = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.fc_net.TwoLayerNet(
                input_dim=D, hidden_dim=H, num_classes=C, weight_scale=self.std
            ),
        )

        model_sol_params = self.model_sol.params
        self.model_sol.params = self.model_sub.params

        self.scores_sub = self.model_sub.loss(X)
        self.loss_sub, self.grads_sub = self.model_sub.loss(X, y)

        self.scores_sol = self.model_sol.loss(X)
        self.loss_sol, self.grads_sol = self.model_sol.loss(X, y)

        self.model_sol.params = model_sol_params

        # Accuracy

        def get_accuracy(text):
            # "Validation accuracy: 0.4"
            first_line = text[0]
            # "0.4"
            last_word = first_line.split(" ")[-1]
            # 0.4
            return float(last_word)

        self.val_accuracy = get_accuracy(
            text_in_cell(
                os.path.join(
                    os.path.dirname(__file__),
                    "./submission/two_layer_net.ipynb",
                ),
                "val_accuracy",
            )
        )
        self.test_accuracy = get_accuracy(
            text_in_cell(
                os.path.join(
                    os.path.dirname(__file__),
                    "./submission/two_layer_net.ipynb",
                ),
                "test_accuracy",
            )
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_6(self):
        """3-6-basic: Validation accuracy threshold"""
        self.assertGreaterEqual(self.val_accuracy, 0.48)

    @graded()
    def test_7(self):
        """3-7-basic: Test accuracy threshold"""
        self.assertGreaterEqual(self.test_accuracy, 0.48)


class Test_4(GradedTestCase):

    def setUp(self):

        self.softmax_test_accuracy = float(
            text_in_cell(
                os.path.join(os.path.dirname(__file__), "./submission/features.ipynb"),
                "svm_test_accuracy",  # this test tag is old from SVM
            )[0]
        )
        self.nn_test_accuracy = float(
            text_in_cell(
                os.path.join(os.path.dirname(__file__), "./submission/features.ipynb"),
                "nn_test_accuracy",
            )[0]
        )

    @graded()
    def test_0(self):
        """4-0-basic: SVM accuracy check"""
        self.assertGreaterEqual(self.softmax_test_accuracy, 0.41)

    @graded()
    def test_1(self):
        """4-1-basic: Neural net accuracy check"""
        self.assertGreaterEqual(self.nn_test_accuracy, 0.56)


class Test_5(GradedTestCase):

    def setUp(self):

        np.random.seed(231)
        N, D, H1, H2, C = 2, 15, 20, 30, 10
        X = np.random.randn(N, D)
        y = np.random.randint(C, size=(N,))
        reg = 1.0

        self.model_sub = submission.fc_net.FullyConnectedNet(
            [H1, H2],
            input_dim=D,
            num_classes=C,
            reg=reg,
            weight_scale=5e-2,
            dtype=np.float64,
        )

        self.model_sol = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.fc_net.FullyConnectedNet(
                [H1, H2],
                input_dim=D,
                num_classes=C,
                reg=reg,
                weight_scale=5e-2,
                dtype=np.float64,
            ),
        )

        model_sol_params = self.model_sol.params
        self.model_sol.params = self.model_sub.params

        self.scores_sub = self.model_sub.loss(X)
        self.loss_sub, self.grads_sub = self.model_sub.loss(X, y)

        self.scores_sol = self.model_sol.loss(X)
        self.loss_sol, self.grads_sol = self.model_sol.loss(X, y)

        self.model_sol.params = model_sol_params

        # Accuracies

        acc_string = text_in_cell(
            os.path.join(
                os.path.dirname(__file__),
                "./submission/FullyConnectedNets.ipynb",
            ),
            "val_test_accuracy",
        )
        self.test_accuracy = float(acc_string[-1].split(" ")[-1])
        self.val_accuracy = float(acc_string[-2].split(" ")[-1])

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_6(self):
        """5-6-basic: validation accuracy"""
        self.assertGreaterEqual(self.val_accuracy, 0.50)

    @graded()
    def test_7(self):
        """5-7-basic: test accuracy"""
        self.assertGreaterEqual(self.test_accuracy, 0.50)


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
