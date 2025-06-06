"""Unit testing for the original SoftAdapt variant."""

from softadapt import SoftAdapt
import numpy as np
import unittest

class TestSoftAdapt(unittest.TestCase):
    """Class for testing our finite difference implementation."""

    @classmethod
    def setUpClass(class_):
        class_.decimal_place = 5

    # First starting with positive slope test cases.
    def test_beta_positive_three_components(self):
        loss_component1 = np.array([1, 2, 3, 4, 5])
        loss_component2 = np.array([150, 100, 50, 10, 0.1])
        loss_component3 = np.array([1500, 1000, 500, 100, 1])

        solutions = np.array([9.9343e-01, 6.5666e-03, 3.8908e-22])

        softadapt_object = SoftAdapt(beta=0.1)
        alpha_0, alpha_1, alpha_2 = softadapt_object.get_component_weights(
                                                                loss_component1,
                                                                loss_component2,
                                                                loss_component3,
                                                                verbose=False)
        self.assertAlmostEqual(
            alpha_0,
            solutions[0],
            self.decimal_place,
            ("Incorrect SoftAdapt calculation for simple 'dominant loss' case."
             "The first loss component failed.")
        )

        self.assertAlmostEqual(
            alpha_1,
            solutions[1],
            self.decimal_place,
            ("Incorrect SoftAdapt calculation for simple 'dominant loss' case."
             "The second loss component failed.")
        )

        self.assertAlmostEqual(
            alpha_2,
            solutions[2],
            self.decimal_place,
            ("Incorrect SoftAdapt calculation for simple 'dominant loss' case."
             "The second loss component failed.")
        )


    # TODO: Add more sophisticated unit tests


if __name__ == "__main__":
    unittest.main()
