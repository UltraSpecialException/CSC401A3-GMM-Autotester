from importlib.util import spec_from_file_location, module_from_spec
import argparse
import numpy as np
import unittest


def load_student_module(args):
    """
    Load the student's solutions and return them as modules.

    The gmm module will be returned followed by the levenshtein module.
    """
    if not hasattr(args, "gmm"):
        raise ValueError("Arguments missing attribute \"gmm\"")

    gmm_spec = spec_from_file_location("gmm_sol", args.gmm)
    gmm_module = module_from_spec(gmm_spec)
    gmm_spec.loader.exec_module(gmm_module)

    return gmm_module


class LogBMXTests(unittest.TestCase):

    def test_bmx_single_row_input_shape(self):
        for i in range(data.shape[0]):
            for m in range(8):
                row = data[i]
                student_result = log_b_m_x(m, row, params)
                self.assertTrue(isinstance(student_result, float),
                                msg=f"Expected type float/numpy.float when giving log_b_m_x a flat vector input. "
                                    f"Got type {type(student_result)}.")

    def test_bmx_single_row_input_values(self):
        for i in range(data.shape[0]):
            for m in range(8):
                row = data[i]
                student_result = log_b_m_x(m, row, params)
                expected_result = log_bmx_expected[m][i]
                self.assertTrue(np.isclose(student_result, expected_result, atol=1e-7),
                                msg=f"\nInput:\n{row}\nLogbmx computed using component {m + 1} should be:"
                                    f"\n{expected_result}.\nGot:\n{student_result}.\nNot close enough.")

    def test_bmx_entire_dataset_input_shape(self):
        for m in range(8):
            student_result = log_b_m_x(m, data, params)
            expected_shape = (data.shape[0],)
            self.assertEqual(expected_shape, student_result.shape,
                             msg=f"Return value from log_b_m_x when given input of shape {data.shape} should "
                                 f"have shape {expected_shape}. Got shape: {student_result.shape}.")

    def test_bmx_entire_dataset_input_values(self):
        for m in range(8):
            student_result = log_b_m_x(m, data, params)
            expected_result = log_bmx_expected[m]
            is_close = np.isclose(expected_result, student_result, atol=1e-7)
            not_close_idx = list(np.where(np.equal(is_close, False))[0])
            error_msg = ", ".join([str(idx) for idx in not_close_idx])
            self.assertTrue(is_close.all(), msg=f"Logbmx of entire dataset computed using component {m + 1} should be:"
                                                f"\n{expected_result}.\nGot:\n{student_result}.\n"
                                                f"Results at indices {error_msg} not close enough.")


class LogPMXTests(unittest.TestCase):

    def test_pmx(self):
        if args.structured:
            self.run_test_pmx_structured()
        else:
            self.run_test_pmx_non_structured()

    def run_test_pmx_structured(self):
        student_result = log_p_m_x(log_bmx_expected, params)
        self.assertEqual(log_pmx_expected.shape, student_result.shape,
                         msg=f"Return value from log_pmx should have the same shape as the input. "
                             f"Expected shape: {log_pmx_expected.shape}. Got: {student_result.shape}.")

        is_close = np.isclose(log_pmx_expected, student_result, atol=1e-7)
        not_close_row, not_close_col = np.where(np.equal(is_close, False))
        not_close_idx = np.stack([not_close_row, not_close_col]).T
        error_msg = []
        for idx in not_close_idx:
            expected = log_pmx_expected[idx[0], idx[1]]
            got = student_result[idx[0], idx[1]]
            msg = f"Index {str(list(idx))} | Expected: {expected} | Got: {got}"
            error_msg.append(msg)

        error_msg = "\n".join(error_msg)
        self.assertTrue(is_close.all(), msg=f"Expected logpmx values different significantly from "
                                            f"yours at indices:\n{error_msg}")

    def run_test_pmx_non_structured(self):
        for m in range(8):
            student_result = log_p_m_x(m, data, params)
            self.assertEqual((data.shape[0],), student_result.shape,
                             msg=f"Return value from log_p_m_x when given input of shape {data.shape} should "
                                 f"have shape {(data.shape[0],)}. Got shape: {student_result.shape}.")
            is_close = np.isclose(log_pmx_expected[m], student_result, atol=1e-7)
            not_close_idx = np.where(np.equal(is_close, False))[0]
            error_msg = ", ".join([str(item) for item in not_close_idx])
            self.assertTrue(is_close.all(), msg=f"Logpmx of entire dataset using component {m + 1} "
                                                f"should be:\n{log_pmx_expected[m]}.\nGot:\n{student_result}.\n"
                                                f"Results at indices {error_msg} not close enough.")


class LogLikTest(unittest.TestCase):

    def test_log_lik(self):
        student_result = logLik(log_bmx_expected, params)
        self.assertTrue(isinstance(student_result, float),
                        msg=f"Expected type float/numpy.float when calculating log-likelihood of entire dataset "
                            f"using GMM with function logLik. Got: {type(student_result)}.")
        self.assertTrue(np.isclose(student_result, log_lik_expected, atol=1e-7),
                        msg=f"Expected log-likelihood of entire dataset using the GMM is:\n{log_lik_expected}.\n"
                            f"Got:\n{student_result}.\nNot close enough.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmm", required=True, help="Path to a3_gmm.py or a3_gmm_structured.py", type=str)
    parser.add_argument("--structured", action="store_true",
                        help="Use this argument if you used the a3_gmm_structured.py version")
    args = parser.parse_args()

    gmm_mod = load_student_module(args)

    data = np.load("test_data.npy")
    log_b_m_x = gmm_mod.log_b_m_x
    log_bmx_expected = np.load("logbmx_expected.npy")

    log_p_m_x = gmm_mod.log_p_m_x
    log_pmx_expected = np.load("logpmx_expected.npy")

    logLik = gmm_mod.logLik
    log_lik_expected = np.load("logLik_expected.npy")

    params = gmm_mod.theta("Test Theta", 8, 13)
    params.mu = np.load("param_mu.npy")
    params.Sigma = np.load("param_sigma.npy")
    params.omega = np.load("param_omega.npy")

    unittest.main(argv=["student_tests.py", "-v"])
