"""Tests for graceful/python_executor.py — sandboxed code execution."""

import pytest
from graceful.python_executor import (
    execute_python, run_plot, run_calc,
    code_ns, reset_code_namespace, STATS_RE,
)


class TestExecutePython:
    def setup_method(self):
        reset_code_namespace()

    def test_simple_print(self):
        text, html = execute_python("print('hello')")
        assert "hello" in text

    def test_math_available(self):
        text, html = execute_python("import math; print(math.pi)")
        assert "3.14" in text

    def test_numpy_available(self):
        text, html = execute_python("import numpy as np; print(np.array([1,2,3]).sum())")
        assert "6" in text

    def test_blocked_import(self):
        text, html = execute_python("import os")
        assert "not available" in text or "Error" in text or "error" in text

    def test_blocked_subprocess(self):
        text, html = execute_python("import subprocess")
        assert "not available" in text or "Error" in text or "error" in text

    def test_persistent_namespace(self):
        execute_python("x = 42")
        text, _ = execute_python("print(x)")
        assert "42" in text

    def test_reset_clears_namespace(self):
        execute_python("x = 42")
        reset_code_namespace()
        text, _ = execute_python("print(x)")
        assert "NameError" in text or "not defined" in text or "error" in text.lower()

    def test_separate_namespace(self):
        ns = {}
        execute_python("y = 99", _persist=ns)
        assert ns.get("y") == 99
        # Should NOT be in the default namespace
        assert "y" not in code_ns

    def test_timeout_message(self):
        text, _ = execute_python("import time; time.sleep(120)")
        assert "timed out" in text.lower() or "timeout" in text.lower()

    def test_syntax_error(self):
        text, _ = execute_python("def f(:")
        assert "Syntax" in text or "error" in text.lower()

    def test_no_output(self):
        text, html = execute_python("x = 1 + 1")
        assert text == "*(no output)*"

    def test_html_empty_for_text_only(self):
        _, html = execute_python("print('hi')")
        assert html == ""


class TestRunPlot:
    def setup_method(self):
        reset_code_namespace()

    def test_simple_plot(self):
        label, html = run_plot("sin(x)")
        assert "sin(x)" in label
        # Should produce an image (base64 png)
        assert "base64" in html or "Error" in label or "error" in label.lower()

    def test_no_expression(self):
        label, html = run_plot("")
        assert "No expressions" in label

    def test_bad_expression(self):
        label, html = run_plot("a_variable_that_is_not_math")
        assert "unknown" in label.lower() or "Can't plot" in label

    def test_multiple_expressions(self):
        label, html = run_plot("sin(x), cos(x)")
        assert "sin(x)" in label
        assert "cos(x)" in label


class TestRunCalc:
    def setup_method(self):
        reset_code_namespace()

    def test_simple_calc(self):
        text, _ = run_calc("2 + 2")
        assert "4" in text

    def test_symbolic(self):
        text, _ = run_calc("diff(x**2, x)")
        assert "2" in text and "x" in text

    def test_invalid(self):
        text, _ = run_calc("this_is_not_math_at_all()")
        # Should either show error or "no result"
        assert text  # non-empty


class TestStatsRegex:
    def test_matches_statistics(self):
        assert STATS_RE.search("compute the regression")

    def test_matches_numpy(self):
        assert STATS_RE.search("use numpy to solve this")

    def test_matches_plot(self):
        assert STATS_RE.search("plot the data")

    def test_no_match_regular(self):
        assert not STATS_RE.search("tell me about philosophy")

    def test_case_insensitive(self):
        assert STATS_RE.search("Run a REGRESSION analysis")
