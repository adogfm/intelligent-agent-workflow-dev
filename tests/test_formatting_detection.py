"""Text formatting must not be reported as numeric division by zero."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

from agent import ReviewAgent
from code_checker import CodeChecker
from main import main
from models import ReviewReport


TEXT_FORMATS = (
    '"value=%s" % 0',
    "'value=%d' % +0",
    'r"value=%s" % -0',
    'u"数值=%s" % 0.0',
    '"value=%s" % -0.0',
    '"value=%s" % 0j',
    'b"value=%d" % 0',
    'b"value=%r" % -0.0',
    '("value=%s") % 0',
    '"value=" "%s" % 0',
    'f"value={label}: %s" % 0',
    '("value=" + "%s") % 0',
    '(b"value=" + b"%d") % 0',
    '("%s" * +1) % 0',
    '(1 * b"%d") % 0',
    '("%s" % "%s") % 0',
    '(b"%s" % b"%d") % 0',
    '("%s" if flag else "%d") % 0',
    '(b"%r" if flag else b"%d") % 0',
    '(template := "%s") % 0',
)


class FormattingDetectionTests(unittest.TestCase):
    def test_known_text_formats_do_not_report_division_by_zero(self):
        for expression in TEXT_FORMATS:
            with self.subTest(expression=expression):
                # Execute only these trusted fixtures to establish their actual semantics.
                value = eval(expression, {"label": "demo", "flag": True})
                self.assertIsInstance(value, (str, bytes))
                self.assertEqual(CodeChecker().check(f"result = {expression}\n"), [])

    def test_numeric_zero_modulo_remains_a_blocker(self):
        for expression in ("42 % 0", "42 % -0.0", "item_id % +0", "(-42) % 0", "(1 + 2) % 0", "True % 0"):
            with self.subTest(expression=expression):
                with self.assertRaises(ZeroDivisionError):
                    eval(expression, {"item_id": 7})
                findings = CodeChecker().check(f"result = {expression}\n")
                self.assertEqual([item.rule_id for item in findings], ["division-by-zero"])
                self.assertEqual(findings[0].severity, "high")
                self.assertEqual(findings[0].line, 1)

    def test_division_and_floor_division_checks_are_preserved(self):
        for expression in ("total / 0", "total // 0", "42 / -0.0", "42 // +0"):
            with self.subTest(expression=expression):
                findings = CodeChecker().check(f"result = {expression}\n")
                self.assertEqual([item.rule_id for item in findings], ["division-by-zero"])

    def test_numeric_errors_nested_inside_formatting_are_still_visited(self):
        for expression in (
            'f"value={1 / 0}: %s" % 0',
            '"%s" % (1 % 0)',
            '("%s" % "%s") % (1 // 0)',
            '("%s" + f"{1 % 0}") % 0',
        ):
            with self.subTest(expression=expression):
                findings = CodeChecker().check(f"result = {expression}\n")
                self.assertEqual([item.rule_id for item in findings], ["division-by-zero"])

    def test_other_findings_are_not_hidden_by_text_formatting(self):
        code = 'try:\n    text = "%s" % 0\n    value = 1 % 0\nexcept:\n    pass\n'
        findings = CodeChecker().check(code)
        self.assertEqual([item.rule_id for item in findings], ["division-by-zero", "bare-except"])
        self.assertEqual([item.line for item in findings], [3, 4])

    def test_classifying_f_strings_does_not_execute_calls(self):
        code = 'text = f"{undefined_callable()}: %s" % 0\n'
        self.assertEqual(CodeChecker().check(code), [])

    def test_review_and_cli_accept_formatting_but_reject_numeric_modulo(self):
        for code, expected_exit in (('text = "value=%s" % 0\n', 0), ("value = 7 % 0\n", 1)):
            with self.subTest(code=code):
                reviewer = Mock()
                reviewer.is_mock = False
                reviewer.review.return_value = ReviewReport(summary="审查完成", issues=[])
                report = ReviewAgent(reviewer).review(code)
                self.assertEqual(bool(report.issues), expected_exit == 1)
                with patch("sys.argv", ["main.py"]), \
                        patch("main.read_code_from_stdin", return_value=code), \
                        patch("main.create_reviewer", return_value=reviewer), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), expected_exit)

    def test_unknown_types_keep_conservative_modulo_detection(self):
        for expression in ("template % 0", 'str(value) % 0', '("%s" if flag else 42) % 0'):
            with self.subTest(expression=expression):
                findings = CodeChecker().check(f"result = {expression}\n")
                self.assertEqual([item.rule_id for item in findings], ["division-by-zero"])
                self.assertIn("类型", findings[0].reason)

    def test_custom_operand_may_change_an_intermediate_formatting_result(self):
        code = (
            "class NumericFormat(str):\n"
            "    def __rmod__(self, other):\n"
            "        return 7\n"
            'value = ("%s" % NumericFormat("ignored")) % 0\n'
        )
        with self.assertRaises(ZeroDivisionError):
            exec(code, {})
        findings = CodeChecker().check(code)
        self.assertEqual([item.rule_id for item in findings], ["division-by-zero"])
        self.assertEqual(findings[0].line, 4)


if __name__ == "__main__":
    unittest.main()
