"""CLI exit status must reflect the final configured acceptance checks."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from llm import LLMReviewError
from main import main
from models import (
    CodeGenerationResult, ReviewIssue, ReviewReport,
    TestExecutionResult as ExecutionResult, TestGenerationResult as GenerationResult,
)


BAD = "def add(a, b):\n    return a - b\n"
GOOD = "def add(a, b):\n    return a + b\n"
BARE_EXCEPT = "try:\n    value = int('1')\nexcept:\n    value = 0\n"
CLEAN = ReviewReport(summary="审查通过", issues=[])
ISSUE = ReviewIssue(
    problem="实现与需求不一致", severity="high", reason="结果错误", suggestion="修复后复验",
)
BLOCKED = ReviewReport(summary="仍有阻断问题", issues=[ISSUE])
PASSED = ExecutionResult(status="passed", tests_run=1, passed=1)


def reviewer_for(report=CLEAN):
    reviewer = Mock()
    reviewer.is_mock = False
    reviewer.review.return_value = report
    return reviewer


class CLIExitCodeTests(unittest.TestCase):
    def invoke(self, arguments=(), *, reviewer=None, code=GOOD, test_result=PASSED):
        reviewer = reviewer if reviewer is not None else reviewer_for()
        generator = Mock()
        generator.generate.return_value = CodeGenerationResult(code=BAD, summary="初始实现")
        tests = Mock()
        tests.generate.return_value = GenerationResult(
            test_cases=[{"name": "test_add", "code": "self.assertEqual(module.add(2, 3), 5)"}],
            summary="验证加法",
        )
        fixer = Mock()
        fixer.fix.return_value = CodeGenerationResult(code=GOOD, summary="修复加法")
        runner = Mock()
        runner.run.return_value = test_result
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["main.py", *arguments]), \
                patch("main.read_code_from_stdin", return_value=code), \
                patch("main.create_reviewer", return_value=reviewer), \
                patch("main.create_code_generator", return_value=generator), \
                patch("main.create_test_generator", return_value=tests), \
                patch("main.create_code_fixer", return_value=fixer), \
                patch("main.TestRunner", return_value=runner), \
                redirect_stdout(output), redirect_stderr(errors):
            status = main()
        return status, output.getvalue(), errors.getvalue()

    def test_stdin_review_returns_one_for_every_blocking_severity(self):
        for severity in ("critical", "high", "medium", "low"):
            with self.subTest(severity=severity):
                issue = ISSUE.model_copy(update={"severity": severity})
                reviewer = reviewer_for(ReviewReport(summary="存在问题", issues=[issue]))
                status, output, _ = self.invoke(reviewer=reviewer)
                self.assertEqual(status, 1)
                self.assertIn(f"[{severity.upper()}]", output)

    def test_clean_review_and_optional_recommendations_return_zero(self):
        optional = ISSUE.model_copy(update={"category": "improvement"})
        for report in (
            CLEAN,
            ReviewReport(summary="可选建议", issues=[], recommendations=[optional]),
            ReviewReport(summary="建议自动归类", issues=[optional]),
        ):
            with self.subTest(report=report):
                status, _, _ = self.invoke(reviewer=reviewer_for(report))
                self.assertEqual(status, 0)

    def test_file_review_returns_one_for_local_blocker(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "division.py"
            path.write_text("value = 1 / 0\n", encoding="utf-8")
            status, output, _ = self.invoke([str(path)])
        self.assertEqual(status, 1)
        self.assertIn("[HIGH]", output)

    def test_generation_blockers_return_one_even_when_tests_pass(self):
        for options in ([], ["--run-tests"], ["--run-tests", "--fix-code"]):
            with self.subTest(options=options):
                status, output, _ = self.invoke(
                    ["--requirements", "实现加法", *options], reviewer=reviewer_for(BLOCKED),
                )
                self.assertEqual(status, 1)
                self.assertIn(ISSUE.problem, output)

    def test_clean_generation_succeeds_with_enabled_checks(self):
        for options in ([], ["--run-tests"]):
            with self.subTest(options=options):
                status, _, _ = self.invoke(["--requirements", "实现加法", *options])
                self.assertEqual(status, 0)

    def test_unsuccessful_tests_block_a_clean_review(self):
        for status in ("failed", "error", "timeout"):
            with self.subTest(status=status):
                result = ExecutionResult(status=status, tests_run=1)
                exit_code, _, _ = self.invoke(
                    ["--requirements", "实现加法", "--run-tests"], test_result=result,
                )
                self.assertEqual(exit_code, 1)

    def test_single_fix_uses_final_review_instead_of_initial_review(self):
        for initial, final, expected in ((BLOCKED, CLEAN, 0), (BLOCKED, BLOCKED, 1)):
            for options in ([], ["--run-tests"]):
                with self.subTest(final=final, options=options):
                    reviewer = reviewer_for()
                    reviewer.review.side_effect = [initial, final]
                    status, _, _ = self.invoke(
                        ["--requirements", "实现加法", "--fix-code", *options], reviewer=reviewer,
                    )
                    self.assertEqual(status, expected)
                    self.assertEqual(reviewer.review.call_args.args[0], GOOD)

    def test_apply_fixes_returns_zero_after_clean_re_review(self):
        reviewer = reviewer_for()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixable.py"
            path.write_text(BARE_EXCEPT, encoding="utf-8")
            status, output, _ = self.invoke([str(path), "--apply-fixes"], reviewer=reviewer)
            self.assertIn("except Exception:", path.read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual(reviewer.review.call_count, 2)
        self.assertIn("except Exception:", reviewer.review.call_args.args[0])
        self.assertIn("修改后代码复审", output)

    def test_apply_fixes_returns_one_when_blockers_remain(self):
        for code, report in (
            (BARE_EXCEPT.replace("int('1')", "1 / 0"), CLEAN),
            (BARE_EXCEPT, BLOCKED),
        ):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "partially_fixed.py"
                path.write_text(code, encoding="utf-8")
                status, output, _ = self.invoke(
                    [str(path), "--apply-fixes"], reviewer=reviewer_for(report),
                )
                self.assertEqual(status, 1)
                self.assertIn("补丁已应用到", output)

    def test_apply_fixes_without_patch_uses_current_review(self):
        for report, expected in ((CLEAN, 0), (BLOCKED, 1)):
            with self.subTest(report=report), tempfile.TemporaryDirectory() as temporary:
                reviewer = reviewer_for(report)
                path = Path(temporary) / "unchanged.py"
                path.write_text(GOOD, encoding="utf-8")
                status, _, _ = self.invoke([str(path), "--apply-fixes"], reviewer=reviewer)
                self.assertEqual(status, expected)
                self.assertEqual(reviewer.review.call_count, 1)

    def test_apply_fixes_re_review_error_returns_one(self):
        reviewer = reviewer_for()
        reviewer.review.side_effect = [CLEAN, LLMReviewError("复审失败")]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixable.py"
            path.write_text(BARE_EXCEPT, encoding="utf-8")
            status, _, errors = self.invoke([str(path), "--apply-fixes"], reviewer=reviewer)
        self.assertEqual(status, 1)
        self.assertIn("LLM 审查失败", errors)

    def test_real_cli_process_returns_one_for_blocking_review(self):
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.update(DASHSCOPE_API_KEY="your_api_key_here", OPENAI_API_KEY="your_api_key_here")
        for entry in (["main.py"], ["-c", "from main import main; raise SystemExit(main())"]):
            with self.subTest(entry=entry):
                result = subprocess.run(
                    [sys.executable, *entry], cwd=root, env=environment,
                    input="value = 1 / 0\nEND\n", capture_output=True,
                    text=True, encoding="utf-8", timeout=30,
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("[HIGH]", result.stdout)
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
