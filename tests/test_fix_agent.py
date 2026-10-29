"""Offline feedback-driven repair tests, including real regression tests."""

from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent import CodeReviewAgent
from coding_agent import CodingAgent, create_coding_workflow
from fix_agent import FixAgent, create_fix_workflow
from llm import (
    LLMFixError,
    MockCodeFixer,
    MockCodeGenerator,
    MockReviewer,
    MockTestGenerator,
    OpenAICodeFixer,
    create_code_fixer,
)
from main import main
from models import (
    CodeGenerationResult,
    ReviewIssue,
    ReviewReport,
    TestCase as GeneratedCase,
    TestExecutionResult as ExecutionResult,
    TestFailure as FailureDetail,
    TestGenerationResult as GenerationResult,
)
from test_agent import TestAgent as WorkflowTestAgent
from workflow import Agent, WorkflowExecutionError, WorkflowState, WorkflowStatus


BAD_CODE = "def add(a, b):\n    return a - b\n"
GOOD_CODE = "def add(a, b):\n    return a + b\n"
ISSUE = ReviewIssue(problem="add 使用了减法", severity="high", reason="不符合需求", suggestion="改为相加")
REPORT = ReviewReport(summary="存在加法错误", issues=[ISSUE])
FAILED_TEST = ExecutionResult(
    status="failed", tests_run=1, failed=1,
    failures=[FailureDetail(test="test_add", traceback="AssertionError: -1 != 5")],
)
CASES = [GeneratedCase(name="test_add", code="self.assertEqual(module.add(2, 3), 5)")]


class FixAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixer = Mock()
        self.fixer.fix.return_value = CodeGenerationResult(code=GOOD_CODE, summary="改为正确的加法")
        self.agent = FixAgent(self.fixer)

    def test_feedback_is_forwarded_and_code_is_committed_atomically(self) -> None:
        state = WorkflowState({
            "code": BAD_CODE, "report": REPORT, "test_result": FAILED_TEST,
            "test_cases": CASES, "requirements": "实现相加", "context": "保留 add 接口",
        })
        run = create_fix_workflow(self.agent).run(state)

        self.assertIsInstance(self.agent, Agent)
        self.fixer.fix.assert_called_once_with(
            BAD_CODE, report=REPORT, test_result=FAILED_TEST, test_cases=CASES,
            requirements="实现相加", context="保留 add 接口",
        )
        self.assertEqual(state.get("code"), GOOD_CODE)
        self.assertEqual(state.get("fixed_code"), GOOD_CODE)
        self.assertEqual(state.get("code_before_fix"), BAD_CODE)
        self.assertEqual(state.get("fix_result"), run.output)
        self.assertTrue(run.output.changed)
        self.assertIn("-    return a - b", run.output.diff)
        self.assertIn("+    return a + b", run.output.diff)
        self.assertEqual(state.get("report_before_fix"), REPORT)
        self.assertEqual(state.get("test_result_before_fix"), FAILED_TEST)
        self.assertIsNone(state.get("report"))
        self.assertIsNone(state.get("test_result"))
        self.assertEqual(state.get("test_cases"), CASES)
        self.assertTrue(state.get("reuse_test_cases"))
        self.assertEqual(state.version, 1)
        self.assertEqual(state.history[0].source, "fix")

    def test_review_only_and_test_only_feedback_are_supported(self) -> None:
        for feedback in ({"report": REPORT}, {"test_result": FAILED_TEST}):
            with self.subTest(feedback=feedback):
                run = create_fix_workflow(self.agent).run({"code": BAD_CODE, **feedback})
                self.assertEqual(run.state.get("code"), GOOD_CODE)

    def test_timeout_is_actionable_test_feedback(self) -> None:
        run = create_fix_workflow(self.agent).run({
            "code": BAD_CODE, "test_result": ExecutionResult(status="timeout", message="超时"),
        })
        self.assertTrue(run.output.changed)
        self.assertEqual(self.fixer.fix.call_args.kwargs["test_result"].status, "timeout")

    def test_no_issues_or_failures_skips_provider(self) -> None:
        report = ReviewReport(summary="没有问题", issues=[])
        state = WorkflowState({"code": GOOD_CODE, "report": report,
                               "test_result": ExecutionResult(status="passed", tests_run=1, passed=1)})
        run = create_fix_workflow(self.agent).run(state)
        self.assertTrue(run.output.skipped)
        self.assertFalse(run.output.changed)
        self.assertEqual(state.get("code"), GOOD_CODE)
        self.assertIs(state.get("report"), report)
        self.assertEqual(state.get("test_result").status, "passed")
        self.fixer.fix.assert_not_called()

    def test_unchanged_output_preserves_current_diagnostics(self) -> None:
        self.fixer.fix.return_value = CodeGenerationResult(code=BAD_CODE, summary="未能修复")
        run = create_fix_workflow(self.agent).run({"code": BAD_CODE, "report": REPORT,
                                                  "test_result": FAILED_TEST})
        self.assertFalse(run.output.changed)
        self.assertFalse(run.output.skipped)
        self.assertEqual(run.state.get("test_result"), FAILED_TEST)
        self.assertEqual(run.state.get("report"), REPORT)

    def test_provider_failure_or_invalid_code_preserves_state(self) -> None:
        for output in (None, "not a result", CodeGenerationResult.model_construct(code="def broken(:", summary="bad")):
            with self.subTest(output=output):
                self.fixer.fix.side_effect = LLMFixError("服务不可用") if output is None else None
                self.fixer.fix.return_value = output
                state = WorkflowState({"code": BAD_CODE, "report": REPORT, "test_result": FAILED_TEST})
                with self.assertRaises(WorkflowExecutionError) as caught:
                    create_fix_workflow(self.agent).run(state)
                self.assertIsInstance(caught.exception.__cause__, LLMFixError)
                self.assertEqual(state.status, WorkflowStatus.FAILED)
                self.assertEqual(state.get("code"), BAD_CODE)
                self.assertIs(state.get("report"), REPORT)
                self.assertEqual(state.version, 0)
                self.assertIsNone(state.get("fixed_code"))

    def test_invalid_or_stale_feedback_is_rejected_before_provider(self) -> None:
        inputs = [
            {}, {"code": " "}, {"code": 123}, {"code": BAD_CODE},
            {"code": BAD_CODE, "report": "invalid"},
            {"code": BAD_CODE, "test_result": "invalid"},
            {"code": BAD_CODE, "report": REPORT, "requirements": None},
            {"code": BAD_CODE, "report": REPORT, "test_cases": "invalid"},
            {"code": BAD_CODE, "report": None, "test_result": None, "report_before_fix": REPORT},
        ]
        for values in inputs:
            with self.subTest(values=values), self.assertRaises(WorkflowExecutionError):
                create_fix_workflow(self.agent).run(values)
        self.fixer.fix.assert_not_called()


class FixWorkflowTests(unittest.TestCase):
    def make_agents(self, fixed_code: str = GOOD_CODE):
        coding = Mock()
        coding.generate.return_value = CodeGenerationResult(code=BAD_CODE, summary="初始代码")
        reviewer = Mock()
        reviewer.review.side_effect = lambda code, findings, **kwargs: (
            REPORT if code == BAD_CODE else ReviewReport(summary="没有审查问题", issues=[])
        )
        generator = Mock()
        generator.generate.return_value = GenerationResult(test_cases=CASES, summary="验证加法")
        fixer = Mock()
        fixer.fix.return_value = CodeGenerationResult(code=fixed_code, summary="修复加法")
        return coding, reviewer, generator, fixer

    def test_full_workflow_fixes_and_retests_the_original_failed_cases(self) -> None:
        coding, reviewer, generator, fixer = self.make_agents()
        workflow = create_coding_workflow(
            CodingAgent(coding), CodeReviewAgent(reviewer),
            test_agent=WorkflowTestAgent(generator), fix_agent=FixAgent(fixer),
        )
        run = workflow.run({"requirements": "实现相加"})
        state = run.state
        self.assertEqual([item.step for item in run.executions],
                         ["coding", "review", "test", "fix", "verify_review", "verify_test"])
        self.assertEqual(state.get("test_result_before_fix").status, "failed")
        self.assertEqual(state.get("test_result").status, "passed")
        self.assertEqual(state.get("test_cases"), CASES)
        generator.generate.assert_called_once()
        self.assertEqual(reviewer.review.call_count, 2)
        self.assertEqual(fixer.fix.call_args.kwargs["test_result"].status, "failed")
        self.assertTrue(fixer.fix.call_args.kwargs["test_result"].failures)
        self.assertEqual(state.get("code"), GOOD_CODE)
        self.assertEqual(state.get("report").issues, [])
        self.assertFalse(state.get("reuse_test_cases"))
        self.assertEqual(state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(state.version, 6)

    def test_test_only_workflow_and_unresolved_failures_do_not_loop(self) -> None:
        coding, _, generator, fixer = self.make_agents("def add(a, b):\n    return a + b + 1\n")
        run = create_coding_workflow(
            CodingAgent(coding), test_agent=WorkflowTestAgent(generator), fix_agent=FixAgent(fixer)
        ).run({"requirements": "实现相加"})
        self.assertEqual([item.step for item in run.executions], ["coding", "test", "fix", "verify_test"])
        self.assertEqual(run.state.get("test_result").status, "failed")
        fixer.fix.assert_called_once()
        generator.generate.assert_called_once()

    def test_review_only_workflow_uses_offline_fixer_then_refreshes_review(self) -> None:
        code = "try:\n    work()\nexcept:\n    pass\n"
        coding = Mock()
        coding.generate.return_value = CodeGenerationResult(code=code, summary="初始代码")
        reviewer = Mock()
        reviewer.review.return_value = ReviewReport(summary="Reviewer 完成", issues=[])
        run = create_coding_workflow(
            CodingAgent(coding), CodeReviewAgent(reviewer), fix_agent=FixAgent(MockCodeFixer())
        ).run({"requirements": "异常处理"})
        self.assertIn("except Exception:", run.state.get("code"))
        self.assertEqual(run.state.get("report").issues, [])
        self.assertEqual([item.step for item in run.executions], ["coding", "review", "fix", "verify_review"])

    def test_unchanged_repairs_skip_verification(self) -> None:
        coding, reviewer, _, fixer = self.make_agents(BAD_CODE)
        run = create_coding_workflow(
            CodingAgent(coding), CodeReviewAgent(reviewer), fix_agent=FixAgent(fixer)
        ).run({"requirements": "实现相加"})
        self.assertEqual([item.step for item in run.executions], ["coding", "review", "fix"])
        reviewer.review.assert_called_once()
        self.assertFalse(run.output.changed)
        self.assertTrue(run.state.get("report").issues)

    def test_factory_requires_a_feedback_node(self) -> None:
        with self.assertRaises(ValueError):
            create_coding_workflow(CodingAgent(MockCodeGenerator()), fix_agent=FixAgent(MockCodeFixer()))


class CodeFixerTests(unittest.TestCase):
    def test_factory_selects_mock_or_existing_api_configuration(self) -> None:
        for key in ("", "your_api_key_here"):
            with self.subTest(key=key), patch("llm.load_dotenv"), patch.dict(
                os.environ, {"DASHSCOPE_API_KEY": key}, clear=True
            ):
                self.assertIsInstance(create_code_fixer(), MockCodeFixer)
        for name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
            with self.subTest(name=name), patch("llm.load_dotenv"), patch.dict(
                os.environ, {name: "test-key"}, clear=True
            ), patch("llm.OpenAICodeFixer") as real:
                self.assertIs(create_code_fixer(), real.return_value)
                real.assert_called_once_with(api_key="test-key")

    def test_real_provider_receives_both_diagnostics_and_regression_cases(self) -> None:
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(
                content=json.dumps({"code": GOOD_CODE, "summary": "修复加法"}), refusal=None,
            )
        )])
        with patch("llm.load_dotenv"):
            fixer = OpenAICodeFixer(api_key="test-key", model="qwen-test", client=client)
        result = fixer.fix(BAD_CODE, report=REPORT, test_result=FAILED_TEST,
                           test_cases=CASES, requirements="实现相加", context="保留接口")
        self.assertEqual(result.code, GOOD_CODE)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "qwen-test")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(payload["code"], BAD_CODE)
        self.assertEqual(payload["report"]["issues"][0]["problem"], ISSUE.problem)
        self.assertEqual(payload["test_result"]["failures"][0]["traceback"], "AssertionError: -1 != 5")
        self.assertEqual(payload["test_cases"], [case.model_dump() for case in CASES])
        self.assertEqual(payload["requirements"], "实现相加")
        self.assertEqual(payload["context"], "保留接口")

    def test_invalid_empty_truncated_filtered_and_refused_responses_fail(self) -> None:
        client = Mock()
        with patch("llm.load_dotenv"):
            fixer = OpenAICodeFixer(api_key="test-key", client=client)
        choices_list = [[], [SimpleNamespace(finish_reason="length")],
                        [SimpleNamespace(finish_reason="content_filter")],
                        [SimpleNamespace(finish_reason="stop", message=SimpleNamespace(refusal="拒绝"))]]
        for content in (None, "not-json", '{"code":"def broken(:", "summary":"bad"}'):
            choices_list.append([SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content=content, refusal=None))])
        for choices in choices_list:
            with self.subTest(choices=choices), self.assertRaises(LLMFixError):
                client.chat.completions.create.return_value = SimpleNamespace(choices=choices)
                fixer.fix(BAD_CODE, report=REPORT, test_result=FAILED_TEST, test_cases=CASES)

    def test_mock_leaves_business_failures_unmodified(self) -> None:
        result = MockCodeFixer().fix(BAD_CODE, report=REPORT, test_result=FAILED_TEST, test_cases=CASES)
        self.assertEqual(result.code, BAD_CODE)
        self.assertIn("代码保持不变", result.summary)


class FixCliTests(unittest.TestCase):
    def test_cli_fix_flag_reports_state_changes_and_remaining_blocker(self) -> None:
        fixer = Mock()
        fixer.fix.return_value = CodeGenerationResult(code=GOOD_CODE, summary="修复加法")
        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--requirements", "实现相加", "--run-tests", "--fix-code"]), \
                patch("main.create_code_generator", return_value=MockCodeGenerator()), \
                patch("main.create_reviewer", return_value=MockReviewer()), \
                patch("main.create_test_generator", return_value=MockTestGenerator()), \
                patch("main.create_code_fixer", return_value=fixer), redirect_stdout(output):
            self.assertEqual(main(), 1)
        self.assertIn("Fix Agent 修复结果", output.getvalue())
        self.assertIn("修复后的代码", output.getvalue())
        self.assertIn(GOOD_CODE, output.getvalue())
        self.assertIn("示例问题", output.getvalue())

    def test_cli_fix_flag_requires_generation_input(self) -> None:
        with patch("sys.argv", ["main.py", "--fix-code"]), \
                patch("main.create_code_fixer") as factory, redirect_stderr(io.StringIO()):
            self.assertEqual(main(), 2)
            factory.assert_not_called()

    def test_cli_fix_error_returns_one(self) -> None:
        fixer = Mock()
        fixer.fix.side_effect = LLMFixError("修复服务不可用")
        error = io.StringIO()
        with patch("sys.argv", ["main.py", "--requirements", "实现相加", "--fix-code"]), \
                patch("main.create_code_generator", return_value=MockCodeGenerator()), \
                patch("main.create_reviewer", return_value=MockReviewer()), \
                patch("main.create_code_fixer", return_value=fixer), redirect_stderr(error):
            self.assertEqual(main(), 1)
        self.assertIn("LLM 代码修复失败", error.getvalue())


if __name__ == "__main__":
    unittest.main()
