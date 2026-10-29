"""Syntax-only retries inside a single Fix Agent execution."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent import CodeReviewAgent
from fix_agent import FixAgent, create_fix_workflow
from llm import LLMFixError, LLMFixSyntaxError, OpenAICodeFixer
from models import (
    CodeGenerationResult, ReviewIssue, ReviewReport, TestCase as GeneratedCase,
    TestExecutionResult as ExecutionResult, TestGenerationResult as GenerationResult,
)
from test_agent import TestAgent as WorkflowTestAgent
from workflow import WorkflowExecutionError, WorkflowState, WorkflowStatus, create_repair_workflow


ORIGINAL = "def add(a, b):\n    return a - b\n"
VALID = "def add(a, b):\n    return a + b\n"
INVALID = "def add(a, b)\n    return a + b\n"
REPORT = ReviewReport(summary="加法有误", issues=[ReviewIssue(
    problem="使用减法", severity="high", reason="结果不符合需求", suggestion="改为加法",
)])
CASES = [GeneratedCase(name="test_add", code="self.assertEqual(module.add(2, 3), 5)")]


def completion(code: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(refusal=None, content=json.dumps({
            "code": code, "summary": "修复加法",
        })),
    )])


class FixSyntaxRetryTests(unittest.TestCase):
    def remote(self, responses):
        client = Mock()
        client.chat.completions.create.side_effect = responses
        with patch("llm.load_dotenv"):
            fixer = OpenAICodeFixer(api_key="test-key", model="qwen-test", client=client)
        return fixer, client

    def test_syntax_error_retries_with_diagnostics_and_original_feedback(self) -> None:
        fixer, client = self.remote([completion(INVALID), completion(VALID)])
        state = WorkflowState({"code": ORIGINAL, "report": REPORT,
                               "requirements": "实现相加", "context": "保留 add 接口", "test_cases": CASES})
        run = create_fix_workflow(FixAgent(fixer)).run(state)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        first, retry = [json.loads(call.kwargs["messages"][1]["content"])
                        for call in client.chat.completions.create.call_args_list]
        self.assertEqual(first["context"], "保留 add 接口")
        self.assertIn("保留 add 接口", retry["context"])
        self.assertIn("syntax_error", retry["context"])
        self.assertIn("invalid_generated_code", retry["context"])
        self.assertIn("def add(a, b)", retry["context"])
        for key in ("code", "requirements", "report", "test_cases", "test_contract"):
            self.assertEqual(first[key], retry[key])
        self.assertEqual(retry["code"], ORIGINAL)
        self.assertEqual(state.get("code"), VALID)
        self.assertEqual(state.version, 1)
        self.assertEqual(len(run.executions), 1)
        self.assertEqual(run.output.generation_attempts, 2)
        self.assertEqual(len(run.output.syntax_errors), 1)

    def test_success_on_the_third_retry_is_allowed(self) -> None:
        fixer, client = self.remote([completion(INVALID)] * 3 + [completion(VALID)])
        run = create_fix_workflow(FixAgent(fixer)).run({"code": ORIGINAL, "report": REPORT})
        self.assertEqual(client.chat.completions.create.call_count, 4)
        self.assertEqual(run.output.generation_attempts, 4)
        self.assertEqual(len(run.output.syntax_errors), 3)
        self.assertEqual(run.state.get("code"), VALID)

    def test_exhaustion_stops_at_three_retries_without_committing(self) -> None:
        fixer, client = self.remote([completion(INVALID)] * 4)
        state = WorkflowState({"code": ORIGINAL, "report": REPORT, "test_cases": CASES})
        with self.assertRaises(WorkflowExecutionError) as caught:
            create_fix_workflow(FixAgent(fixer)).run(state)
        self.assertEqual(client.chat.completions.create.call_count, 4)
        self.assertIsInstance(caught.exception.__cause__, LLMFixSyntaxError)
        self.assertIn("4 次生成调用", str(caught.exception))
        self.assertIn("3 次重试", str(caught.exception))
        self.assertEqual(state.status, WorkflowStatus.FAILED)
        self.assertEqual(state.get("code"), ORIGINAL)
        self.assertIs(state.get("report"), REPORT)
        self.assertEqual(state.get("test_cases"), CASES)
        self.assertIsNone(state.get("fixed_code"))
        self.assertEqual(state.version, 0)
        self.assertEqual(state.history, ())

    def test_injected_provider_syntax_validation_is_also_retryable(self) -> None:
        for construct_inside_provider in (False, True):
            with self.subTest(construct_inside_provider=construct_inside_provider):
                fixer = Mock()
                state = WorkflowState({"code": ORIGINAL, "report": REPORT})
                calls = 0

                def generate(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    self.assertEqual(state.get("code"), ORIGINAL)
                    self.assertEqual(state.version, 0)
                    if calls == 1:
                        if construct_inside_provider:
                            return CodeGenerationResult(code=INVALID, summary="无效代码")
                        return CodeGenerationResult.model_construct(code=INVALID, summary="无效代码")
                    return CodeGenerationResult(code=VALID, summary="有效代码")

                fixer.fix.side_effect = generate
                run = create_fix_workflow(FixAgent(fixer)).run(state)
                self.assertEqual(fixer.fix.call_count, 2)
                self.assertEqual(run.output.code, VALID)
                self.assertEqual(state.version, 1)

    def test_compile_failure_not_detected_by_ast_is_retried(self) -> None:
        fixer = Mock()
        fixer.fix.side_effect = [
            CodeGenerationResult.model_construct(code="return 1", summary="供应商绕过校验"),
            CodeGenerationResult(code=VALID, summary="修复作用域语法"),
        ]
        run = create_fix_workflow(FixAgent(fixer)).run({"code": ORIGINAL, "report": REPORT})
        self.assertEqual(fixer.fix.call_count, 2)
        self.assertIn("outside function", run.output.syntax_errors[0])
        self.assertEqual(run.output.code, VALID)

    def test_non_syntax_errors_are_not_retried(self) -> None:
        for response in (
            LLMFixError("服务失败"),
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                refusal=None, content="not-json",
            ))]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                refusal=None, content=json.dumps({"code": VALID, "summary": " "}),
            ))]),
        ):
            with self.subTest(response=response):
                fixer, client = self.remote([response])
                with self.assertRaises(WorkflowExecutionError) as caught:
                    create_fix_workflow(FixAgent(fixer)).run({"code": ORIGINAL, "report": REPORT})
                self.assertIsInstance(caught.exception.__cause__, LLMFixError)
                self.assertNotIsInstance(caught.exception.__cause__, LLMFixSyntaxError)
                self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_first_pass_and_no_issue_paths_keep_existing_behavior(self) -> None:
        fixer = Mock()
        fixer.fix.return_value = CodeGenerationResult(code=VALID, summary="首次成功")
        run = create_fix_workflow(FixAgent(fixer)).run({"code": ORIGINAL, "report": REPORT})
        fixer.fix.assert_called_once()
        self.assertEqual(run.output.generation_attempts, 1)
        self.assertEqual(run.output.syntax_errors, ())
        fixer.reset_mock()
        run = create_fix_workflow(FixAgent(fixer)).run({
            "code": VALID, "report": ReviewReport(summary="没有问题", issues=[]),
        })
        fixer.fix.assert_not_called()
        self.assertEqual(run.output.generation_attempts, 0)
        self.assertTrue(run.output.skipped)

    def test_retry_configuration_can_disable_retries_and_rejects_invalid_values(self) -> None:
        for value in (-1, True, 1.5, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FixAgent(Mock(), max_syntax_retries=value)
        fixer, client = self.remote([completion(INVALID)])
        with self.assertRaises(WorkflowExecutionError):
            create_fix_workflow(FixAgent(fixer, max_syntax_retries=0)).run({"code": ORIGINAL, "report": REPORT})
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_internal_retries_count_as_one_workflow_repair_attempt(self) -> None:
        fixer, client = self.remote([completion(INVALID), completion(INVALID), completion(VALID)])
        reviewer = Mock()
        reviewer.review.side_effect = lambda code, findings, **kwargs: (
            REPORT if code == ORIGINAL else ReviewReport(summary="没有问题", issues=[])
        )
        generator = Mock()
        generator.generate.return_value = GenerationResult(test_cases=CASES, summary="验证加法")
        runner = Mock()
        runner.run.side_effect = lambda code, cases: ExecutionResult(
            status="passed" if code == VALID else "failed", tests_run=1,
            passed=int(code == VALID), failed=int(code != VALID),
        )
        workflow = create_repair_workflow(
            CodeReviewAgent(reviewer), FixAgent(fixer), WorkflowTestAgent(generator, runner), max_fix_attempts=1,
        )
        run = workflow.run({"code": ORIGINAL})
        self.assertEqual([item.step for item in run.executions], ["review", "test", "fix", "test", "review"])
        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertEqual(run.state.get("fix_attempts"), 1)
        self.assertTrue(run.state.get("repair_succeeded"))
        record = run.state.get("repair_history")[0]
        self.assertEqual(record.fix_result.generation_attempts, 3)
        self.assertEqual(len(record.fix_result.syntax_errors), 2)
        self.assertEqual(record.status, "resolved")
        generator.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
