"""Offline generation and coding-to-review workflow tests."""

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
from llm import (
    LLMGenerationError,
    MockCodeGenerator,
    MockReviewer,
    OpenAICodeGenerator,
    create_code_generator,
    parse_code_generation,
)
from main import main
from models import CodeGenerationResult, ReviewReport
from workflow import Agent, WorkflowExecutionError, WorkflowState, WorkflowStatus


GENERATED_CODE = "def add(a: int, b: int) -> int:\n    return a + b\n"


class CodingAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = Mock()
        self.generator.generate.return_value = CodeGenerationResult(
            code=GENERATED_CODE, summary="实现整数相加。"
        )
        self.agent = CodingAgent(self.generator)

    def test_node_commits_artifact_and_preserves_input_state(self) -> None:
        state = WorkflowState({"requirements": "  实现整数相加  ", "context": "函数名为 add"})

        run = create_coding_workflow(self.agent).run(state)

        self.assertIsInstance(self.agent, Agent)
        self.generator.generate.assert_called_once_with("实现整数相加", context="函数名为 add")
        self.assertEqual(state.get("requirements"), "  实现整数相加  ")
        self.assertEqual(state.get("code"), GENERATED_CODE)
        self.assertEqual(run.output, state.get("generation_result"))
        self.assertEqual(state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(state.version, 1)
        self.assertEqual(state.history[0].source, "coding")
        self.assertEqual(state.history[0].updated_keys, ("code", "generation_result"))

    def test_generated_code_is_consumed_by_existing_review_node(self) -> None:
        reviewer = Mock()
        reviewer.review.return_value = ReviewReport(summary="审查完成", issues=[])
        workflow = create_coding_workflow(self.agent, CodeReviewAgent(reviewer))

        run = workflow.run({"requirements": "实现整数相加", "task": "只检查命名"})

        self.assertEqual([item.step for item in run.executions], ["coding", "review"])
        self.assertEqual(reviewer.review.call_args.args, (GENERATED_CODE, []))
        self.assertEqual(reviewer.review.call_args.kwargs["task"], "只检查命名")
        self.assertEqual(run.output, run.state.get("report"))
        self.assertEqual(run.state.get("generation_result").code, GENERATED_CODE)
        self.assertEqual(run.state.version, 2)

    def test_invalid_input_fails_before_generation(self) -> None:
        for values in ({}, {"requirements": " "}, {"requirements": 123},
                       {"requirements": "实现相加", "context": None}):
            with self.subTest(values=values):
                state = WorkflowState(values)
                with self.assertRaises(WorkflowExecutionError):
                    create_coding_workflow(self.agent).run(state)
                self.assertEqual(state.status, WorkflowStatus.FAILED)
                self.assertEqual(state.version, 0)
                self.assertIsNone(state.get("code"))
        self.generator.generate.assert_not_called()

    def test_provider_failure_preserves_previous_code_and_skips_review(self) -> None:
        self.generator.generate.side_effect = LLMGenerationError("服务不可用")
        reviewer = Mock()
        state = WorkflowState({"requirements": "实现相加", "code": "old = True"})

        with self.assertRaises(WorkflowExecutionError) as caught:
            create_coding_workflow(self.agent, CodeReviewAgent(reviewer)).run(state)

        self.assertIs(caught.exception.state, state)
        self.assertIsInstance(caught.exception.__cause__, LLMGenerationError)
        self.assertEqual(state.get("code"), "old = True")
        self.assertEqual(state.error, "服务不可用")
        self.assertEqual(state.current_step, "coding")
        self.assertEqual(state.version, 0)
        reviewer.review.assert_not_called()

    def test_invalid_custom_provider_output_is_rejected_before_commit(self) -> None:
        invalid = CodeGenerationResult.model_construct(code="def broken(:", summary="invalid")
        for result in (invalid, "print('not a result')"):
            with self.subTest(result=result):
                self.generator.generate.return_value = result
                state = WorkflowState({"requirements": "实现相加"})
                with self.assertRaises(WorkflowExecutionError) as caught:
                    create_coding_workflow(self.agent).run(state)
                self.assertIsInstance(caught.exception.__cause__, LLMGenerationError)
                self.assertEqual(state.version, 0)
                self.assertIsNone(state.get("code"))


class CodeGeneratorTests(unittest.TestCase):
    def test_json_and_json_fences_are_parsed(self) -> None:
        payload = json.dumps({"code": GENERATED_CODE, "summary": "实现相加"})
        for content in (payload, f"```json\n{payload}\n```"):
            with self.subTest(content=content):
                self.assertEqual(parse_code_generation(content).code, GENERATED_CODE)

    def test_invalid_schema_and_invalid_python_are_rejected(self) -> None:
        payloads = [
            None, "", "not-json", '{}',
            '{"code": 123, "summary": "wrong type"}',
            '{"code": " ", "summary": "empty code"}',
            '{"code": "# comment only", "summary": "no implementation"}',
            '{"code": "def broken(:", "summary": "invalid syntax"}',
            '{"code": "pass", "summary": " "}',
        ]
        for content in payloads:
            with self.subTest(content=content):
                with self.assertRaises(LLMGenerationError):
                    parse_code_generation(content)

    def test_missing_or_placeholder_key_selects_offline_mock(self) -> None:
        for key in ("", "your_api_key_here"):
            with self.subTest(key=key), patch("llm.load_dotenv"), patch.dict(
                os.environ, {"DASHSCOPE_API_KEY": key}, clear=True
            ):
                generator = create_code_generator()
                self.assertIsInstance(generator, MockCodeGenerator)
                result = generator.generate("实现相加\n包括 ' 引号", context="已有接口")
                self.assertIn("[MOCK MODE]", result.summary)
                self.assertIn("未实现", result.summary)

    def test_configured_keys_select_real_generator(self) -> None:
        for variable in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
            with self.subTest(variable=variable), patch("llm.load_dotenv"), patch.dict(
                os.environ, {variable: "test-key"}, clear=True
            ), patch("llm.OpenAICodeGenerator") as real_generator:
                self.assertIs(create_code_generator(), real_generator.return_value)
                real_generator.assert_called_once_with(api_key="test-key")

    def test_real_provider_forwards_requirements_and_context(self) -> None:
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content=json.dumps({"code": GENERATED_CODE, "summary": "实现相加"}),
                refusal=None,
            ))
        ])
        with patch("llm.load_dotenv"):
            generator = OpenAICodeGenerator(
                api_key="test-key", base_url="https://example.com/v1", model="qwen-test",
                client=client,
            )

        result = generator.generate("实现相加", context="函数名为 add")

        self.assertEqual(result.code, GENERATED_CODE)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "qwen-test")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        self.assertEqual(json.loads(request["messages"][1]["content"]), {
            "requirements": "实现相加", "context": "函数名为 add",
        })

    def test_truncated_filtered_refused_or_empty_responses_fail(self) -> None:
        choices = [
            [],
            [SimpleNamespace(finish_reason="length", message=SimpleNamespace())],
            [SimpleNamespace(finish_reason="content_filter", message=SimpleNamespace())],
            [SimpleNamespace(finish_reason="stop", message=SimpleNamespace(refusal="拒绝"))],
            [SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=None))],
        ]
        client = Mock()
        with patch("llm.load_dotenv"):
            generator = OpenAICodeGenerator(api_key="test-key", client=client)
        for response_choices in choices:
            with self.subTest(choices=response_choices):
                client.chat.completions.create.return_value = SimpleNamespace(choices=response_choices)
                with self.assertRaises(LLMGenerationError):
                    generator.generate("实现相加")


class CodingCliTests(unittest.TestCase):
    def test_requirements_cli_reports_offline_generation_and_mock_blocker(self) -> None:
        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--requirements", "实现相加"]), patch(
            "main.create_code_generator", return_value=MockCodeGenerator()
        ), patch("main.create_reviewer", return_value=MockReviewer()), redirect_stdout(output):
            self.assertEqual(main(), 1)
        self.assertIn("Coding Agent", output.getvalue())
        self.assertIn("get_requirements", output.getvalue())
        self.assertIn("代码审查结果", output.getvalue())
        self.assertIn("示例问题", output.getvalue())

    def test_invalid_generation_cli_options_fail_before_loading_providers(self) -> None:
        args = [
            ["--requirements", " "],
            ["example.py", "--requirements", "实现相加"],
            ["--requirements", "实现相加", "--apply-fixes"],
            ["--context", "没有需求"],
        ]
        with patch("main.create_code_generator") as factory:
            for arguments in args:
                with self.subTest(arguments=arguments), patch("sys.argv", ["main.py", *arguments]), \
                        redirect_stderr(io.StringIO()):
                    self.assertEqual(main(), 2)
        factory.assert_not_called()

    def test_generation_error_has_exit_code_one(self) -> None:
        generator = Mock()
        generator.generate.side_effect = LLMGenerationError("服务不可用")
        error = io.StringIO()
        with patch("sys.argv", ["main.py", "--requirements", "实现相加"]), patch(
            "main.create_code_generator", return_value=generator
        ), patch("main.create_reviewer", return_value=MockReviewer()), redirect_stderr(error):
            self.assertEqual(main(), 1)
        self.assertIn("LLM 代码生成失败", error.getvalue())


if __name__ == "__main__":
    unittest.main()
