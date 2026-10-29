"""Original error-message contracts and password validation repair regressions."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent import CodeReviewAgent
from coding_agent import CodingAgent, create_coding_workflow
from fix_agent import FixAgent
from llm import OpenAICodeFixer
from models import (
    CodeGenerationResult, ReviewReport, TestCase as GeneratedCase, TestExecutionResult as ExecutionResult,
    TestFailure as FailureDetail, TestGenerationResult as GenerationResult,
)
from prompt import build_fix_input
from test_agent import TestAgent as WorkflowTestAgent, create_test_workflow
from test_feedback import enrich_test_result, extract_test_expectations


def error_case(name: str, username, password, message: str, *, regex: bool = False) -> GeneratedCase:
    call = f"module.register_user({username!r}, {password!r})"
    if regex:
        code = f"with self.assertRaisesRegex(ValueError, {message!r}):\n    {call}"
    else:
        code = (f"with self.assertRaises(ValueError) as caught:\n    {call}\n"
                f"self.assertEqual(str(caught.exception), {message!r})")
    return GeneratedCase(name=name, code=code)


COMPLEXITY_MESSAGE = "密码长度至少8位且包含大小写字母和数字"
REGISTRATION_CASES = [
    GeneratedCase(name="test_register_success", code=(
        "self.assertEqual(module.register_user(' alice ', 'Secure123!'), "
        "{'username': 'alice', 'registered': True})"
    )),
    error_case("test_missing_username", None, "Secure123", "用户名不能为空"),
    error_case("test_blank_username", " ", "Secure123", "用户名不能为空"),
    error_case("test_empty_password", "alice", "", "^密码不能为空$", regex=True),
    error_case("test_invalid_password_type", "alice", None, "密码不能为空"),
    error_case("test_short_weak_password", "alice", "123456", "密码过于简单"),
    error_case("test_lowercase_weak_password", "alice", "password", "^密码过于简单$", regex=True),
    error_case("test_complex_weak_password", "alice", "Password1", "密码过于简单"),
    error_case("test_nonweak_short_password", "alice", "Abc12", COMPLEXITY_MESSAGE),
    error_case("test_nonweak_missing_digit", "alice", "Abcdefgh", COMPLEXITY_MESSAGE),
]

INITIAL_CODE = '''WEAK_PASSWORDS = {"123456", "password", "Password1"}

def register_user(username, password):
    if not isinstance(username, str) or not username.strip():
        raise ValueError("username is required")
    if not isinstance(password, str) or not password:
        raise ValueError("password is required")
    if (len(password) < 8 or not any(char.isupper() for char in password)
            or not any(char.islower() for char in password)
            or not any(char.isdigit() for char in password)):
        raise ValueError("密码必须包含大小写字母和数字")
    if password in WEAK_PASSWORDS:
        raise ValueError("弱密码")
    return {"username": username.strip(), "registered": True}
'''

NORMALIZED_CODE = INITIAL_CODE.replace("username is required", "用户名不能为空").replace(
    "password is required", "密码不能为空"
).replace("密码必须包含大小写字母和数字", COMPLEXITY_MESSAGE).replace("弱密码", "密码过于简单")

FIXED_CODE = '''WEAK_PASSWORDS = {"123456", "password", "Password1"}

def register_user(username, password):
    if not isinstance(username, str) or not username.strip():
        raise ValueError("用户名不能为空")
    if not isinstance(password, str) or not password:
        raise ValueError("密码不能为空")
    if password in WEAK_PASSWORDS:
        raise ValueError("密码过于简单")
    has_special = any(not char.isalnum() and not char.isspace() for char in password)
    other_categories = sum((
        any(char.isupper() for char in password),
        any(char.islower() for char in password),
        any(char.isdigit() for char in password),
    ))
    if len(password) < 8 or not has_special or other_categories < 2:
        raise ValueError("密码长度至少8位且包含大小写字母和数字")
    return {"username": username.strip(), "registered": True}
'''


class FailureEvidenceTests(unittest.TestCase):
    def test_literal_regex_and_substring_expectations_keep_exact_text(self) -> None:
        case = GeneratedCase(name="test_messages", code=(
            "expected = '密码不能为空。'\n"
            "self.assertEqual(str(error), expected)\n"
            "self.assertRaisesRegex(ValueError, '^密码过于简单$')\n"
            "self.assertIn('复杂度不足', str(error))\n"
            "self.assertEqual(response, {'error': '用户名不能为空'})"
        ))
        expectations = extract_test_expectations(case)
        self.assertEqual([(item.kind, item.expected) for item in expectations], [
            ("exact", "密码不能为空。"), ("regex", "^密码过于简单$"),
            ("contains", "复杂度不足"), ("exact", "用户名不能为空"),
        ])

    def test_dynamic_and_reassigned_values_are_not_evaluated_or_guessed(self) -> None:
        for body in (
            "expected = module.make_message()\nself.assertEqual(str(error), expected)",
            "expected = 'old'\nexpected = module.make_message()\nself.assertEqual(str(error), expected)",
            "first, expected = 'ab'\nself.assertEqual(str(error), expected)",
        ):
            with self.subTest(body=body):
                self.assertEqual(extract_test_expectations(GeneratedCase(name="test_dynamic", code=body)), [])

    def test_failure_context_is_attached_without_mutating_or_masking_the_result(self) -> None:
        case = REGISTRATION_CASES[5]
        original = ExecutionResult(status="failed", tests_run=1, failed=1, failures=[FailureDetail(
            test="test_short_weak_password (builtins.GeneratedTests.test_short_weak_password) (value='123456')",
            traceback="AssertionError: wrong message",
        )])
        result = enrich_test_result(original, [case])
        failure = result.failures[0]
        self.assertEqual(failure.test_name, case.name)
        self.assertEqual(failure.test_code, case.code)
        self.assertEqual(failure.expectations[0].expected, "密码过于简单")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failed, 1)
        self.assertEqual(original.failures[0].test_code, "")
        self.assertEqual(original.test_expectations, {})

    def test_saved_failure_evidence_survives_when_cases_are_unavailable(self) -> None:
        case = REGISTRATION_CASES[5]
        saved = enrich_test_result(ExecutionResult(
            status="failed", failures=[FailureDetail(test=case.name, traceback="failed")]
        ), [case])
        payload = json.loads(build_fix_input(INITIAL_CODE, None, saved, []))
        evidence = payload["failed_test_contexts"][0]
        self.assertEqual(evidence["test_code"], case.code)
        self.assertEqual(evidence["expectations"][0]["expected"], "密码过于简单")
        self.assertIn(case.name, payload["test_result"]["test_expectations"])

    def test_fix_input_distinguishes_positive_and_negative_assertions(self) -> None:
        case = GeneratedCase(name="test_negation", code="self.assertNotIn('复杂度错误', str(error))")
        payload = json.loads(build_fix_input(INITIAL_CODE, None, None, [case]))
        expectation = payload["test_contract"][0]["expectations"][0]
        self.assertEqual(expectation["assertion"], "assertNotIn")
        self.assertEqual(expectation["kind"], "contains")
        self.assertEqual(expectation["expected"], "复杂度错误")

    def test_runtime_errors_also_retain_the_original_case(self) -> None:
        case = REGISTRATION_CASES[5]
        result = enrich_test_result(ExecutionResult(status="failed", errors=1, error_details=[
            FailureDetail(test=case.name, traceback="NameError")
        ]), [case])
        self.assertEqual(result.error_details[0].test_code, case.code)
        self.assertEqual(result.error_details[0].expectations[0].expected, "密码过于简单")


class RegistrationRepairTests(unittest.TestCase):
    def test_test_agent_preserves_expected_error_messages_from_real_failures(self) -> None:
        generator = Mock()
        generator.generate.return_value = GenerationResult(
            test_cases=REGISTRATION_CASES, summary="注册行为与错误提示"
        )
        run = create_test_workflow(WorkflowTestAgent(generator)).run({"code": INITIAL_CODE})
        result = run.state.get("test_result")
        self.assertEqual(result.status, "failed")
        failures = {failure.test_name: failure for failure in result.failures}
        self.assertEqual(failures["test_missing_username"].expectations[0].expected, "用户名不能为空")
        self.assertEqual(failures["test_short_weak_password"].expectations[0].expected, "密码过于简单")
        self.assertEqual(failures["test_short_weak_password"].test_code, REGISTRATION_CASES[5].code)
        self.assertIn("密码必须包含大小写字母和数字", failures["test_short_weak_password"].traceback)

    def test_workflow_repairs_messages_then_order_and_retests_until_all_pass(self) -> None:
        requirements = (
            "实现 register_user，最小长度8，必须包含非字母数字且非空白的特殊字符，"
            "并在大写、小写、数字中至少满足两类；错误提示遵循原测试，弱密码检测先于复杂度检查"
        )
        coding = Mock()
        coding.generate.return_value = CodeGenerationResult(code=INITIAL_CODE, summary="注册回归输入")
        reviewer = Mock()
        reviewer.review.return_value = ReviewReport(summary="本回归通过测试反馈定位业务问题", issues=[])
        generator = Mock()
        generator.generate.return_value = GenerationResult(
            test_cases=REGISTRATION_CASES, summary="固定注册契约，复验不得改写"
        )
        client = Mock()
        requests: list[dict] = []

        def respond(**request):
            payload = json.loads(request["messages"][1]["content"])
            requests.append(payload)
            self.assertEqual(payload["requirements"], requirements)
            self.assertNotIn("先检测弱密码", request["messages"][0]["content"])
            self.assertNotIn("register_user", request["messages"][0]["content"])
            self.assertEqual(payload["test_cases"], [case.model_dump() for case in REGISTRATION_CASES])
            self.assertEqual(payload["test_result"]["status"], "failed")
            failures = {item["test_name"]: item for item in payload["failed_test_contexts"]}
            short = failures["test_short_weak_password"]
            self.assertEqual(short["test_code"], REGISTRATION_CASES[5].code)
            self.assertEqual(short["expectations"][0]["expected"], "密码过于简单")
            if len(requests) == 1:
                self.assertIn("test_missing_username", failures)
                repaired = NORMALIZED_CODE
            else:
                self.assertNotIn("test_missing_username", failures)
                self.assertEqual(set(failures), {"test_short_weak_password", "test_lowercase_weak_password"})
                repaired = FIXED_CODE
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(refusal=None, content=json.dumps({
                    "code": repaired, "summary": "统一错误提示" if len(requests) == 1 else "调整弱密码检测顺序",
                }, ensure_ascii=False)),
            )])

        client.chat.completions.create.side_effect = respond
        with patch("llm.load_dotenv"):
            fixer = OpenAICodeFixer(api_key="test-key", model="qwen-test", client=client)
        workflow = create_coding_workflow(
            CodingAgent(coding), CodeReviewAgent(reviewer),
            test_agent=WorkflowTestAgent(generator), fix_agent=FixAgent(fixer), max_fix_attempts=3,
        )
        run = workflow.run({"requirements": requirements})
        state = run.state
        self.assertEqual(state.get("repair_stop_reason"), "resolved")
        self.assertTrue(state.get("repair_succeeded"))
        self.assertEqual(state.get("fix_attempts"), 2)
        self.assertEqual(state.get("code"), FIXED_CODE)
        self.assertEqual(state.get("test_result").status, "passed")
        self.assertEqual(state.get("test_result").tests_run, 10)
        self.assertEqual(state.get("test_result").passed, 10)
        self.assertEqual(state.get("test_result").failed, 0)
        self.assertEqual(state.get("test_result").errors, 0)
        self.assertEqual([round.status for round in state.get("repair_history")], ["unresolved", "resolved"])
        self.assertEqual(state.get("test_cases"), REGISTRATION_CASES)
        generator.generate.assert_called_once()
        self.assertEqual(len(requests), 2)
        self.assertEqual([execution.step for execution in run.executions],
                         ["coding", "review", "test", "fix", "test", "review", "fix", "test", "review"])

    def test_weak_passwords_beat_complexity_but_basic_input_validation_stays_first(self) -> None:
        generator = Mock()
        generator.generate.return_value = GenerationResult(test_cases=REGISTRATION_CASES, summary="复验")
        run = create_test_workflow(WorkflowTestAgent(generator)).run({"code": FIXED_CODE})
        self.assertEqual(run.output.status, "passed")
        self.assertEqual(run.output.passed, len(REGISTRATION_CASES))

    def test_special_character_and_any_two_other_categories_are_required(self) -> None:
        valid = ("Abcdefg!", "ABCD123!", "abcde12!", "Abcd123!", "_abcde12")
        invalid = ("Abcd1234", "abcdefg!", "ABCDEFG!", "1234567!", "!@#$%^&*", "Abcd123 ", "Abc12!")
        cases = [GeneratedCase(
            name=f"test_accept_combination_{index}",
            code=(f"self.assertEqual(module.register_user('alice', {password!r}), "
                  "{'username': 'alice', 'registered': True})"),
        ) for index, password in enumerate(valid)]
        cases += [error_case(f"test_reject_combination_{index}", "alice", password, COMPLEXITY_MESSAGE)
                  for index, password in enumerate(invalid)]
        generator = Mock()
        generator.generate.return_value = GenerationResult(test_cases=cases, summary="密码字符类别边界")
        run = create_test_workflow(WorkflowTestAgent(generator)).run({"code": FIXED_CODE})
        self.assertEqual(run.output.status, "passed")
        self.assertEqual(run.output.tests_run, len(valid) + len(invalid))
        self.assertEqual(run.output.passed, len(valid) + len(invalid))
        self.assertEqual(run.output.failed, 0)
        self.assertEqual(run.output.errors, 0)


if __name__ == "__main__":
    unittest.main()
