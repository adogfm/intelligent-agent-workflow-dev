"""Regression contracts for task-specific policies in actual fix requests.

Only the SDK client is mocked: OpenAICodeFixer builds the real messages.
These tests detect conflicting prompt instructions, not real-model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llm import OpenAICodeFixer
from models import ReviewIssue, ReviewReport, TestCase as GeneratedCase


SOURCE = '''WEAK_PASSWORDS = {"123456"}

def register_user(username, password):
    if not isinstance(password, str) or not password:
        raise ValueError("密码不能为空")
    if password in WEAK_PASSWORDS:
        raise ValueError("弱密码")
    if len(password) < 8:
        raise ValueError("旧长度错误")
    return {"username": username, "registered": True}
'''
CONTEXT = "离线演示任务；按明确业务需求和原始验收测试修复，不扩展认证功能。"

# Removed unconditional directives, each contradicted by a scenario below.
SPECIAL_RULE = "修复 register_user 的密码复杂度判断时，必须包含特殊字符，并在大写、小写、数字三类中至少满足两类。"
CATEGORY_RULE = "特殊字符是非字母数字且非空白的字符；不能将空格视为特殊字符，也不能仅统计任意三类或强制大写、小写、数字全部存在。"
WEAK_FIRST_RULE = "涉及密码校验时，在类型和空值等基础校验后先检测弱密码，再做密码长度、大小写、数字等复杂度检查。"
WEAK_ERROR_RULE = "密码同时属于弱密码且不满足复杂度时，应按原测试返回弱密码提示，不能先返回复杂度错误。"
KEEP_OLD_RULE = "保持现有最小长度限制与错误提示原文，不因字符类别规则调整而改写文案。"


def accepts(password):
    return (
        f"self.assertEqual(module.register_user('alice', {password!r}), "
        "{'username': 'alice', 'registered': True})"
    )


def rejects(password, message):
    return (
        "with self.assertRaises(ValueError) as caught:\n"
        f"    module.register_user('alice', {password!r})\n"
        f"self.assertEqual(str(caught.exception), {message!r})"
    )


@dataclass(frozen=True)
class PolicyScenario:
    requirements: str
    case_code: str
    conflicts: tuple[str, ...]


SCENARIOS = {
    "length_only": PolicyScenario(
        "register_user 的密码仅检查非空字符串且长度至少6，不要求字符类别，不增加弱密码黑名单；abcdef 应可注册。",
        accepts("abcdef"), (SPECIAL_RULE, WEAK_FIRST_RULE),
    ),
    "all_three": PolicyScenario(
        "register_user 要求至少8位，必须同时含大写、小写和数字；不要求特殊字符。Abcd1234 必须接受，Abcdefg! 必须拒绝。",
        accepts("Abcd1234") + "\n" + rejects("Abcdefg!", "必须包含大小写字母和数字"),
        (SPECIAL_RULE, CATEGORY_RULE),
    ),
    "special_only": PolicyScenario(
        "register_user 要求至少8位且含特殊字符，不要求字母或数字；!@#$%^&* 必须接受。",
        accepts("!@#$%^&*"), (SPECIAL_RULE,),
    ),
    "any_three": PolicyScenario(
        "register_user 要求至少8位，在大写、小写、数字、特殊字符四类中任选三类；特殊字符不是必需，Abcd1234 应被接受。",
        accepts("Abcd1234"), (SPECIAL_RULE, CATEGORY_RULE),
    ),
    "space_is_special": PolicyScenario(
        "register_user 要求至少8位并含特殊字符；本演示明确允许空格并将空格计作特殊字符，Abcd123 后跟一个空格应被接受。",
        accepts("Abcd123 "), (CATEGORY_RULE,),
    ),
    "complexity_first": PolicyScenario(
        "register_user 先校验密码长度至少8，再检查弱密码黑名单；123456 同时过短且属于弱密码时，必须返回长度不足，不能返回弱密码提示。",
        rejects("123456", "长度不足"), (WEAK_FIRST_RULE, WEAK_ERROR_RULE),
    ),
    "changed_length": PolicyScenario(
        "修改 register_user：将原最小长度8改为12，将原长度错误文案改为账户口令至少12字符。；Abcd123! 必须按新文案拒绝。",
        rejects("Abcd123!", "账户口令至少12字符。"), (KEEP_OLD_RULE,),
    ),
}


class FixPromptRequirementsTests(unittest.TestCase):
    def capture_request(self, requirements, cases, *, code=SOURCE):
        client = Mock()
        # A transport fixture only; this response is not asserted to satisfy the task.
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                refusal=None, content=json.dumps({"code": code, "summary": "离线请求捕获响应"}),
            )),
        ])
        with patch("llm.load_dotenv"), patch("llm.OpenAI", side_effect=AssertionError("Real SDK client forbidden")):
            fixer = OpenAICodeFixer(
                api_key="offline-test-key", base_url="https://example.invalid/v1",
                model="offline-prompt-contract", client=client,
            )
            fixer.fix(
                code, report=ReviewReport(summary="按明确需求修复", issues=[ReviewIssue(
                    problem="实现与当前业务需求不符", severity="high", reason="需遵守本任务验收标准",
                    suggestion="只修改与需求不符的行为", category="requirement", requirement_ref=requirements,
                )]), test_result=None, test_cases=cases, requirements=requirements, context=CONTEXT,
            )
        client.chat.completions.create.assert_called_once()
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual([message["role"] for message in request["messages"]], ["system", "user"])
        return request["messages"][0]["content"], json.loads(request["messages"][1]["content"])

    def assert_no_conflicting_global_rules(self, name):
        scenario = SCENARIOS[name]
        case = GeneratedCase(name=f"test_{name}", code=scenario.case_code)
        system, payload = self.capture_request(scenario.requirements, [case])
        self.assertEqual(payload["requirements"], scenario.requirements)
        self.assertEqual(payload["test_cases"][0]["code"], scenario.case_code)
        conflicts = [rule for rule in scenario.conflicts if rule in system]
        self.assertEqual(conflicts, [], "实际 system 消息不应保留与当前用户需求相反的无条件业务规则")

    def test_original_requirements_context_and_tests_are_forwarded(self):
        for name, scenario in SCENARIOS.items():
            with self.subTest(name=name):
                case = GeneratedCase(name=f"test_{name}", code=scenario.case_code)
                _, payload = self.capture_request(scenario.requirements, [case])
                self.assertEqual(payload["requirements"], scenario.requirements)
                self.assertEqual(payload["context"], CONTEXT)
                self.assertEqual(payload["code"], SOURCE)
                self.assertEqual(payload["test_cases"], [case.model_dump()])
                self.assertEqual(payload["test_contract"][0]["code"], case.code)
                self.assertEqual(payload["report"]["issues"][0]["requirement_ref"], scenario.requirements)

    def test_exact_and_regex_error_contracts_are_preserved(self):
        message = "账户口令至少12字符。"
        cases = [
            GeneratedCase(name="test_exact", code=rejects("Abcd123!", message)),
            GeneratedCase(name="test_regex", code=(
                "with self.assertRaisesRegex(ValueError, '^账户口令至少12字符。$'):\n"
                "    module.register_user('alice', 'Abcd123!')"
            )),
        ]
        _, payload = self.capture_request(SCENARIOS["changed_length"].requirements, cases)
        exact, regex = [item["expectations"][0] for item in payload["test_contract"]]
        self.assertEqual((exact["kind"], exact["expected"]), ("exact", message))
        self.assertEqual((regex["kind"], regex["expected"]), ("regex", "^账户口令至少12字符。$"))

    def test_explicit_matching_legacy_policy_remains_in_task_payload(self):
        requirements = "register_user 至少8位，必须含特殊字符及大写小写数字中至少两类；弱密码检查先于复杂度，错误提示沿用原测试。"
        case = GeneratedCase(name="test_matching_policy", code=accepts("Abcdefg!"))
        _, payload = self.capture_request(requirements, [case])
        self.assertEqual(payload["requirements"], requirements)
        self.assertEqual(payload["test_contract"][0]["code"], case.code)

    def test_explicit_requirement_overrides_old_test_without_rewriting_evidence(self):
        requirements = SCENARIOS["changed_length"].requirements
        old_case = GeneratedCase(name="test_old_message", code=rejects("Abcd123!", "旧长度错误"))
        system, payload = self.capture_request(requirements, [old_case])
        self.assertIn("业务规则以 requirements 中的明确用户需求为准", system)
        self.assertIn("若与明确用户需求冲突，以用户需求为准", system)
        self.assertIn("在 summary 中说明冲突，不改写测试或宣称冲突已解决", system)
        self.assertIn("用户明确要求修改提示时遵循新要求", system)
        self.assertEqual(payload["requirements"], requirements)
        self.assertEqual(payload["test_cases"], [old_case.model_dump()])
        self.assertEqual(payload["test_contract"][0]["expectations"][0]["expected"], "旧长度错误")

    def test_non_registration_task_receives_no_password_policy(self):
        requirements = "实现 add(a, b)，返回两个数的和。"
        code = "def add(a, b):\n    return a - b\n"
        case = GeneratedCase(name="test_add", code="self.assertEqual(module.add(2, 3), 5)")
        system, payload = self.capture_request(requirements, [case], code=code)
        for domain_term in ("register_user", "密码", "特殊字符"):
            with self.subTest(domain_term=domain_term):
                self.assertNotIn(domain_term, system)
        self.assertIn("用户未明确规定的行为参考原测试契约和已有功能，不自行添加业务限制", system)
        self.assertEqual(payload["requirements"], requirements)
        self.assertEqual(payload["code"], code)
        self.assertEqual(payload["test_cases"], [case.model_dump()])

    def test_length_only_policy_does_not_gain_category_or_blacklist_rules(self):
        self.assert_no_conflicting_global_rules("length_only")

    def test_three_required_categories_do_not_gain_special_character_rule(self):
        self.assert_no_conflicting_global_rules("all_three")

    def test_special_character_only_policy_does_not_gain_other_categories(self):
        self.assert_no_conflicting_global_rules("special_only")

    def test_any_three_categories_policy_does_not_force_special_characters(self):
        self.assert_no_conflicting_global_rules("any_three")

    def test_explicit_space_policy_is_not_overridden(self):
        self.assert_no_conflicting_global_rules("space_is_special")

    def test_explicit_complexity_first_order_is_not_overridden(self):
        self.assert_no_conflicting_global_rules("complexity_first")

    def test_requested_length_and_message_changes_are_not_frozen(self):
        self.assert_no_conflicting_global_rules("changed_length")


if __name__ == "__main__":
    unittest.main()
