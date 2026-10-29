"""Python code generation node for the stateful workflow engine."""

from __future__ import annotations

import json

from pydantic import ValidationError

from agent import ReviewAgent
from llm import CodeGenerator, LLMCodeSyntaxError, LLMGenerationError
from models import CodeGenerationResult, generated_code_syntax_error
from workflow import END, Agent, AgentResult, BaseAgent, StateView, WorkflowEngine, create_repair_workflow


class CodingAgent(BaseAgent):
    """Generate Python source and publish it for downstream workflow nodes."""

    name = "coding"

    def __init__(self, generator: CodeGenerator, *, max_syntax_retries: int = 3) -> None:
        if isinstance(max_syntax_retries, bool) or not isinstance(max_syntax_retries, int) or max_syntax_retries < 0:
            raise ValueError("max_syntax_retries 必须是非负整数。")
        self._generator = generator
        self._max_syntax_retries = max_syntax_retries

    def generate(self, requirements: str, *, context: str = "") -> CodeGenerationResult:
        """Generate a validated artifact with bounded syntax-only retries.

        ``max_syntax_retries=0`` disables retries. Each retry retains the
        original requirement/context and adds only the latest invalid source
        and compiler diagnostic. No source is executed or published here.
        """

        if not isinstance(requirements, str):
            raise TypeError("用户需求必须是 str。")
        if not requirements.strip():
            raise ValueError("用户需求不能为空。")
        if not isinstance(context, str):
            raise TypeError("生成上下文必须是 str。")

        retry_context = context
        for retry in range(self._max_syntax_retries + 1):
            try:
                return self._generate_once(requirements.strip(), context=retry_context)
            except LLMCodeSyntaxError as exc:
                if retry == self._max_syntax_retries:
                    raise LLMCodeSyntaxError(
                        f"生成代码在 {retry + 1} 次生成调用后仍未通过 Python 语法检查"
                        f"（最多 {self._max_syntax_retries} 次重试）：{exc.reason}",
                        generated_code=exc.generated_code, reason=exc.reason,
                    ) from exc
                retry_context = context + "\n\n以下为上一次生成的语法失败诊断数据，请修正语法并重新返回完整代码：\n" + json.dumps({
                    "syntax_retry": retry + 1, "syntax_error": exc.reason,
                    "invalid_generated_code": exc.generated_code,
                }, ensure_ascii=False)
        raise AssertionError("语法重试循环未返回结果")

    def _generate_once(self, requirements: str, *, context: str) -> CodeGenerationResult:
        try:
            result = self._generator.generate(requirements, context=context)
            if not isinstance(result, CodeGenerationResult):
                raise LLMGenerationError("CodeGenerator 必须返回 CodeGenerationResult。")
            # Revalidate injected providers as well as real LLM responses before commit.
            return CodeGenerationResult.model_validate(result.model_dump())
        except ValidationError as exc:
            syntax = generated_code_syntax_error(exc)
            if syntax is not None:
                raise LLMCodeSyntaxError(
                    f"代码生成结果无效：{exc}", generated_code=syntax.code, reason=syntax.reason,
                ) from exc
            raise LLMGenerationError(f"代码生成结果无效：{exc}") from exc

    def run(self, state: StateView) -> AgentResult:
        """Read requirements/context and return an atomic state update."""

        requirements = state.require("requirements", str)
        context = state.get("context", "")
        result = self.generate(requirements, context=context)
        return AgentResult(
            updates={"code": result.code, "generation_result": result},
            output=result,
        )


def create_coding_workflow(
    agent: CodingAgent,
    review_agent: ReviewAgent | None = None,
    *,
    test_agent: Agent | None = None,
    fix_agent: Agent | None = None,
    max_fix_attempts: int | None = None,
) -> WorkflowEngine:
    """Build a generation workflow with optional review, testing and repair.

    Input state requires ``requirements`` and may include ``context`` and the
    review ``task``. The final output belongs to the last enabled node; previous
    artifacts remain in state. Changed repairs trigger one review/retest pass;
    existing test cases are reused rather than generated again. Supplying
    max_fix_attempts enables the bounded Review -> Fix -> Test -> Review loop.
    """

    if max_fix_attempts is not None:
        if review_agent is None or fix_agent is None or test_agent is None:
            raise ValueError("自动修复循环需要 Review、Fix 和 Test 节点。")
        engine = create_repair_workflow(
            review_agent, fix_agent, test_agent, max_fix_attempts=max_fix_attempts
        )
        engine.add_step("coding", agent, next_step="review").set_entrypoint("coding")
        return engine
    if fix_agent is not None and review_agent is None and test_agent is None:
        raise ValueError("Fix 节点至少需要 Review 或 Test 节点提供反馈。")
    engine = WorkflowEngine("coding")
    after_test = "fix" if fix_agent is not None else END
    after_review = "test" if test_agent is not None else after_test
    engine.add_step("coding", agent, next_step="review" if review_agent is not None else after_review)
    if review_agent is not None:
        engine.add_step("review", review_agent, next_step=after_review)
    if test_agent is not None:
        engine.add_step("test", test_agent, next_step=after_test)
    if fix_agent is not None:
        verification = "verify_review" if review_agent is not None else "verify_test"
        engine.add_step(
            "fix", fix_agent,
            router=lambda state, result: verification if state.get("fix_changed") else END,
        )
        if review_agent is not None:
            engine.add_step("verify_review", review_agent,
                            next_step="verify_test" if test_agent is not None else END)
        if test_agent is not None:
            engine.add_step("verify_test", test_agent)
    return engine
