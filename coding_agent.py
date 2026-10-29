"""Python code generation node for the stateful workflow engine."""

from __future__ import annotations

from pydantic import ValidationError

from agent import ReviewAgent
from llm import CodeGenerator, LLMCodeSyntaxError, LLMGenerationError
from models import CodeGenerationResult, generated_code_syntax_error
from workflow import END, Agent, AgentResult, BaseAgent, StateView, WorkflowEngine, create_repair_workflow


class CodingAgent(BaseAgent):
    """Generate Python source and publish it for downstream workflow nodes."""

    name = "coding"

    def __init__(self, generator: CodeGenerator) -> None:
        self._generator = generator

    def generate(self, requirements: str, *, context: str = "") -> CodeGenerationResult:
        """Generate a validated artifact, also usable outside a workflow."""

        if not isinstance(requirements, str):
            raise TypeError("用户需求必须是 str。")
        if not requirements.strip():
            raise ValueError("用户需求不能为空。")
        if not isinstance(context, str):
            raise TypeError("生成上下文必须是 str。")

        try:
            result = self._generator.generate(requirements.strip(), context=context)
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
