"""Python code review agent command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import ReviewAgent, create_review_workflow
from coding_agent import CodingAgent, create_coding_workflow
from file_editor import FileEditError
from fix_agent import FixAgent
from llm import (
    ConfigurationError,
    LLMFixError,
    LLMGenerationError,
    LLMReviewError,
    LLMTestGenerationError,
    create_code_generator,
    create_code_fixer,
    create_reviewer,
    create_test_generator,
)
from models import CodeGenerationResult, ReviewReport, TestExecutionResult
from planner import DEFAULT_REVIEW_TASK
from test_agent import TestAgent, TestRunner
from workflow import WorkflowExecutionError


def read_code_from_stdin() -> str:
    print("请粘贴 Python 代码；单独输入 END 后回车开始审查：")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Agent 审查 Python 代码。",
        epilog="退出码：0 表示已配置的检查通过；1 表示仍有阻断问题或执行失败；2 表示输入或配置错误。",
    )
    parser.add_argument("path", nargs="?", help="可选的 .py 文件；省略时从终端粘贴代码")
    parser.add_argument(
        "--task",
        default=DEFAULT_REVIEW_TASK,
        help="审查任务；Task Planner 会据此决定是否调用 CodeChecker",
    )
    parser.add_argument(
        "--requirements",
        help="根据用户需求生成 Python 代码，再通过 Workflow 进行审查",
    )
    parser.add_argument(
        "--context",
        default="",
        help="代码生成的可选背景或已有代码，需与 --requirements 一起使用",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="对 --requirements 生成的代码自动生成并执行测试",
    )
    parser.add_argument(
        "--fix-code",
        action="store_true",
        help="根据审查问题和测试失败修复生成代码，随后重新验证",
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        help="启用自动修复循环并设置最大修复次数，需与 --fix-code --run-tests 一起使用",
    )
    parser.add_argument(
        "--test-timeout",
        type=float,
        default=10.0,
        help="测试子进程超时秒数，默认 10 秒",
    )
    parser.add_argument(
        "--apply-fixes",
        action="store_true",
        help="生成并应用安全的本地修复补丁，然后重新运行 CodeChecker",
    )
    return parser.parse_args()


def load_code(path_value: str | None) -> str:
    if path_value is None:
        return read_code_from_stdin()

    path = Path(path_value)
    if path.suffix.lower() != ".py":
        raise ValueError("仅支持审查 .py 文件。")
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"文件不存在：{path}") from exc
    except OSError as exc:
        raise ValueError(f"无法读取文件：{exc}") from exc


def _configure_output() -> None:
    """Keep Chinese CLI output readable under legacy Windows code pages."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    # argparse writes --help and errors before returning parsed arguments.
    _configure_output()
    try:
        arguments = parse_arguments()
        if arguments.max_fix_attempts is not None:
            if arguments.max_fix_attempts < 0:
                raise ValueError("最大修复次数必须是非负整数。")
            if arguments.requirements is None or not arguments.fix_code or not arguments.run_tests:
                raise ValueError("--max-fix-attempts 需与 --requirements --fix-code --run-tests 一起使用。")
        if arguments.requirements is not None:
            if arguments.path is not None or arguments.apply_fixes:
                raise ValueError("--requirements 不能与文件路径或 --apply-fixes 同时使用。")
            if not arguments.requirements.strip():
                raise ValueError("用户需求不能为空。")
            test_agent = None
            if arguments.run_tests:
                runner = TestRunner(timeout_seconds=arguments.test_timeout)
                test_agent = TestAgent(create_test_generator(), runner)
            coding_agent = CodingAgent(create_code_generator())
            review_agent = ReviewAgent(create_reviewer())
            fix_agent = FixAgent(create_code_fixer()) if arguments.fix_code else None
            workflow = create_coding_workflow(
                coding_agent, review_agent, test_agent=test_agent, fix_agent=fix_agent,
                max_fix_attempts=arguments.max_fix_attempts,
            )
            run = workflow.run({
                "requirements": arguments.requirements,
                "context": arguments.context,
                "task": arguments.task,
            })
            generation = run.state.require("generation_result", CodeGenerationResult)
            print("\n=== Coding Agent 代码生成结果 ===")
            print(generation.summary)
            print(generation.code)
            if fix_agent is not None:
                fix_result = run.state.get("fix_result")
                if fix_result is not None:
                    print(fix_result.to_text())
                if arguments.max_fix_attempts is not None:
                    print(f"\n自动修复次数：{run.state.get('fix_attempts')}；"
                          f"结束原因：{run.state.get('repair_stop_reason')}")
                    for record in run.state.get("repair_history"):
                        print(f"第 {record.attempt} 轮：{record.status}；"
                              f"执行步骤：{' -> '.join(item.step for item in record.executions)}")
                if run.state.get("code") != generation.code:
                    print("\n=== 修复后的代码 ===")
                    print(run.state.require("code", str))
            print(run.state.require("report", ReviewReport).to_text())
            if test_agent is not None:
                print(run.state.get("test_generation_result").summary)
                test_result = run.state.require("test_result", TestExecutionResult)
                print(test_result.to_text())
            return 0 if run.report.succeeded else 1
        if arguments.fix_code:
            raise ValueError("--fix-code 需与 --requirements 一起使用。")
        if arguments.run_tests:
            raise ValueError("--run-tests 需与 --requirements 一起使用。")
        if arguments.context:
            raise ValueError("--context 需与 --requirements 一起使用。")
        code = load_code(arguments.path)
        if not code.strip():
            raise ValueError("没有收到可审查的 Python 代码。")

        reviewer = create_reviewer()
        if reviewer.is_mock:
            print("[MOCK MODE] 未检测到有效 DASHSCOPE_API_KEY，使用本地固定审查结果。")

        agent = ReviewAgent(reviewer)
        if arguments.apply_fixes:
            if arguments.path is None:
                raise ValueError("--apply-fixes 只能用于传入的 .py 文件。")
            run = agent.review_and_fix(Path(arguments.path), code, task=arguments.task)
            print(run.review.report.to_text())
            final_review = run.review.report
            print("\n=== FileEditor 结果 ===")
            if run.edit.applied:
                print(run.edit.diff.rstrip())
                print(f"补丁已应用到：{run.edit.path}")
                print(f"已修复：{', '.join(run.edit.fixed_rules)}")
                print("\n=== CodeChecker 修改后验证 ===")
                if run.verification_findings:
                    for finding in run.verification_findings:
                        print(f"- [{finding.severity.upper()}] {finding.problem}")
                else:
                    print("验证通过：未发现基础问题。")
                print("\n=== 修改后代码复审 ===")
                final_review = agent.review(run.edit.updated_code, task=arguments.task)
                print(final_review.to_text())
            else:
                print("没有可安全自动应用的修改。")
            for reason in run.edit.skipped_reasons:
                print(f"未自动修复：{reason}")
            return 0 if not final_review.issues and not run.verification_findings else 1
        else:
            workflow = create_review_workflow(agent)
            run = workflow.run({"code": code, "task": arguments.task})
            report = run.state.require("report", ReviewReport)
            print(report.to_text())
            return 0 if run.report.succeeded else 1
    except (ValueError, ConfigurationError, FileEditError) as exc:
        print(f"配置或输入错误：{exc}", file=sys.stderr)
        return 2
    except LLMReviewError as exc:
        print(f"LLM 审查失败：{exc}", file=sys.stderr)
        return 1
    except LLMGenerationError as exc:
        print(f"LLM 代码生成失败：{exc}", file=sys.stderr)
        return 1
    except LLMTestGenerationError as exc:
        print(f"LLM 测试生成失败：{exc}", file=sys.stderr)
        return 1
    except LLMFixError as exc:
        print(f"LLM 代码修复失败：{exc}", file=sys.stderr)
        return 1
    except WorkflowExecutionError as exc:
        cause = exc.__cause__
        if isinstance(cause, LLMReviewError):
            print(f"LLM 审查失败：{cause}", file=sys.stderr)
        elif isinstance(cause, LLMGenerationError):
            print(f"LLM 代码生成失败：{cause}", file=sys.stderr)
        elif isinstance(cause, LLMTestGenerationError):
            print(f"LLM 测试生成失败：{cause}", file=sys.stderr)
        elif isinstance(cause, LLMFixError):
            print(f"LLM 代码修复失败：{cause}", file=sys.stderr)
        else:
            print(f"Workflow 执行失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n操作已取消。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
