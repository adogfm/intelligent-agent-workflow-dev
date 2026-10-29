"""Representative offline benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    description: str
    code: str
    expected_rule_ids: tuple[str, ...]


BENCHMARK_CASES = (
    BenchmarkCase(
        case_id="syntax_error",
        description="函数定义缺少参数列表闭合",
        code="def broken(:\n    pass\n",
        expected_rule_ids=("syntax-error",),
    ),
    BenchmarkCase(
        case_id="bare_except",
        description="异常处理使用裸 except",
        code="try:\n    work()\nexcept:\n    pass\n",
        expected_rule_ids=("bare-except",),
    ),
    BenchmarkCase(
        case_id="true_division_zero",
        description="普通除法使用字面量零作为分母",
        code="ratio = total / 0\n",
        expected_rule_ids=("division-by-zero",),
    ),
    BenchmarkCase(
        case_id="floor_division_zero",
        description="整除使用负浮点零作为分母",
        code="pages = total // -0.0\n",
        expected_rule_ids=("division-by-zero",),
    ),
    BenchmarkCase(
        case_id="modulo_zero",
        description="取模使用正号零作为右操作数",
        code="bucket = item_id % +0\n",
        expected_rule_ids=("division-by-zero",),
    ),
    BenchmarkCase(
        case_id="multiple_findings",
        description="同一文件同时包含零除和裸 except",
        code="try:\n    value = 10 / 0\nexcept:\n    value = None\n",
        expected_rule_ids=("division-by-zero", "bare-except"),
    ),
    BenchmarkCase(
        case_id="clean_baseline",
        description="合法异常处理与动态非零检查",
        code=(
            "def divide(total, count):\n"
            "    if count == 0:\n"
            "        raise ValueError('count must not be zero')\n"
            "    try:\n"
            "        return total / count\n"
            "    except TypeError:\n"
            "        return None\n"
        ),
        expected_rule_ids=(),
    ),
    BenchmarkCase(
        case_id="string_formatting_zero",
        description="字符串百分号格式化零值不属于数值取模",
        code='text = "value=%s" % 0\n',
        expected_rule_ids=(),
    ),
    BenchmarkCase(
        case_id="bytes_formatting_zero",
        description="字节串百分号格式化零值不属于数值取模",
        code='text = b"value=%d" % 0\n',
        expected_rule_ids=(),
    ),
    BenchmarkCase(
        case_id="formatting_with_nested_division",
        description="排除格式化误报时仍检测插值中的实际零除",
        code='text = f"value={10 / 0}: %s" % 0\n',
        expected_rule_ids=("division-by-zero",),
    ),
)
