import ast
import math
import re

from ..models import ToolOutput


SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}

SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
}

SAFE_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left**right,
}

SAFE_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}

EXPRESSION_PATTERN = re.compile(r"[0-9A-Za-z\.\+\-\*\/\(\)%×÷\^,\s]{3,}")


def normalize_expression(expression: str) -> str:
    normalized = expression.strip()
    normalized = normalized.replace("×", "*").replace("÷", "/").replace("％", "%")
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)
    normalized = normalized.replace("^", "**")
    normalized = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", normalized)
    return normalized


def extract_expression(query: str) -> str:
    candidates = [match.group(0).strip() for match in EXPRESSION_PATTERN.finditer(query)]
    candidates = [candidate for candidate in candidates if any(character.isdigit() for character in candidate)]
    if not candidates:
        return ""
    return max(candidates, key=len)


def evaluate_node(node):
    if isinstance(node, ast.Expression):
        return evaluate_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只允許數字常數。")

    if isinstance(node, ast.Name):
        if node.id in SAFE_NAMES:
            return SAFE_NAMES[node.id]
        raise ValueError(f"不支援的名稱: {node.id}")

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        if operator_type not in SAFE_BINARY_OPERATORS:
            raise ValueError("不支援的二元運算。")
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        return SAFE_BINARY_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        if operator_type not in SAFE_UNARY_OPERATORS:
            raise ValueError("不支援的單元運算。")
        value = evaluate_node(node.operand)
        return SAFE_UNARY_OPERATORS[operator_type](value)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            raise ValueError("不支援的函式。")
        arguments = [evaluate_node(argument) for argument in node.args]
        return SAFE_FUNCTIONS[node.func.id](*arguments)

    raise ValueError("算式包含不支援的語法。")


def safe_eval(expression: str):
    tree = ast.parse(expression, mode="eval")
    return evaluate_node(tree)


def build_calculator_tool_output(query: str) -> ToolOutput:
    extracted = extract_expression(query)
    if not extracted:
        return ToolOutput(
            name="calculator",
            content="無法從這句話中擷取出明確算式。若你想精準使用 calculator，請把算式寫得更明確。",
            metadata={"success": False},
        )

    normalized = normalize_expression(extracted)
    try:
        result = safe_eval(normalized)
        return ToolOutput(
            name="calculator",
            content=f"原始算式: {extracted}\n標準化算式: {normalized}\n計算結果: {result}",
            metadata={"success": True, "expression": normalized, "result": result},
        )
    except Exception as error:
        return ToolOutput(
            name="calculator",
            content=f"算式解析失敗。\n原始算式: {extracted}\n標準化算式: {normalized}\n錯誤: {error}",
            metadata={"success": False, "expression": normalized},
        )
