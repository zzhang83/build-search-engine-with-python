# Create an AST from a boolean expression. AST is a tuple
# consisting of an operator and a list of operands.
#
# Note: NOT is not supported.

# Example:
#   given a AND b
#   returns ('AND', ['a','b'])
#
# Copyright 2006, by Paul McGuir
# Modified: 02/2011 for csci1580
# Modified: 02/2018 for data2040

from typing import Union, Tuple

import pyparsing


class _BoolOperand(object):
    symbol = None

    def __init__(self, t):
        self.args = t[0][0::2]

    def __str__(self):
        sep = ' {} '.format(self.symbol)
        return '(' + sep.join(map(str, self.args)) + ')'

    def eval_expr(self) -> Union[str, Tuple[str, list]]:
        lst = []
        for arg in self.args:
            if not isinstance(arg, _BoolOperand):
                elem = arg
            else:
                elem = arg.eval_expr()
            lst.append(elem)
        return self.symbol, lst


class _BoolAnd(_BoolOperand):
    symbol = 'AND'


class _BoolOr(_BoolOperand):
    symbol = 'OR'


PRINTABLES_NO_PAREN = ''.join(c for c in pyparsing.printables if c not in '()')
BOOL_OPERAND = pyparsing.Word(PRINTABLES_NO_PAREN)
OP_LIST = [('AND', 2, pyparsing.opAssoc.LEFT, _BoolAnd),
           ('OR', 2, pyparsing.opAssoc.LEFT, _BoolOr)]
BOOL_EXPR = pyparsing.operatorPrecedence(BOOL_OPERAND, OP_LIST)


def validate_expr(expr: str) -> None:
    stripped = expr.strip()
    if stripped == '':
        raise ValueError('expr should not be an empty string or whitespace')
    if stripped.startswith('AND') or stripped.startswith('OR'):
        raise ValueError('expr starts with an operator (AND/OR)')
    if stripped.endswith('AND') or stripped.endswith('OR'):
        raise ValueError('expr ends with an operator (AND/OR)')


def parse_boolean(expr: str) -> Union[str, Tuple[str, list]]:
    validate_expr(expr)
    parsed_expr = BOOL_EXPR.parseString(expr)[0]
    if isinstance(parsed_expr, str):
        return expr
    return parsed_expr.eval_expr()
