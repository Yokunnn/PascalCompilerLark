from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, Union
from enum import Enum


class AstNode(ABC):
    def __init__(self, row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__()
        self.row = row
        self.line = line
        for k, v in props.items():
            setattr(self, k, v)

    @property
    def childs(self) -> Tuple['AstNode', ...]:
        return ()

    @abstractmethod
    def __str__(self) -> str:
        pass

    @property
    def tree(self) -> [str, ...]:
        res = [str(self)]
        childs_temp = self.childs
        for i, child in enumerate(childs_temp):
            ch0, ch = '├', '│'
            if i == len(childs_temp) - 1:
                ch0, ch = '└', ' '
            res.extend(((ch0 if j == 0 else ch) + ' ' + s for j, s in enumerate(child.tree)))
        return res

    def visit(self, func: Callable[['AstNode'], None]) -> None:
        func(self)
        map(func, self.childs)

    def __getitem__(self, index):
        return self.childs[index] if index < len(self.childs) else None


class ExprNode(AstNode):
    pass


class LiteralNode(ExprNode):
    def __init__(self, literal: str,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.literal = literal
        self.value = eval(literal)

    def __str__(self) -> str:
        return '{0} ({1})'.format(self.literal, type(self.value).__name__)


class IdentNode(ExprNode):
    def __init__(self, name: str,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.name = str(name)

    def __str__(self) -> str:
        return str(self.name)

class ArrIdentNode(ExprNode):
    def __init__(self, num: LiteralNode, ident: IdentNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.num = num
        self.ident = ident

    @property
    def childs(self) -> Tuple[LiteralNode, IdentNode]:
        return self.num, self.ident

    def __str__(self) -> str:
        return "arr_decl"

class ArrNode(ExprNode):
    def __init__(self, id: IdentNode, expr: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.id = id
        self.expr = expr

    @property
    def childs(self) -> Tuple[IdentNode, ExprNode]:
        return self.id, self.expr

    def __str__(self) -> str:
        return "arr"

class BinOp(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    DIV2 = 'div'
    MOD = 'mod'
    GE = '>='
    LE = '<='
    NEQUALS = '<>'
    EQUALS = '='
    GT = '>'
    LT = '<'
    AND = 'and'
    OR = 'or'
    XOR = 'xor'
    NOT = 'not'


class BinOpNode(ExprNode):
    def __init__(self, op: BinOp, arg1: ExprNode, arg2: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.op = op
        self.arg1 = arg1
        self.arg2 = arg2

    @property
    def childs(self) -> Tuple[ExprNode, ExprNode]:
        return self.arg1, self.arg2

    def __str__(self) -> str:
        return str(self.op.value)


class StmtNode(ExprNode):
    pass


class VarsDeclNode(StmtNode):
    def __init__(self, vars_type: StmtNode, *vars_list: Tuple[AstNode, ...],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.vars_type = vars_type
        self.vars_list = vars_list

    @property
    def childs(self) -> Tuple[ExprNode, ...]:
        # return self.vars_type, (*self.vars_list)
        return (self.vars_type,) + self.vars_list

    def __str__(self) -> str:
        return 'var'


class CallNode(StmtNode):
    def __init__(self, func: IdentNode, *params: Tuple[ExprNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.func = func
        self.params = params

    @property
    def childs(self) -> Tuple[IdentNode, ...]:
        # return self.func, (*self.params)
        return (self.func,) + self.params

    def __str__(self) -> str:
        return 'call'


class AssignNode(StmtNode):
    def __init__(self, var: IdentNode, val: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.var = var
        self.val = val

    @property
    def childs(self) -> Tuple[IdentNode, ExprNode]:
        return self.var, self.val

    def __str__(self) -> str:
        return ':='


class IfNode(StmtNode):
    def __init__(self, cond: ExprNode, then_stmt: StmtNode, else_stmt: Optional[StmtNode] = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.cond = cond
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt

    @property
    def childs(self) -> Tuple[ExprNode, StmtNode, Optional[StmtNode]]:
        return (self.cond, self.then_stmt) + ((self.else_stmt,) if self.else_stmt else tuple())

    def __str__(self) -> str:
        return 'if'


class ForNode(StmtNode):
    def __init__(self, init: Union[StmtNode, None], cond: Union[ExprNode, StmtNode, None],
                 step: Union[StmtNode, None], body: Union[StmtNode, None] = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.init = init if init else _empty
        self.cond = cond if cond else _empty
        self.step = step if step else _empty
        self.body = body if body else _empty

    @property
    def childs(self) -> Tuple[AstNode, ...]:
        return self.init, self.cond, self.step, self.body

    def __str__(self) -> str:
        return 'for'

class WhileNode(StmtNode):
    def __init__(self, expr: ExprNode, stmt: StmtNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.expr = expr
        self.stmt = stmt

    @property
    def childs(self) -> Tuple[ExprNode, StmtNode]:
        return self.expr, self.stmt

    def __str__(self) -> str:
        return 'while'

class RepeatNode(StmtNode):
    def __init__(self, stmt: StmtNode, expr: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.expr = expr
        self.stmt = stmt

    @property
    def childs(self) -> Tuple[StmtNode, ExprNode]:
        return self.stmt, self.expr

    def __str__(self) -> str:
        return 'repeat'

class PascalForNode(StmtNode):
    def __init__(self, assign: AssignNode, expr: ExprNode, stmt: StmtNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.assign = assign
        self.expr = expr
        self.stmt = stmt

    @property
    def childs(self) -> Tuple[AssignNode, ExprNode, StmtNode]:
        return self.assign, self.expr, self.stmt,

    def __str__(self) -> str:
        return 'for'

class StmtListNode(StmtNode):
    def __init__(self, *exprs: StmtNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.exprs = exprs

    @property
    def childs(self) -> Tuple[StmtNode, ...]:
        return self.exprs

    def __str__(self) -> str:
        return '...'

class VarVarsNode(VarsDeclNode):
    def __init__(self, *exprs: VarsDeclNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.exprs = exprs

    @property
    def childs(self) -> Tuple[VarsDeclNode, ...]:
        return self.exprs

    def __str__(self) -> str:
        return '...'

class BlockNode(StmtNode):
    def __init__(self, vars: VarsDeclNode, stmt_list: StmtListNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.vars = vars
        self.stmt_list = stmt_list

    @property
    def childs(self) -> Tuple[VarsDeclNode, StmtListNode]:
        # return self.vars_type, (*self.vars_list)
        return self.vars, self.stmt_list

    def __str__(self) -> str:
        return 'block'

class ProgramNode(StmtNode):
    def __init__(self, id: IdentNode, block: BlockNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.id = id
        self.block = block

    @property
    def childs(self) -> Tuple[IdentNode, BlockNode]:
        return self.id, self.block

    def __str__(self) -> str:
        return 'prog'

class RwsNode(StmtNode):
    def __init__(self, id: IdentNode, expr: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.id = id
        self.expr = expr

    @property
    def childs(self) -> Tuple[IdentNode, ExprNode]:
        return self.id, self.expr

    def __str__(self) -> str:
        return 'rws'


_empty = StmtListNode()
