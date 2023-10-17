from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Callable, Tuple, Optional, Union
from enum import Enum

from semantic import TYPE_CONVERTIBILITY, BIN_OP_TYPE_COMPATIBILITY,\
    BinOp, TypeDesc, IdentDesc, ScopeType, IdentScope, SemanticException


class AstNode(ABC):
    init_action: Callable[['AstNode'], None] = None

    def __init__(self, row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__()
        self.row = row
        self.line = line
        for k, v in props.items():
            setattr(self, k, v)
        if AstNode.init_action is not None:
            AstNode.init_action(self)
        self.node_type: Optional[TypeDesc] = None
        self.node_ident: Optional[IdentDesc] = None

    @property
    def childs(self) -> Tuple['AstNode', ...]:
        return ()

    @abstractmethod
    def __str__(self) -> str:
        pass

    def to_str(self):
        return str(self)

    def to_str_full(self):
        r = ''
        if self.node_ident:
            r = str(self.node_ident)
        elif self.node_type:
            r = str(self.node_type)
        return self.to_str() + (' : ' + r if r else '')

    @property
    def tree(self) -> [str, ...]:
        res = [self.to_str_full()]
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

    def semantic_error(self, message: str):
        raise SemanticException(message, self.row, self.col)

    def semantic_check(self, scope: IdentScope) -> None:
        pass


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

    def semantic_check(self, scope: IdentScope) -> None:
        if isinstance(self.value, bool):
            self.node_type = TypeDesc.BOOL
        elif isinstance(self.value, int):
            self.node_type = TypeDesc.INT
        elif isinstance(self.value, str):
            self.node_type = TypeDesc.CHAR
        else:
            self.semantic_error('Неизвестный тип {} для {}'.format(type(self.value), self.value))


class IdentNode(ExprNode):
    def __init__(self, name: str,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.name = str(name)

    def __str__(self) -> str:
        return str(self.name)

    def semantic_check(self, scope: IdentScope) -> None:
        ident = scope.get_ident(self.name)
        if ident is None:
            self.semantic_error('Идентификатор {} не найден'.format(self.name))
        self.node_type = ident.type
        self.node_ident = ident

# TODO: realization semantic for arrays
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


class TypeNode(IdentNode):
    """Класс для представления в AST-дереве типов данный
       (при появлении составных типов данных должен быть расширен)
    """

    def __init__(self, name: str,
                 row: Optional[int] = None, col: Optional[int] = None, **props) -> None:
        super().__init__(name, row=row, col=col, **props)
        self.type = None
        with suppress(SemanticException):
            self.type = TypeDesc.from_str(name)

    def to_str_full(self):
        return self.to_str()

    def semantic_check(self, scope: IdentScope) -> None:
        if self.type is None:
            self.semantic_error('Неизвестный тип {}'.format(self.name))


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

    def semantic_check(self, scope: IdentScope) -> None:
        self.arg1.semantic_check(scope)
        self.arg2.semantic_check(scope)

        if self.arg1.node_type.is_simple or self.arg2.node_type.is_simple:
            compatibility = BIN_OP_TYPE_COMPATIBILITY[self.op]
            args_types = (self.arg1.node_type.base_type, self.arg2.node_type.base_type)
            if args_types in compatibility:
                self.node_type = TypeDesc.from_base_type(compatibility[args_types])
                return

            if self.arg2.node_type.base_type in TYPE_CONVERTIBILITY:
                for arg2_type in TYPE_CONVERTIBILITY[self.arg2.node_type.base_type]:
                    args_types = (self.arg1.node_type.base_type, arg2_type)
                    if args_types in compatibility:
                        self.arg2 = type_convert(self.arg2, TypeDesc.from_base_type(arg2_type))
                        self.node_type = TypeDesc.from_base_type(compatibility[args_types])
                        return
            if self.arg1.node_type.base_type in TYPE_CONVERTIBILITY:
                for arg1_type in TYPE_CONVERTIBILITY[self.arg1.node_type.base_type]:
                    args_types = (arg1_type, self.arg2.node_type.base_type)
                    if args_types in compatibility:
                        self.arg1 = type_convert(self.arg1, TypeDesc.from_base_type(arg1_type))
                        self.node_type = TypeDesc.from_base_type(compatibility[args_types])
                        return

        self.semantic_error("Оператор {} не применим к типам ({}, {})".format(
            self.op, self.arg1.node_type, self.arg2.node_type
        ))


class StmtNode(ExprNode):
    pass


class VarsDeclNode(StmtNode):
    def __init__(self, var: IdentNode, vars_type: TypeNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.vars_type = vars_type
        self.var = var

    @property
    def childs(self) -> Tuple[ExprNode, ...]:
        # return self.vars_type, (*self.vars_list)
        return self.var, self.vars_type

    def __str__(self) -> str:
        return 'var'

    def semantic_check(self, scope: IdentScope) -> None:
        self.vars_type.semantic_check(scope)
        try:
            scope.add_ident(IdentDesc(self.var.name, self.vars_type.type, ScopeType.GLOBAL))
        except SemanticException as e:
            self.var.semantic_error(e.message)
        self.var.semantic_check(scope)

class VarVarsNode(StmtNode):
    def __init__(self, *exprs: VarsDeclNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.exprs = exprs

    @property
    def childs(self) -> Tuple[VarsDeclNode, ...]:
        return self.exprs

    def __str__(self) -> str:
        return 'variables'

    def semantic_check(self, scope: IdentScope) -> None:
        for expr in self.exprs:
            expr.semantic_check(scope)


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

    def semantic_check(self, scope: IdentScope) -> None:
        func = scope.get_ident(self.func.name)
        if func is None:
            self.semantic_error('Функция {} не найдена'.format(self.func.name))
        if not func.type.func:
            self.semantic_error('Идентификатор {} не является функцией'.format(func.name))
        if len(func.type.params) != len(self.params):
            self.semantic_error('Кол-во аргументов {} не совпадает (ожидалось {}, передано {})'.format(
                func.name, len(func.type.params), len(self.params)
            ))
        params = []
        error = False
        decl_params_str = fact_params_str = ''
        for i in range(len(self.params)):
            param: ExprNode = self.params[i]
            param.semantic_check(scope)
            if (len(decl_params_str) > 0):
                decl_params_str += ', '
            decl_params_str += str(func.type.params[i])
            if (len(fact_params_str) > 0):
                fact_params_str += ', '
            fact_params_str += str(param.node_type)
            try:
                params.append(type_convert(param, func.type.params[i]))
            except:
                error = True
        if error:
            self.semantic_error('Фактические типы ({1}) аргументов функции {0} не совпадают с формальными ({2})\
                                    и не приводимы'.format(
                func.name, fact_params_str, decl_params_str
            ))
        else:
            self.params = tuple(params)
            self.func.node_type = func.type
            self.func.node_ident = func
            self.node_type = func.type.return_type


class ReturnNode(StmtNode):
    """Класс для представления в AST-дереве оператора return
    """

    def __init__(self, val: ExprNode,
                 row: Optional[int] = None, col: Optional[int] = None, **props) -> None:
        super().__init__(row=row, col=col, **props)
        self.val = val

    def __str__(self) -> str:
        return 'return'

    @property
    def childs(self) -> Tuple[ExprNode]:
        return (self.val, )

    def semantic_check(self, scope: IdentScope) -> None:
        self.val.semantic_check(IdentScope(scope))
        func = scope.curr_func
        if func is None:
            self.semantic_error('Оператор return применим только к функции')
        self.val = type_convert(self.val, func.func.type.return_type, self, 'возвращаемое значение')


class ParamNode(StmtNode):
    """Класс для представления в AST-дереве объявления параметра функции
    """

    def __init__(self, name: IdentNode,  type_: TypeNode,
                 row: Optional[int] = None, col: Optional[int] = None, **props) -> None:
        super().__init__(row=row, col=col, **props)
        self.type = type_
        self.name = name

    def __str__(self) -> str:
        return str(self.type)

    @property
    def childs(self) -> Tuple[IdentNode]:
        return self.name,

    def semantic_check(self, scope: IdentScope) -> None:
        self.type.semantic_check(scope)
        self.name.node_type = self.type.type
        try:
            self.name.node_ident = scope.add_ident(IdentDesc(self.name.name, self.type.type, ScopeType.PARAM))
        except SemanticException:
            raise self.name.semantic_error('Параметр {} уже объявлен'.format(self.name.name))


class FuncNode(StmtNode):
    """Класс для представления в AST-дереве объявления функции
    """

    def __init__(self, name: IdentNode, params: Tuple[ParamNode], type_: TypeNode, body: StmtNode,
                 row: Optional[int] = None, col: Optional[int] = None, **props) -> None:
        super().__init__(row=row, col=col, **props)
        self.type = type_
        self.name = name
        self.params = params
        self.body = body

    def __str__(self) -> str:
        return 'function'

    @property
    def childs(self) -> Tuple[AstNode, ...]:
        return _GroupNode(str(self.type), self.name), _GroupNode('params', *self.params), self.body

    def semantic_check(self, scope: IdentScope) -> None:
        if scope.curr_func:
            self.semantic_error("Объявление функции ({}) внутри другой функции не поддерживается".format(self.name.name))
        parent_scope = scope
        self.type.semantic_check(scope)
        scope = IdentScope(scope)

        # временно хоть какое-то значение, чтобы при добавлении параметров находить scope функции
        scope.func = EMPTY_IDENT
        params = []
        for param in self.params:
            # при проверке параметров происходит их добавление в scope
            param.semantic_check(scope)
            params.append(param.type.type)

        type_ = TypeDesc(None, self.type.type, tuple(params))
        func_ident = IdentDesc(self.name.name, type_)
        scope.func = func_ident
        self.name.node_type = type_
        try:
            self.name.node_ident = parent_scope.curr_global.add_ident(func_ident)
        except SemanticException as e:
            self.name.semantic_error("Повторное объявление функции {}".format(self.name.name))
        self.body.semantic_check(scope)


class TypeConvertNode(ExprNode):
    """Класс для представления в AST-дереве операций конвертации типов данных
       (в языке программирования может быть как expression, так и statement)
    """

    def __init__(self, expr: ExprNode, type_: TypeDesc,
                 row: Optional[int] = None, col: Optional[int] = None, **props) -> None:
        super().__init__(row=row, col=col, **props)
        self.expr = expr
        self.type = type_
        self.node_type = type_

    def __str__(self) -> str:
        return 'convert'

    @property
    def childs(self) -> Tuple[AstNode, ...]:
        return (_GroupNode(str(self.type), self.expr), )


def type_convert(expr: ExprNode, type_: TypeDesc, except_node: Optional[AstNode] = None, comment: Optional[str] = None) -> ExprNode:
    """Метод преобразования ExprNode узла AST-дерева к другому типу
    :param expr: узел AST-дерева
    :param type_: требуемый тип
    :param except_node: узел, о которого будет исключение
    :param comment: комментарий
    :return: узел AST-дерева c операцией преобразования
    """

    if expr.node_type is None:
        except_node.semantic_error('Тип выражения не определен')
    if expr.node_type == type_:
        return expr
    if expr.node_type.is_simple and type_.is_simple and \
            expr.node_type.base_type in TYPE_CONVERTIBILITY and type_.base_type in TYPE_CONVERTIBILITY[expr.node_type.base_type]:
        return TypeConvertNode(expr, type_)
    else:
        (except_node if except_node else expr).semantic_error('Тип {0}{2} не конвертируется в {1}'.format(
            expr.node_type, type_, ' ({})'.format(comment) if comment else ''
        ))


class _GroupNode(AstNode):
    """Класс для группировки других узлов (вспомогательный, в синтаксисе нет соотвествия)
    """

    def __init__(self, name: str, *childs: AstNode,
                 row: Optional[int] = None, col: Optional[int] = None, **props) -> None:
        super().__init__(row=row, col=col, **props)
        self.name = name
        self._childs = childs

    def __str__(self) -> str:
        return self.name

    @property
    def childs(self) -> Tuple['AstNode', ...]:
        return self._childs


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

    def semantic_check(self, scope: IdentScope) -> None:
        self.var.semantic_check(scope)
        self.val.semantic_check(scope)
        self.val = type_convert(self.val, self.var.node_type, self, 'присваиваемое значение')
        self.node_type = self.var.node_type


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

    def semantic_check(self, scope: IdentScope) -> None:
        self.cond.semantic_check(scope)
        self.cond = type_convert(self.cond, TypeDesc.BOOL, None, 'условие')
        self.then_stmt.semantic_check(IdentScope(scope))
        if self.else_stmt:
            self.else_stmt.semantic_check(IdentScope(scope))


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

    def semantic_check(self, scope: IdentScope) -> None:
        scope = IdentScope(scope)
        self.init.semantic_check(scope)
        if self.cond == EMPTY_STMT:
            self.cond = LiteralNode('true')
        self.cond.semantic_check(scope)
        self.cond = type_convert(self.cond, TypeDesc.BOOL, None, 'условие')
        self.step.semantic_check(scope)
        self.body.semantic_check(IdentScope(scope))

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

    def semantic_check(self, scope: IdentScope) -> None:
        self.expr.semantic_check(scope)
        self.expr = type_convert(self.expr, TypeDesc.BOOL, None, 'условие')
        self.stmt.semantic_check(scope)

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

    def semantic_check(self, scope: IdentScope) -> None:
        self.expr.semantic_check(scope)
        self.expr = type_convert(self.expr, TypeDesc.BOOL, None, 'условие')
        self.stmt.semantic_check(scope)

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

    def semantic_check(self, scope: IdentScope) -> None:
        self.assign.semantic_check(scope)
        self.expr.semantic_check(scope)
        self.expr = type_convert(self.expr, TypeDesc.BOOL, None, 'условие')
        self.stmt.semantic_check(scope)

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

    def semantic_check(self, scope: IdentScope) -> None:
        for expr in self.exprs:
            expr.semantic_check(scope)


class FuncListNode(StmtNode):
    def __init__(self, *exprs: FuncNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.exprs = exprs

    @property
    def childs(self) -> Tuple[FuncNode, ...]:
        return self.exprs

    def __str__(self) -> str:
        return '...'

    def semantic_check(self, scope: IdentScope) -> None:
        for expr in self.exprs:
            expr.semantic_check(scope)


class BlockNode(StmtNode):
    def __init__(self, vars: VarVarsNode, func_list: Optional[FuncListNode], stmt_list: StmtListNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.vars = vars
        self.func_list = func_list
        self.stmt_list = stmt_list

    @property
    def childs(self) -> Tuple[VarVarsNode, FuncListNode, StmtListNode]:
        # return self.vars_type, (*self.vars_list)
        return self.vars, self.func_list, self.stmt_list

    def __str__(self) -> str:
        return 'block'

    def semantic_check(self, scope: IdentScope) -> None:
        self.vars.semantic_check(scope)
        self.func_list.semantic_check(scope)
        self.stmt_list.semantic_check(scope)

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

    def semantic_check(self, scope: IdentScope) -> None:
        self.block.semantic_check(scope)

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

    def semantic_check(self, scope: IdentScope) -> None:
        self.id.semantic_check(scope)
        self.expr.semantic_check(scope)


_empty = StmtListNode()
EMPTY_STMT = StmtListNode()
EMPTY_IDENT = IdentDesc('', TypeDesc.BOOL)
