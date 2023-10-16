from collections.abc import Iterable
from contextlib import suppress

from lark import Lark, Transformer
from lark.lexer import Token

from mel_ast import *

file = open("grammarPascal.lark")
parser = Lark(file, start='start')

class MelASTBuilder(Transformer):
    def _call_userfunc(self, tree, new_children=None):
        # Assumes tree is already transformed
        children = new_children if new_children is not None else tree.children
        try:
            f = getattr(self, tree.data)
        except AttributeError:
            return self.__default__(tree.data, children, tree.meta)
        else:
            return f(*children)


    def __getattr__(self, item):
        if isinstance(item, str) and item.upper() == item:
            return lambda x: x

        if item in ('bin_op', ):
            def get_bin_op_node(*args):
                op = BinOp(args[1].value)
                return BinOpNode(op, args[0], args[2],
                                 **{'token': args[1], 'line': args[1].line, 'column': args[1].column})
            return get_bin_op_node

        if item in ('statement_list', ):
            def get_node(*args):
                return StmtListNode(*sum(([*n] if isinstance(n, Iterable) else [n] for n in args), []))

            return get_node
        # if item in ('var_decl', ):
        #     def get_node(*args):
        #         return VarsDeclNode(*sum(([*n] if isinstance(n, Iterable) else [n] for n in args), []))
        #
        #     return get_node
        #
        # if item in ('var_vars', ):
        #     def get_node(*args):
        #         return VarVarsNode(*sum(([*n] if isinstance(n, Iterable) else [n] for n in args), []))
        #
        #     return get_node

        if item in ('program', ):
            def get_node(*args):
                return ProgramNode(args[0], args[1])

            return get_node

        else:
            def get_node(*args):
                props = {}
                if len(args) == 1 and isinstance(args[0], Token):
                    props['token'] = args[0]
                    props['line'] = args[0].line
                    props['column'] = args[0].column
                    args = [args[0].value]
                with suppress(NameError):
                    cls = eval(''.join(x.capitalize() for x in item.split('_')) + 'Node')
                    return cls(*args, **props)
                return args
            return get_node


def parse(prog: str) -> StmtListNode:
    prog = parser.parse(str(prog))
    # print(prog.pretty('  '))
    prog = MelASTBuilder().transform(prog)
    return prog