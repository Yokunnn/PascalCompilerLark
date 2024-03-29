from typing import List, Union

from semantic import BaseType


class CodeLabel:
    def __init__(self):
        self.index = None

    def __str__(self):
        return 'IL_' + str(self.index)


class CodeLine:
    def __init__(self, code: str, *params: Union[str, CodeLabel], label: CodeLabel = None):
        self.code = code
        self.label = label
        self.params = params

    def __str__(self):
        line = ''
        if self.label:
            line += str(self.label) + ': '
        line += self.code
        for p in self.params:
            line += ' ' + str(p)
        return line


class CodeGenerator:
    def __init__(self):
        self.code_lines: List[CodeLine] = []

    def add(self, code: str, *params: Union[str, CodeLabel], label: CodeLabel = None):
        self.code_lines.append(CodeLine(code, *params, label=label))

    @property
    def code(self) -> [str, ...]:
        index = 0
        for cl in self.code_lines:
            line = cl.code
            if cl.label:
                cl.label.index = index
                index += 1
        code: List[str] = []
        for cl in self.code_lines:
            code.append(str(cl))
        return code

    def start(self) -> None:
        self.add('.assembly program')
        self.add('{')
        self.add('}')
        self.add('.class public Program')
        self.add('{')

    def end(self) -> None:
        self.add('}')


MSIL_TYPE_NAMES = {
    BaseType.INT: 'int32',
    BaseType.BOOL: 'bool',
    BaseType.CHAR: 'char'
}

RUNTIME_CLASS_NAME = 'CompilerDemo.Runtime'
PROGRAM_CLASS_NAME = 'Program'
