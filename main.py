import os
import mel_parser
import semantic
import msil


def main():

    with open("test.txt", "r") as file:
        prog = file.read()

    prog = mel_parser.parse(prog)
    print('ast:')
    print(*prog.tree, sep=os.linesep)
    print()

    print('semanthic_check:')
    try:
        scope = semantic.prepare_global_scope()
        prog.semantic_check(scope)
        print(*prog.tree, sep=os.linesep)
    except semantic.SemanticException as e:
        print('Ошибка: {}'.format(e.message))
        return
    print()

    print('msil:')
    try:
        gen = msil.CodeGenerator()
        gen.start()
        prog.msil(gen)
        gen.end()
        print(*gen.code, sep=os.linesep)
    except semantic.SemanticException as e:
        print('Ошибка: {}'.format(e.message))
        return
    print()


if __name__ == "__main__":
    main()
