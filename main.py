import os
import mel_parser
import semantic


def main():
    prog = '''
        program CompilerTest;
        var a: integer;
        begin
            a := 5;
        end.
    '''
    prog2 = '''
        program CompilerTest;
        var a: integer;
            b: array[1..10] of char;
        begin
            temp := 5;
            for i := 1 to 10 do
                begin
                    b[i+3]:=b[q]+1;
                end
            if temp>=2112 then
                temp:=0
                else
                temp:=100
                
            while a and b<10 do
                begin
                    var b: Boolean;
                end
                
            repeat
                begin
                    read6(a);
                end
            until b div 5 = 4
        end.
    '''

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


if __name__ == "__main__":
    main()
