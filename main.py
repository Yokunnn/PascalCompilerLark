import os
import mel_parser
import semantic


def main():
    prog = '''
        program CompilerTest;
        var a: integer;
            i: integer;
        function Sum(a: integer, b: integer): integer;
            begin
                b := a + b;
                return b;
            end;
        function Summary(a: integer, b: integer): integer;
            begin
                b := a + b;
                return b;
            end;
        begin
            a := 5;
            
            repeat
                begin
                    a := a - 1;
                    Write(a);
                    Inc(a);
                end
            until a * 5 = 4
            
            while a >= 3 and a < 10 do
                begin
                    a := a + 3;
                end
                
            if a>=2112 then
                a:=0
                else    
                a:=100
                
            for i := 1 to 10 do
                begin
                    a := a + i;
                end
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
