
import os
import re
import pickle
from data_statistics import data_static

train_test = 'train'
PATH  = r"dataset/{}".format(train_test)#the path of the dataset
# TRAIN_OUTPUT = r"{}_processed".format(tr)#the path of the processed path
# TEST_OUTPUT = r"test_processed"
OUT_PATH = r"{}_processed".format(train_test)
TOKEN_RE = re.compile(r'[(),]|[^\s(),]+')#
QUANTIFIER_RE = re.compile(r"^([!?\\@]|\?!|lambda)([a-zA-Z0-9$%_'<>]+)\.$")
STRIP_TOKEN_RE = re.compile(r'\s(\*|part|\/)')
INFIX_OP = {
    "=", "/\\", "==>", "\\/", "o", ",", "+", "*", "EXP", "<=", "<", ">=", ">", "-",
    "DIV", "MOD", "treal_add", "treal_mul", "treal_le", "treal_eq", "/", "|-", "pow",
    "div", "rem", "==", "divides", "IN", "INSERT", "UNION", "INTER", "DIFF", "DELETE",
    "SUBSET", "PSUBSET", "HAS_SIZE", "CROSS", "<=_c", "<_c", "=_c", ">=_c", ">_c", "..",
    "$", "PCROSS"
}#中缀操作
# lambda and \ are semantically the same, but different in syntax (token). Thus
# cannot simply substitute lambda with \ in parsing lambda is changed to !lambda to solve name conflict with function
# Same for !, ?, ?!, @
FO_QUANT = {'!!', '!?', '!?!'}#量词
HOL_QUANT = {'!@', '!\\', '!lambda'}#
QUANTIFIER = FO_QUANT | HOL_QUANT
OPS = QUANTIFIER | INFIX_OP

class Tokenization(object):
    '''A helper class to read tokenization.

    Parameters
    ----------
    tokenization : str
        Tokenization string
    '''

    def __init__(self, tokenization):
        self.i = -1  # index
        self.tokens = tokenization.split()

    def next(self):
        '''Read next token in tokenization'''
        self.i += 1
        return self.tokens[self.i]



def geneate_dataset(path,out_path,partition,converter,files = None):
    digits = len(str(partition))
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if files is None:
        files = os.listdir(path)
        files.sort(key = lambda x:int(x))
    for i,fname in enumerate(files):
        output = []
        if i >= partition:
            break
        fpath = os.path.join(path,fname)
        print("process file{}/{} at {}".format(i+1,len(files),fpath))
        with open(fpath,'r') as f:
            next(f)
            conj_symbol = next(f)
            conj_token = next(f)
            assert conj_symbol[0]=='C'
            conjecture = converter(conj_symbol[2:],conj_token[2:])
            for line in f:
                if line and line[0] in '+-':
                    statement = converter(line[2:],next(f)[2:])
                    flag = 1 if line[0]=='+'else 0
                    record = flag,conjecture,statement
                    output.append(record)
            with open (
                os.path.join(out_path,'holstep_{}'.format(train_test) + format(i+1,"0{}d".format(digits))),
                'wb'
            ) as f:
                print('Saving to file {}/{}'.format(i + 1, partition))
                pickle.dump(output,f,pickle.HIGHEST_PROTOCOL)




def converter(text,tokenizaiton):
    '''
    :param text:
    :param tokenizaiton:
    :return:
    '''
    tokens = TOKEN_RE.findall(text)
    stack = [[]]
    for token in tokens:
        if token == '(':
            stack.append([])
        else:
            if token == ')':
                token = _pack_term(stack.pop())
            stack[-1].append(token)
    assert len(stack)==1
    process_text = _pack_term(stack[0])
    tokenizaiton = STRIP_TOKEN_RE.sub('',' '+tokenizaiton)
    typed_formula = _add_type(process_text,tokenizaiton)
    tree_parser = formula_to_tree(typed_formula)
    return tree_parser.graph

def _get_typed_name(term, token):
    '''Mark type info based on the corresponding single token

    Parameters
    ----------
    token : str
        Token extracted from tokenization.
    '''
    assert isinstance(term, str)
    if token[0] == 'c':
        assert term == token[1:], 'Term: {} ||| token: {}'.format(term, token)
        return term + ':c'
    elif token[0] == 'f':
        return term + ':' + token
    elif token[0] == 'b':
        return term + ':' + token
    else:
        raise ValueError('Unknown token! Term: {} ||| token: {}'.format(term, token))

def _repack(formula, token):
    '''Use token to add type info and repack the Formulas

    Parameters
    ----------
    token : Tokenization
        The tokenization object
    '''
    if isinstance(formula, tuple):
        # Ensure every tupled formula has more than one elements.
        assert len(formula) > 1, formula
        if formula[0] in QUANTIFIER:
            assert len(formula) == 3, formula
            # Corresponding token is deleted in clean due to ambiguity
            if formula[0] != '!\\':
                t_quant = token.next()
                assert _check_quant(formula[0],
                                    t_quant), 'Term: {} ||| token: {}'.format(
                                        formula[0], t_quant)
            return (formula[0], formula[1] + ':b', _repack(formula[2], token))
        else:  # Function
            result = []
            for x in formula:
                if isinstance(x, tuple):
                    result.append(_repack(x, token))
                else:
                    result.append(_get_typed_name(x, token.next()))
            return tuple(result)
    else:  # single token
        return _get_typed_name(formula, token.next())

def _check_quant(term, token):
    '''Check if quantifier is consistant in term and token'''
    if term == '!!':
        return term[1] == token
    elif term in QUANTIFIER:
        return term[1:] == token[1:]
    else:
        return 'c' + term == token



def _fix_omitted_forall(formula, token):
    '''Handle omitted forall in token'''
    #处理token模式下删掉的全局量词
    if formula[0] == '!!':
        assert len(formula) == 3

        return (formula[0], formula[1] + ':b', _fix_omitted_forall(formula[2], token))
    else:
        return _repack(formula, token)


def _add_type(processed_formula, tokenization):
    '''Add type information right after every constant & variable, separated by #.'''
    tokenization = Tokenization(tokenization)#
    if len(processed_formula) == 2:
        assert processed_formula[0] == '|-'
        return ('|-:c', _fix_omitted_forall(processed_formula[1], tokenization))
    else:
        # Case: A, B, C |- D
        # Note in tokenization, it's ==> A ==> B ==> C D
        formula = ['|-:c']
        for i, t in enumerate(processed_formula[1:]):
            if i < len(processed_formula) - 2:
                token = tokenization.next()
            assert token == 'c==>', 'Token is {}'.format(token)
            formula.append(_repack(t, tokenization))
        return tuple(formula)



def _pack_term(l):
    '''Pack a list of tokens into a term
    '''
    if len(l) == 2:
        if isinstance(l[0], str):
            # print("len(1)=2{}".format(l))
            match = QUANTIFIER_RE.match(l[0])
            if match:  # if has quantifier
                quantifier, var = match.groups()
                quantifier = '!' + quantifier
                term = (quantifier, var, l[1])
            else:  # if is unary operator, or curried function application
                term = tuple(l)
        else:  # if is curried function.
            assert isinstance(l[0], tuple)
            # If the first is not quantifier \ or @ or infix op, merge curried
            if l[0][0] not in OPS:
                term = l[0] + tuple(l[1:])
            else:
                term = tuple(l)
    elif len(l) == 3 :  # Handle infix operation
        assert l[1] in INFIX_OP
        term = (l[1], l[0], l[2])
    elif len(l) > 3 and l[-2] == '|-':
        assert l[1:-2:2] == [',' for _ in range((len(l) - 3) // 2)]
        term = tuple([l[-2]] + l[::2])
    else:
        raise ValueError('Unexpected syntax: {}'.format(l))
    return term


class Node(object):
    def __init__(self, name,des,depth):
        self.name = name
        self.des = des
        self.depth = depth

def formula_tree(text,graph,depth=0,destination = []):
    midlle_des = destination.copy()
    if isinstance(text,str):
        if text == '!lambda':
            text = '!\\'
        node = Node(text,des=midlle_des,depth=depth)
        graph.append(node)
        return
    else:
        formula_tree(text[0],graph,depth,midlle_des)
        depth = depth + 1
        for index in range(len(text[1:])):
            arg = text[1+index]
            midlle_des.append(index)
            formula_tree(arg,graph,depth,midlle_des)
            midlle_des.pop()


class formula_to_tree():
    def __init__(self,formula):
        self.formula = formula
        self.graph = []
        self.curry_formual_tree()
        self.change_name()
    # def formula_tree(self,text,parents = None,depth = 0,child_id = 0,des = []):
    #     if isinstance(text,str):
    #         if text == '!lambda':
    #             text = '!\\'
    #         node = Node(text)
    #         node.depth = depth
    #         node.child_id = child_id
    #         node.des = des
    #         if parents is not None:
    #             node.income.append(parents)
    #         self.graph.append(node)
    #         return node
    #     else:
    #
    #         node = self.formula_tree(text[0],parents,depth,child_id,des)
    #         child_id = 0
    #         depth = depth + 1
    #         for index in len(text[1:]):
    #             des.append(index)
    #         for arg in text[1:]:
    #             child_id += 1
    #             argnode = self.formula_tree(arg,node,depth,child_id)
    #             node.outcome.append(argnode)
    def curry_formual_tree(self):
        node_list = []
        formula_tree(self.formula,node_list)
        self.graph = node_list

    def change_name(self):
        dict = {}
        b_match= r'b|f[0-9]+'
        match = re.compile(b_match)
        for node in self.graph:
            type = node.name.split(':')
            if len(type)==2:
                if match.match(type[-1]):
                    dict[type[0]]=type[-1]
        for node in self.graph:
            type = node.name.split(':')
            if len(type)==2:
                if type[-1]=='c':
                    node.name = type[0]
                elif type[-1][0]=='f':
                    node.name = type[-1]
                elif type[-1]=='b':
                    if type[0] in dict.keys():
                        node.name = dict[type[0]]
                    else:
                        node.name = type[-1]
                else:
                    node.name = type[-1]
            else:
                continue









if __name__ == '__main__':

    geneate_dataset(PATH,OUT_PATH,9999,converter)
    #data = data_static(TRAIN_OUTPUT,TEST_OUTPUT)
    #with open("data_class",'wb') as f:
    #    pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    #print("max_depth:{}".format(data.max_depth))
    #print("max_nodesize:{}".format(data.max_nodesize))
    #print("vocab_size:{}".format(len(data.dict)))
    # formula = ('|-', ('!!', 'n', ('!!', 'm', ('=', ('-', ('SUC', 'm'), ('SUC', 'n')), ('-', 'm', 'n')))))
    # # formula = ('|-', ('=', ('*', ('NUMERAL', ('BIT1', '_0')), ('NUMERAL', ('BIT0', ('BIT1', '_0')))), ('NUMERAL', ('BIT0', ('BIT1', '_0')))))
    # formula = ('|-:c', ('!!', 'n:b', ('!!', 'm:b', ('=:c', ('-:c', ('SUC:c', 'm:f0'), ('SUC:c', 'n:f1')), ('-:c', 'm:f0', 'n:f1')))))
    # conter = formula_to_tree(formula)
    #
    # for node in conter.graph:
    #     print(node.name,node.des,node.depth)
