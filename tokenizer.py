import unicodedata
import regex as re

SPLIT_PATTERN = r'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+'

def get_stats(ids,counts=None):
    counts = {} if counts is None else counts
    for a,b in zip(ids,ids[1:]):
        counts[(a,b)] = counts.get((a,b),0) + 1
    return counts

def merge(ids,pair,idx):
    out = []
    i = 0
    while i < len(ids):
        if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def replace_control_characters(s):
    out = []
    for ch in s:
        if unicodedata.category(ch)[0] == 'C':
            out.append(ch)
        else:
            out.append(f'\\u{ord(ch):04x}')
    return ''.join(out)

def render_token(t):
    return replace_control_characters(t.decode('utf-8',errors='replace'))

class Tokenizer:
    def __init__(self,pattern=None):
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.pattern = SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.vocab = self._build_vocab()

    def train(self,texts,vocab_size,verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text = texts if isinstance(texts,str) else ' '.join(texts)
        chunks = re.findall(self.compiled_pattern,text)
        ids = [list(ch.encode('utf-8',errors='ignore')) for ch in chunks]
        for i in range(num_merges):
            stats = {}
            for chunk in ids: get_stats(chunk,stats)
            pair = max(stats,key=stats.get)
            idx = 256 + i
            ids = [merge(chunk,pair,idx) for chunk in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose: print(f'merge {i+1}/{num_merges}: {pair} -> {idx} had {stats[pair]} occurrences')
        return self

    def register_special_tokens(self,special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in special_tokens.items()}
        self.vocab = self._build_vocab()

    def _encode_chunk(self,bs):
        ids = list(bs)
        while True:
            stats = get_stats(ids)
            pair = min(stats,key=lambda p:self.merges.get(p,float('inf')))
            if pair not in self.merges: break
            ids = merge(ids,pair,self.merges[pair])
        return ids

    def encode(self,text,allowed_special='none_raise'):
        if allowed_special == 'all':
            special = self.special_tokens
        elif allowed_special in ('none','none_raise'):
            special = {}
            if allowed_special == 'none_raise': assert all(tok not in text for tok in self.special_tokens)
        elif isinstance(allowed_special,set):
            special = {k:v for k,v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f'allowed_special={allowed_special} not understood')
        if not special:
            chunks = re.findall(self.compiled_pattern,text)
            out = []
            for ch in chunks: out.extend(self._encode_chunk(ch.encode('utf-8',errors='ignore')))
            return out
        pat = '(' + '|'.join(re.escape(k) for k in special) + ')'
        parts = re.split(pat,text)
        out = []
        for p in parts:
            if p in special: out.append(special[p])
            else: out.extend(self._encode_chunk(p.encode('utf-8',errors='ignore')))
        return out

    def decode(self,ids):
        out = []
        for i in ids:
            if i in self.vocab: out.append(self.vocab[i])
            elif i in self.inverse_special_tokens: out.append(self.inverse_special_tokens[i].encode('utf-8'))
            else: raise ValueError(f'invalid token id: {i}')
        return b''.join(out).decode('utf-8',errors='replace')

    def _build_vocab(self):
        vocab = {i:bytes([i]) for i in range(256)}
        for (a,b),idx in self.merges.items(): vocab[idx] = vocab[a] + vocab[b]
        for tok,idx in self.special_tokens.items(): vocab[idx] = tok.encode('utf-8')
        return vocab

    def save(self,file_prefix):
        model_file = file_prefix + '.model'
        with open(model_file,'w',encoding='utf-8') as f:
            f.write('minbpe v1\n')
            f.write(self.pattern + '\n')
            f.write(f'{len(self.special_tokens)}\n')
            for tok,idx in self.special_tokens.items(): f.write(f'{tok} {idx}\n')
            for (a,b),idx in self.merges.items(): f.write(f'{a} {b}\n')
        vocab_file = file_prefix + '.vocab'
        inv = {idx:pair for pair,idx in self.merges.items()}
        with open(vocab_file,'w',encoding='utf-8') as f:
            for idx,t in self.vocab.items():
                s = render_token(t)
                if idx in inv:
                    i0,i1 = inv[idx]
                    s0 = render_token(self.vocab[i0])
                    s1 = render_token(self.vocab[i1])
                    f.write(f'[{s0}][{s1}] -> [{s}] {idx}\n')
                else: f.write(f'[{s}] {idx}\n')

    def load(self,model_file):
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file,'r',encoding='utf-8') as f:
            assert f.readline().strip() == 'minbpe v1'
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                tok,sid = f.readline().strip().split()
                special_tokens[tok] = int(sid)
            for line in f:
                a,b = map(int,line.split()); merges[(a,b)] = idx; idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in special_tokens.items()}
        self.vocab = self._build_vocab()
