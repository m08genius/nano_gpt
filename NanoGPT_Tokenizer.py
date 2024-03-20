

class NanoGPT_Tokenizer:
    def __init__(self, input_file: str):
        with open(input_file, 'r', encoding='utf-8') as f:
            self._data = f.read()
        print(f"Size:{len(self._data)}")

        self._charset = list(set(self._data))
        self._charset.sort()

        self._vocab_size = len(self._charset)

        print(f"Size: {len(self._charset)}")
        print(f"Set: {''.join(self._charset)}")

        self._stoi = {ch:index for index, ch in enumerate(self._charset)}
        self._itos = {index:ch for index, ch in enumerate(self._charset)}

        self._encode = lambda s: [self._stoi[c] for c in s]
        self._decode = lambda l: [self._itos[i] for i in l]

        enc = self._encode("helloworld")
        dec = self._decode(enc)
        print(dec)


    def get_data(self):
        return self._data

    def get_vocab_size(self):
        return self._vocab_size

    def encode(self, s):
        return self._encode(s)

    def decode(self, l):
        return self._decode(l)

