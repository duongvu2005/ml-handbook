from collections import Counter
from typing import List, Tuple, Set
import string
from transformers import AutoTokenizer


def count_pair_frequencies(X: List[List[str]]) -> Counter:
	"""
	Given a dataset X, returns a counter that count how often each adjacent
	pair of symbols appear in the dataset.
	"""
	pair_freq = Counter()
	for x in X:
		for idx in range(len(x) - 1):
			pair_freq[(x[idx], x[idx+1])] += 1

	return pair_freq


def replace_pair(X: List[List[str]], a: str, b: str, new_tok: str) -> List[List[str]]:
	"""
	Given a dataset X, a pair of token (a, b), and the new token new_tok,
	replace every occurrence of the pair (a, b) with new_tok in the dataset.
	"""
	new_X = []
	for x in X:
		new_seq = []
		idx = 0
		while idx < len(x):
			if idx < len(x) - 1 and x[idx] == a and x[idx+1] == b:
				new_seq.append(new_tok)
				idx += 2
			else:
				new_seq.append(x[idx])
				idx += 1
		new_X.append(new_seq)

	return new_X


def learn_bpe(
	X: List[List[str]],
	target_vocab_size: int,
	base_vocab: Set[str] = None,
	verbose=False
) -> Tuple[Set[str], List[Tuple[str, str, str]]]:
	"""
	Dictionary learning for Byte Pair Encoding (BPE).
	
	Args:
		X: dataset, list of lists of symbols (each x in X is a sequence of chars)
		target_vocab_size: desired vocabulary size
		base_vocab: optional initial vocabulary
		verbose: whether to print the merges

	Returns:
		Sigma: final vocabulary
		M: list of merges (new_token, a, b)
	"""
	# initialize base vocab
	if base_vocab is not None:
		Sigma = base_vocab
	else:
		Sigma = set(char for x in X for char in x)
	
	M = []

	while len(Sigma) < target_vocab_size:
		# get the most common pair
		pair_freq = count_pair_frequencies(X)

		a, b = pair_freq.most_common(1)[0][0]
		# define new token and merge
		new_tok = a + b
		X = replace_pair(X, a, b, new_tok)
		# update
		Sigma.add(new_tok)
		M.append((new_tok, a, b))

		if verbose:
			print(f"Merge {len(M)}: ({a}, {b}) -> {new_tok}")

	return Sigma, M


def generate_basic_vocab(include_special: bool=True) -> Set:
	base_vocab = list(
		string.ascii_letters +
		string.digits +
		string.punctuation +
		' '
	)
	if include_special:
		base_vocab += ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

	return set(base_vocab)


def tokenize(x, merge_history):
	X = [x]
	for tok, a, b in merge_history:
		X = replace_pair(X, a, b, tok)

	return X[0]


# tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
