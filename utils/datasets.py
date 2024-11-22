import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


def generate_random_data(k: int, len: int = 8):
    """
    Generates n random sequences from the following options

    1 1 1 1 1 1
    1 0 1 0 1 0
    0 1 0 1 0 1
    0 0 0 0 0 0

    """
    SOS_token = np.array([2])
    EOS_token = np.array([3])

    data = np.empty((k, len + 2), dtype=np.int64)

    # 1,1,1,1,1,1 -> 1,1,1,1,1
    data[::4] = np.concatenate((SOS_token, np.ones(len), EOS_token))

    # 0,0,0,0 -> 0,0,0,0
    data[1::4] = np.concatenate((SOS_token, np.zeros(len), EOS_token))

    # 1,0,1,0 -> 1,0,1,0,1
    X = np.zeros(len)
    X[::2] = 1
    data[2::4] = np.concatenate((SOS_token, X, EOS_token))

    # 0,1,0,1 -> 0,1,0,1
    X = np.zeros(len)
    X[1::2] = 1
    data[3::4] = np.concatenate((SOS_token, X, EOS_token))

    return data


def add_padding(arr, length, pad_token=4):
    """
    Adds padding to an array, resulting in the last axis growing to
    the given length

    Arguments:
        arr - input sequences
        length - required length of sequences
        pad_token - token to pad with

    Output:
        out - padded array: (arr.shape[0], length) numpy array
    """
    input_length = arr.shape[-1]
    padding_length = length - input_length

    if padding_length < 0:
        raise Exception("axis -1 of arr can't be greater than length")

    out = np.empty((*arr.shape[:-1], length))
    out[..., :input_length] = arr
    out[..., input_length:] = pad_token
    return out


def srs_tgts_from_samples(arr: npt.NDArray, pad_token):
    """
    Takes sequence samples and returns an array with src and tgt sequences
    where the src sequences have been shortend for training a transformer

    Arguments:
        arr - array with sequences
        pad_token - integer to pad src sequences with

    Output:
        out - array with srs and tgt sequences
            (# sequences, 2, sequence length) numpy array
    """

    samples = arr.shape[0]
    length = arr.shape[-1] - 2
    out = np.empty((samples, 2, length + 2), dtype=np.int64)
    out[:, 1] = arr

    out[:, 0] = pad_token
    out[:, 0, 0] = out[:, 1, 0]

    src_lengths = np.zeros((samples), dtype=np.int8)
    baap = samples // (length + 1)
    for i in range(length):
        src_lengths[i * baap : (i + 1) * baap] = i

    # rng = np.random.default_rng()
    # src_lengths = rng.integers(0, length + 1, size=(samples))

    sample_indexes = np.arange(samples)
    for i in range(1, length + 1):
        mask = src_lengths >= i
        out[sample_indexes[mask], 0, i] = out[sample_indexes[mask], 1, i]

    out[sample_indexes, 0, src_lengths + 1] = out[:, 1, -1]
    return out


class RandomSequenceDataset(Dataset):
    def __init__(self, k: int, len=8) -> None:
        super().__init__()
        self.samples = k
        self.data = generate_random_data(k, len=len)

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return self.samples


class PaddedSequenceDataset(Dataset):
    def __init__(self, n: int, seq_lengths: list[int] | int, total_length: int) -> None:
        super().__init__()
        if isinstance(seq_lengths, int):
            self.data = add_padding(generate_random_data(n, seq_lengths), total_length)

        if isinstance(seq_lengths, list):
            self.data = np.empty((n * len(seq_lengths), total_length), dtype=np.int64)
            for i, length in enumerate(seq_lengths):
                self.data[i * n : (i + 1) * n] = add_padding(
                    generate_random_data(n, length), total_length
                )

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return self.data.shape[0]


class PaddedSrcSequenceDataset(Dataset):
    def __init__(self, k: int, length: int, pad_token: int) -> None:
        super().__init__()
        self.samples = k
        self.length = length
        self.pad_token = pad_token
        self.data = self.generate_data()
        self.unique_tgts = np.unique(self.data[:, 1], axis=-2, return_counts=True)

    def __getitem__(self, index):
        return self.data[index, 0], self.data[index, 1]

    def __len__(self) -> int:
        return self.samples

    def generate_data(self):
        data = generate_random_data(self.samples, self.length)
        data = srs_tgts_from_samples(data, self.pad_token)
        return data


def load_from_txt(file_name: str) -> npt.NDArray:
    """
    Load spin configurations from file. Must be text file with spin
    configurations saved as rows. Lines starting with # are considered
    to be comments and skipped.

    Arguments:
        file_name - name of spin configurations file

    Output:
        data - numpy array containing spin configurations
    """

    with open(file_name, "r") as file:
        data = [
            list(map(int, line.strip()))
            for line in file
            if not line.strip().startswith("#") and line.strip()
        ]

    return np.array(data)


class MCMDataset(Dataset):
    def __init__(self, file_name: str, train=True) -> None:
        super().__init__()
        tmp = load_from_txt(file_name)
        samples = np.empty((tmp.shape[0], tmp.shape[1] + 2))
        samples[:, 0], samples[:, -1] = 2, 3
        samples[:, 1:-1] = tmp

        if train:
            self.data = srs_tgts_from_samples(samples[samples.shape[0] // 9 :], 4)
        else:
            self.data = srs_tgts_from_samples(samples[: samples.shape[0] // 9], 4)

        self.unique_tgts = np.unique(self.data[:, 1], axis=-2, return_counts=True)

    def __getitem__(self, index):
        return self.data[index, 0], self.data[index, 1]

    def __len__(self) -> int:
        return self.data.shape[0]


if __name__ == "__main__":
    PaddedSrcSequenceDataset(1000, 10, 4)
