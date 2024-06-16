import numpy as np
import re
import pandas as pd
import sklearn.linear_model


vocab = "ARNDCQEGHILKMFPSTWYVXU"


def one_hot_pad_seqs(s, length, vocab=vocab):
    aa_dict = {k: v for v, k in enumerate(vocab)}
    embedded = np.zeros([length, len(vocab)])
    for i, l in enumerate(s):
        if i >= length:
            break
        idx = aa_dict[l]
        embedded[i, idx] = 1
    embedded = embedded.flatten()
    return embedded


def get_seq(df, length=100):
    seq = df.sequence.values.tolist()
    X = [one_hot_pad_seqs(s, length) for s in seq]
    return np.array(X)


def load_train_data(path, val_split=False):
    df = pd.read_csv(path)
    df.sequence = df.sequence.apply(
        lambda s: re.sub(r"[^A-Z]", "", s.upper())
    )  # remove special characters

    if val_split:
        train = df[(df.set == "train")]
        val = df[(df.set == "val")]
        return train, val
    else:
        return df


def load_test_data(path):
    df = pd.read_csv(path)
    df.sequence = df.sequence.apply(
        lambda s: re.sub(r"[^A-Z]", "", s.upper())
    )  # remove special characters
    return df


def main():

    train = load_train_data("train.csv", val_split=False)
    test = load_test_data("test.csv")

    train_X, test_X = get_seq(train), get_seq(test)
    train_y = np.array(train.target.values.tolist())
    test_id = test.id.values.tolist()

    model = sklearn.linear_model.LinearRegression()
    model.fit(train_X.reshape(train_X.shape[0], -1), train_y)
    test_y = model.predict(test_X.reshape(test_X.shape[0], -1))

    with open("prediction.csv", "w") as f:
        f.write("id,target\n")
        for id, y in zip(test_id, test_y):
            f.write(f"{id},{y}\n")


if __name__ == "__main__":
    main()
