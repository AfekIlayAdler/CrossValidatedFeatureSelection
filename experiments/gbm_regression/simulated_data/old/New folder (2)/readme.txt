def get_x_y():
    X = DataFrame()
    nrows = 10 ** 4
    y = Series(zeros(nrows))
    for i in range(5, 10):
        X[str(i)] = random.randint(0, 2 ** i, nrows)
        X[str(i)] = X[str(i)].astype('category')
        # y += X[str(i)].isin(list(range(2 ** i - 1))) * 1

    X['numeric'] = random.random(nrows)*1
    y += random.random(nrows)*1
    # y += Series(y)
    # X.columns = [str(i) for i in X.columns]
    # y = Series(random.random(nrows))
    return X, y