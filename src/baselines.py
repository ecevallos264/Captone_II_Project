from sklearn.linear_model import LinearRegression


def train_linear_regression(x_train_flat, y_train):
    model = LinearRegression()
    model.fit(x_train_flat, y_train)
    return model