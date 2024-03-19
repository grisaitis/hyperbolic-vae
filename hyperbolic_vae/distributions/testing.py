import geoopt


if __name__ == "__main__":
    ball = geoopt.PoincareBall(c=1.0)
    mean = ball.origin(2)
    print(mean)
    std = 1.0
    # wrapped_normal_dist = geoopt.distributions.WrappedNormal(ball, mean, std)
    random_means = ball.wrapped_normal(3, 2, mean=mean, std=std)
    print(random_means)
    random_points = ball.wrapped_normal(*random_means.shape, mean=random_means, std=0.01)
    print(random_points)
