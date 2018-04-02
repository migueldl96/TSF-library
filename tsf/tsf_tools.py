import numpy as np


def _fixed_window_delegate(serie, n_prev):
    partial_X = []
    begin = 0
    end = n_prev

    while end < len(serie):
        n_samples = serie[begin:end]
        partial_X.append(n_samples)
        begin = begin + 1
        end = end + 1

    return np.array(partial_X)


def _dinamic_window_delegate(serie, handler, metrics, ratio):
    if handler.__name__ == "incremental_variance":
        limit = np.var(serie) * ratio
    else:
        limit = handler(serie) * ratio

    partial_X = []

    for index, output in enumerate(serie[2:]):
        index = index + 2

        # First two previous samples
        pivot = index-2
        samples = serie[pivot:index]
        if handler.__name__ == "incremental_variance":
            n = len(samples)
            previous_var = np.var(samples)
            previous_mean = np.mean(samples)
            while pivot - 1 >= 0 and previous_var < limit:
                n = n+1
                pivot = pivot - 1
                samples = serie[pivot:index]
                previous_var, previous_mean = handler(n, previous_mean, previous_var, serie[pivot - 1])
        else:
            while pivot - 1 >= 0 and handler(serie[pivot-1:index]) < limit:
                pivot = pivot - 1
                samples = serie[pivot:index]

        # Once we have the samples, gather info about them
        samples_info = []
        for metric in metrics:
            samples_info.append(_get_samples_info(samples, metric))
        partial_X.append(samples_info)

    print np.array(partial_X)
    return np.array(partial_X)


def _range_window_delegate(serie, dev, metrics):
    partial_X = []
    for index, output in enumerate(serie[2:]):
        index = index + 2

        # Allowed range from the sample before the output
        previous = serie[index - 1]
        allowed_range = np.arange(previous - dev, previous + dev)

        # Get the samples in the range
        pivot = index - 1
        while pivot - 1 >= 0 and _in_range(serie[pivot - 1], allowed_range):
            pivot = pivot - 1

        # Once we have the samples, gather info about them
        samples = serie[pivot:index]
        samples_info = []

        for metric in metrics:
            samples_info.append(_get_samples_info(samples, metric))
        partial_X.append(samples_info)

    return np.array(partial_X)


def _classchange_window_delegate(umbralized, exogs, metrics):
    partial_X = []

    # Info from every output from all exogs series
    for index, output in enumerate(umbralized):
        index = index + 1
        output_info = []
        pivot = index - 1
        previous = umbralized[pivot]

        # Window size
        while pivot > 0 and previous == umbralized[pivot - 1]:
            pivot = pivot - 1
        start = pivot
        end = index

        # Info from exog series
        for exog in exogs:
            samples = exog[start:end]
            for metric in metrics:
                output_info.append(_get_samples_info(samples, metric))

        partial_X.append(output_info)

    return np.array(partial_X)


def _in_range(value, allowed_range):
    return allowed_range.min() < value < allowed_range.max()


def _get_samples_info(samples, metric):
    # Valid metric?
    valid_metrics = ['mean', 'variance']
    if metric not in valid_metrics and not callable(metric):
        raise ValueError("Unkown '%s' metric" % metric)

    if callable(metric):
        return metric(samples)
    else:
        return {
            'mean': np.mean(samples),
            'variance': np.var(samples)
        }.get(metric)


# Incremental variance stat handler
# Source: http://datagenetics.com/blog/november22017/index.html
def incremental_variance(n_data, previous_mean, previous_var, new_value):
    # We need mean
    def incremental_mean(n, previous, new):
        mean = previous + (new - previous) / float(n)
        return mean

    new_mean = incremental_mean(n_data, previous_mean, new_value)
    previous_sn = previous_var * (n_data-1)
    new_sn = previous_sn + (new_value - previous_mean) * (new_value - new_mean)

    return new_sn/float(n_data), new_mean
