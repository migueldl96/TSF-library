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
    limit = handler(serie) * ratio

    partial_X = []

    for index, output in enumerate(serie[2:]):
        index = index + 2

        # First two previous samples
        pivot = index-2
        samples = serie[pivot:index]

        while handler(samples) < limit and pivot - 1 >= 0:
            pivot = pivot - 1
            samples = serie[pivot:index]

        # Once we have the samples, gather info about them
        samples_info = []
        for metric in metrics:
            samples_info.append(_get_samples_info(samples, metric))
        partial_X.append(samples_info)

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


def _classchange_periods_maker(endog):

    # Info from every output from all exogs series
    periods_vector = []
    for index, output in enumerate(endog):
        index = index + 1
        pivot = index - 1
        previous = endog[pivot]

        # Window size
        while pivot > 0 and previous == endog[pivot - 1]:
            pivot = pivot - 1

        start = pivot
        end = index

        periods_vector.append((start, end))

    return periods_vector


def _classchange_window_delegate(periods, exog, metrics):
    partial_X = []
    print exog
    # Info from exog series
    for period in periods:
        output_info = []
        samples = exog[period[0]:period[1]]
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