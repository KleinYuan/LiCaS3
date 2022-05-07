import numpy as np


def print_stats(metrics):
    for k, metric in metrics.items():
        accuracy = metric['true'] / (metric['true'] + metric['false'])
        accuracy_top2 = (metric['true'] + metric['true-neighbour']) / (metric['true'] + metric['false'])
        print("  Drive {} is\n  Accuracy: {}, Accuracy-top2: {} | {}% ({}%)".format(k, accuracy, accuracy_top2,
                                                                                    np.round(accuracy * 100., 2),
                                                                                    np.round(accuracy_top2 * 100., 2)))


def get_metrics_formatted(metrics):
    report_strings = []
    for k, metric in metrics.items():
        accuracy = metric['true'] / (metric['true'] + metric['false'])
        accuracy_top2 = (metric['true'] + metric['true-neighbour']) / (metric['true'] + metric['false'])
        report_string = "  Drive {} is\n  Accuracy: {}, Accuracy-top2: {} | {}% ({}%)".format(k, accuracy,
                                                                                              accuracy_top2,
                                                                                              np.round(accuracy * 100.,
                                                                                                       2), np.round(
                accuracy_top2 * 100., 2))
        report_strings.append(report_string)
    return report_strings
