from scipy.sparse import csr_matrix
import numpy as np


def cls_report(y_true, y_pred, digits=2, target_names=None):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    s = []
    a = []
    for i in range(len(target_names)):
        s.append(sum(y_true == i))
        a.append(sum((y_pred == i) & (y_true == i)) / sum(y_true == i))

    headers = ["Accuracy", "support"]
    rows = zip(target_names, a, s)

    # longest_last_line_heading = 'weighted avg'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, digits)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
    for row in rows:
        print(*row)
        report += row_fmt.format(*row, width=width, digits=digits)
    report += '\n'

    return report
