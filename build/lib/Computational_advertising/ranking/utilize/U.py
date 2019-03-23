def auc_cal(labels,preds,n_bins=100):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0] * n_bins
    neg_histogram = [0] * n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i] * n_bins)
        if labels[i]==1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)

