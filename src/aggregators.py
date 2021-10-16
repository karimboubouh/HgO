import numpy as np

from src.utils import flatten_grads, unflatten_grad


def aggregate(model, grads, block, gar):
    if model in ['NN', 'DNN', "CNN"]:
        dims = [len(b) for b in block]
        unf = aggregate_vector(flatten_grads(grads), gar)

        return unflatten_grad(unf, dims)
    else:
        return aggregate_vector(grads, gar)


def aggregate_dnn(grads, gar="average"):
    if gar == "average":
        return [[average(gg) for gg in zip(*g)] for g in zip(*grads)]
    elif gar == "median":
        return [[median(gg) for gg in zip(*g)] for g in zip(*grads)]
    else:
        raise NotImplementedError()


def aggregate_cnn(grads, gar="average"):
    raise NotImplementedError("aggregate_cnn not implemented yet!")


def aggregate_vector(grads, gar):
    if gar == "average":
        return average(grads)
    elif gar == "median":
        return median(grads)
    elif gar == "aksel":
        return aksel(grads)
    elif gar == "krum":
        return krum(grads)
    else:
        raise NotImplementedError()


def average(gradients):
    """ Aggregate the gradients using the average aggregation rule."""
    # Assertion

    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    if len(gradients) > 1:
        ar = np.nanmean(gradients, axis=0)
        # nans = np.isnan(ar).sum()
        # if nans > 0:
        #     print(f"AR average got {nans} nan values, skipping update...")
        #     return None
        return ar
    else:
        return gradients[0]


def median(gradients):
    """ Aggregate the gradients using the median aggregation rule."""

    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    if len(gradients) > 1:
        ar = np.nanmedian(gradients, axis=0)
        # nans = np.isnan(ar).sum()
        # if nans > 0:
        #     print(f"AR median got {nans} nan values, skipping update...")
        #     return None
        return ar
    else:
        return gradients[0]


def aksel(gradients):
    """ Aggregate the gradients using the AKSEL aggregation rule."""
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    columns = list(zip(*gradients))
    aksel_grad = []
    for c in columns:
        c = np.array(c)
        c = c[~np.isnan(c)]
        med = np.nanmedian(c, axis=0)
        matrix = c - med
        normsq = [np.linalg.norm(grad) ** 2 for grad in matrix]
        med_norm = np.nanmedian(normsq)
        correct = [c[i] for i, norm in enumerate(normsq) if norm <= med_norm]
        aksel_grad.append(np.mean(correct, axis=0).item())

    # med = np.nanmedian(gradients, axis=0)
    # matrix = gradients - med
    # normsq = [np.linalg.norm(grad) ** 2 for grad in matrix]
    # med_norm = np.nanmedian(normsq)
    # correct = [gradients[i] for i, norm in enumerate(normsq) if norm <= med_norm]
    # return np.mean(correct, axis=0)

    return np.array(aksel_grad).reshape(-1, 1)


def krum(gradients, f=0):
    """ Aggregate the gradients using the Krum aggregation rule."""
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    gradients = np.array(gradients)
    # Distance computations
    columns = list(zip(*gradients))
    krum_grad = []
    for c in columns:
        c = np.array(c)
        c = c[~np.isnan(c)]
        nbworkers = len(c)
        scores = []
        for i in range(nbworkers - 1):
            # sqr_dst = []
            gi = c[i].reshape(-1, 1)
            sqr_dst = np.linalg.norm(gi - c[:-1], axis=0) ** 2
            # for j in range(nbworkers - 1):
            #     gj = c[j].reshape(-1, 1)
            #     dst = np.linalg.norm(gi - gj) ** 2
            #     # dst = np.sqrt(np.nansum(np.square(gi - gj))) ** 2
            #     sqr_dst.append(dst)
            if (nbworkers - f - 2) >= 0:
                indices = list(np.argsort(sqr_dst)[:nbworkers - f - 2])
            else:
                # very few coordinates
                indices = list(np.argsort(sqr_dst))
            sqr_dst = np.array(sqr_dst)
            scores.append(np.sum(sqr_dst[indices]))
        if scores:
            correct = np.argmin(scores)
        else:
            correct = 0
        krum_grad.append(c[correct].item())

    # scores = []
    # for i in range(nbworkers - 1):
    #     sqr_dst = []
    #     gi = gradients[i].reshape(-1, 1)
    #     for j in range(nbworkers - 1):
    #         gj = gradients[j].reshape(-1, 1)
    #         # dst = np.linalg.norm(gi - gj) ** 2
    #         dst = np.sqrt(np.nansum(np.square(gi - gj))) ** 2
    #         sqr_dst.append(dst)
    #     indices = list(np.argsort(sqr_dst)[:nbworkers - f - 2])
    #     sqr_dst = np.array(sqr_dst)
    #     scores.append(np.sum(sqr_dst[indices]))
    # correct = np.argmin(scores)
    # return np.where(np.isnan(gradients[correct]), 0, gradients[correct])

    return np.array(krum_grad).reshape(-1, 1)


if __name__ == '__main__':
    grds = [[1, 2, 3], [4, 5, 6], [1, 2, 3], [1, 2, 3], [4, 5, 6], [1, 2, 3], [1, 2, 3]]
    print(krum(grds))
