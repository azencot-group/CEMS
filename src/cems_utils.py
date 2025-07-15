import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
from intrinsic_dimension import intrinsic_dimension


def shift_bit_length(x):
    return 1 << (x - 1).bit_length()


def get_id(id_path, X, Y):
    if not os.path.exists(id_path):
        X = X.reshape(X.shape[0], -1)
        Z = np.concatenate((X, Y), axis=1)
        Z = torch.tensor(Z, device='cuda')
        id_est = intrinsic_dimension(Z)
        id_est = int(np.ceil(id_est))
        np.save(id_path, id_est)
    else:
        id_est = int(np.load(id_path))

    return id_est




def get_probs(args, data):
    distances = pdist(data, metric=args.dist_metric)
    dist_matrix = squareform(distances)
    return dist_matrix


def get_probs_torch(args, data, device='cuda'):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32).to(device)
    elif not data.is_cuda:
        data = data.to(device)
    dist_matrix = torch.cdist(data, data, p=2, compute_mode='donot_use_mm_for_euclid_dist').cpu().numpy()
    return dist_matrix


def get_probabilities(args, probabilities_dir, X, Y):
    f_name = f'{args.dataset_name}_{args.neigh_type}.npy'
    f_path = os.path.join(probabilities_dir, f_name)
    probabilities = None
    if not os.path.exists(f_path) and (args.neigh_type == 'knn' or args.neigh_type == 'knnp'):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)

        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        xy = np.concatenate((X, Y), axis=1)
        dist_matrix = get_probs_torch(args, xy)


    if args.neigh_type == 'knn':
        if not os.path.exists(f_path):
            np.fill_diagonal(dist_matrix, -np.inf)
            probabilities = np.argsort(dist_matrix, axis=1)
            np.save(f_path, probabilities)
        else:
            probabilities = np.load(f_path)

    elif args.neigh_type == 'knnp':
        if not os.path.exists(f_path):
            for row in dist_matrix:
                # Find the smallest non-zero value
                smallest_non_zero = np.min(row[np.nonzero(row)])
                # Replace zero values with the smallest non-zero value
                row[row == 0] = smallest_non_zero
            probabilities = 1.0 / np.where(dist_matrix != 0, dist_matrix, 1e-8)
            np.fill_diagonal(probabilities, 0)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            probabilities = probabilities
            np.save(f_path, probabilities)
        else:
            probabilities = np.load(f_path)
    return probabilities
