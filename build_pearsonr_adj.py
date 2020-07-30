import numpy as np
from scipy.stats import pearsonr
import json
import os


def SIPM1(time_series_matrix, num_of_vertices, epsilon):
    adj_matrix = np.zeros(shape=(int(num_of_vertices), int(num_of_vertices)))
    # construct new adj_matrix with pearson in CIKM paper
    for i in range(time_series_matrix.shape[1]):
        for j in range(time_series_matrix.shape[1]):
            if pearsonr(time_series_matrix[:, i], time_series_matrix[:, j])[0] > epsilon:
                adj_matrix[i, j] = 1
    return adj_matrix


if __name__ == "__main__":
    dataset_list = [('PEMS03','individual_GLU_mask_emb.json'),
                    ('PEMS04','individual_GLU.json'),
                    ('PEMS07','individual_GLU_mask_emb.json'),
                    ('PEMS08','individual_GLU_mask_emb.json')]
    for dataset, filename in dataset_list:
        config_filename = os.path.join("./Config", dataset, filename)
        with open(config_filename, 'r') as f:
            config = json.loads(f.read())
        print(json.dumps(config, sort_keys=True, indent=4))
        epsilon = 0.7
        num_of_vertices = config['num_of_vertices']
        adj_filename = config['adj_filename']
        id_filename = config['id_filename']
        time_series_filename = config['graph_signal_matrix_filename']
        time_series_matrix = np.load(time_series_filename)['data'][:, :, 0]
        adj = SIPM1(time_series_matrix,num_of_vertices,epsilon)
        np.savez(os.path.join('./data',dataset,dataset+'_pearsonr.npz'), adj)
