from scipy import sparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

from sklearn.cluster import KMeans
from networkx.algorithms.community import kernighan_lin_bisection

curdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def save_result(partition_df, output_filename, header):
    with open(output_filename, 'w') as f:
        f.write(header)
        partition_df.to_csv(f, header=None, index=False, sep=' ', mode='a')


def phi(graph, partition_df, k):
    # Calculate the value of the objective function
    val = 0
    for i in range(k):
        V_i = partition_df.loc[partition_df['part'] == i, 'node'].values
        E = nx.cut_size(graph, V_i)
        size_i = len(V_i)
        val += E / size_i
    return val


def get_partition(G, k, mode='rw', visualize=False):
    print('Computing Laplacian matrix...')
    A = nx.to_scipy_sparse_matrix(G, format='csc')
    if mode == 'sym':
        L = sparse.csgraph.laplacian(A, normed=True).tocsr()
        print('Using the symmetric Laplacian')
        print('Computing the eigenvectors and eigenvalues...')
        vals, vecs = sparse.linalg.eigsh(L, k, sigma=0)
        row_sums = np.sqrt(np.power(vecs, 2).sum(axis=1)).reshape(vecs.shape[0], 1)
        U = np.real(vecs / row_sums)
    elif mode == 'rw':
        degree_dict = nx.degree(G)
        degree_list = [x[1] for x in degree_dict]
        D = sparse.diags(degree_list, 0)
        L = sparse.csgraph.laplacian(A).tocsr().asfptype()
        print('Using the random-walk Laplacian')
        print('Computing the eigenvectors and eigenvalues...')
        vals, vecs = sparse.linalg.eigsh(L, k, M=D, sigma=0)
        U = np.real(vecs)
    else:
        raise Exception('Invalid mode parameter')

    print('Found eigenvalues: ', vals)

    kmeans = KMeans(n_clusters=k, init='k-means++', verbose=0)
    labels = kmeans.fit_predict(U)
    df = pd.DataFrame(list(zip(G.nodes, labels)), columns=['node', 'part'])

    if k == 2:
        # Improve partition with Kernighan-Lin algorithm
        print('Refining partition with Kernighan-Lin algorithm...')
        bisection = (
            df.loc[(df['part'] == 0)]['node'].values, df.loc[(df['part'] == 1)]['node'].values)
        bisection = kernighan_lin_bisection(G, bisection, max_iter=30)

        for node in G.nodes:
            if node in bisection[0]:
                df.loc[df['node'] == node, 'part'] = 0
            else:
                df.loc[df['node'] == node, 'part'] = 1

    os.makedirs(os.path.join(curdir, 'results'), exist_ok=True)

    print('Value of the objective function:', phi(G, df, k))

    for i in range(k):
        csize = len(df[(df['part'] == i)])
        print('Size of cluster {}: {}'.format(i, csize))

    if visualize:
        values = [df.iloc[n]['part'] for n in G.nodes]
        nx.draw(G, with_labels=True, node_color=values)
        plt.show()

    return df


if __name__ == "__main__":
    filename = sys.argv[1]
    num_parts = int(sys.argv[2])
    graph_id = filename.split('.')[0]

    print('Processing', graph_id)
    if filename == 'karate':
        G = nx.karate_club_graph()
        df = get_partition(G, num_parts, mode='rw', visualize=True)
    else:
        graph_filepath = os.path.join(curdir, 'graphs_processed', graph_id + '.txt')
        print('Reading edge list...')
        with open(graph_filepath) as f:
            params = f.readline()
        G = nx.read_edgelist(graph_filepath)
        df = get_partition(G, num_parts, mode='sym')
        os.makedirs(os.path.join(curdir, 'results'), exist_ok=True)
        save_result(df, os.path.join(curdir, 'results', os.path.basename(graph_filepath).replace('.txt', '.output')),
                    params)
