import os
import traceback
import numpy as np
import time
import matplotlib.pyplot as plt
#from cuda import cuda, nvrtc

from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare, animate
from hyperbolicTSNE import load_data, Datasets, SequentialOptimizer, initialization, HyperbolicTSNE
from hyperbolicTSNE.quality_evaluation_ import hyperbolic_nearest_neighbor_preservation

def plot_points(datas, labels, label, x_label, y_label, x_values=None):
    
    if x_values is None:
        for data in datas:
            x = range(1, len(data) + 1)  # x-axis values from 1 to n
            plt.plot(x, data, marker='o', linestyle='-')  # Plot points connected by a line
    else:
        for data in datas:
            plt.plot(x_values, data, marker='o', linestyle='-')  # Plot points connected by a line

    plt.legend(labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(label)
    plt.grid(True)
    plt.show()

def plot_points_separate(datas, labels, label, x_label, y_label, x_values=None):
    
    if x_values is None:
        for data in datas:
            x = range(1, len(data) + 1)  # x-axis values from 1 to n
            plt.plot(x, data, marker='o', linestyle='-')  # Plot points connected by a line
    else:
        for data, x_vals in zip(datas, x_values):
            plt.plot(x_vals, data, marker='o', linestyle='-')  # Plot points connected by a line

    if labels is not None:
        plt.legend(labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(label)
    plt.grid(True)
    plt.show()

def run_mult(count = 5, num_points = 1000, exact=False, use_uniform_grid=False, uniform_grid_n=10,
             logging=False, log_path="", dataset=Datasets.MNIST, perp=30, data_home="datasets"):
    precisions = []
    recalls = []
    runtimes = []

    for i in range(count):
        p, r, t = run(num_points, exact, use_uniform_grid, uniform_grid_n, None, logging, log_path, dataset, perp, data_home)
        precisions.append(p)
        recalls.append(r)
        runtimes.append(t)

    return np.mean(precisions, axis=0), np.mean(recalls, axis=0), np.mean(runtimes, axis=0)

def run(num_points = 1000, exact=False, use_uniform_grid=False, uniform_grid_n=10, seed=None,
        logging=False, log_path="", dataset=Datasets.MNIST, perp=30, data_home="datasets"):

    if (seed == None):
        seed = np.random.randint(100)

    dataX, dataLabels, D, V, _ = load_data(
        dataset, 
        data_home=data_home, 
        random_state=seed, 
        to_return="X_labels_D_V",
        hd_params={"perplexity": perp}, 
        sample=num_points, 
        knn_method="hnswlib"  # we use an approximation of high-dimensional neighbors to speed up computations
    )

    print("Loading data")

    start_time = time.time()


    end_time = time.time()

    execution_time = end_time - start_time
    print("Data loading:", execution_time, "seconds")

    exaggeration_factor = 12  # Just like regular t-SNE, we use early exaggeration with a factor of 12
    learning_rate = (dataX.shape[0] * 1) / (exaggeration_factor * 1000)  # We adjust the learning rate to the hyperbolic setting
    ex_iterations = 250  # The embedder is to execute 250 iterations of early exaggeration, ...
    main_iterations = 750  # ... followed by 750 iterations of non-exaggerated gradient descent.



    # ============= RUNNING =============

    opt_config = dict(
        learning_rate_ex=learning_rate,  # learning rate during exaggeration
        learning_rate_main=learning_rate,  # learning rate main optimization 
        exaggeration=exaggeration_factor, 
        exaggeration_its=ex_iterations, 
        gradientDescent_its=main_iterations, 
        vanilla=False,  # if vanilla is set to true, regular gradient descent without any modifications is performed; for  vanilla set to false, the optimization makes use of momentum and gains
        momentum_ex=0.5,  # Set momentum during early exaggeration to 0.5
        momentum=0.8,  # Set momentum during non-exaggerated gradient descent to 0.8
        exact=exact,  # To use the quad tree for acceleration (like Barnes-Hut in the Euclidean setting) or to evaluate the gradient exactly
        area_split=False,  # To build or not build the polar quad tree based on equal area splitting or - alternatively - on equal length splitting
        n_iter_check=10,  # Needed for early stopping criterion
        size_tol=0.999,  # Size of the embedding to be used as early stopping criterion
        uniform_grid_n = uniform_grid_n,
        use_uniform_grid = use_uniform_grid
    )

    opt_params = SequentialOptimizer.sequence_poincare(**opt_config)

    #print("Sequence defined")

    # Start: configure logging
    if logging:
        logging_dict = {
            "log_path": log_path
        }
        opt_params["logging_dict"] = logging_dict

        log_path = opt_params["logging_dict"]["log_path"]
        # Delete old log path
        if os.path.exists(log_path) and not only_animate:
            import shutil
            shutil.rmtree(log_path)
    # End: logging

    # Compute an initial embedding of the data via PCA
    X_embedded = initialization(
        n_samples=dataX.shape[0],
        n_components=2,
        X=dataX,
        random_state=seed,
        method="pca"
    )

    actual_num_samples = X_embedded.shape[0]

    # Initialize the embedder
    htsne = HyperbolicTSNE(
        init=X_embedded, 
        n_components=2, 
        metric="precomputed", 
        verbose=True, 
        opt_method=SequentialOptimizer, 
        opt_params=opt_params
    )

    start_time = time.time()

    try:
        hyperbolicEmbedding = htsne.fit_transform((D, V))
    except ValueError:
        print("Error!")
        hyperbolicEmbedding = find_last_embedding(log_path)
        traceback.print_exc()

    end_time = time.time()

    execution_time = end_time - start_time
    if exact:
        if use_uniform_grid:
            print(f"[GPU Exact, d={dataset}, n={actual_num_samples}, seed={seed}] Execution time:", execution_time, "seconds")
        else:
            print(f"[CPU Exact, d={dataset}, n={actual_num_samples}, seed={seed}] Execution time:", execution_time, "seconds")
    elif use_uniform_grid:
        print(f"[UGrid, d={dataset}, n={actual_num_samples}, grid_n={uniform_grid_n}, seed={seed}] Execution time:", execution_time, "seconds")
    else:
        print(f"[QTree, d={dataset}, n={actual_num_samples}, seed={seed}] Execution time:", execution_time, "seconds")

        
    _, precisions, recalls, _ = hyperbolic_nearest_neighbor_preservation(
        dataX,
        hyperbolicEmbedding,
        k_start=1,
        k_max=30,
        D_X=None,
        exact_nn=True,
        consider_order=False,
        strict=False,
        to_return="full"
    )

    print(f"Precision: {precisions}, recall: {recalls}")

    # Create a rendering of the embedding and save it to a file
    if not os.path.exists("results"):
        os.mkdir("results")
    fig = plot_poincare(hyperbolicEmbedding, dataLabels)
    fig.show()
    fig.savefig(f"results/{dataset.name}-inexact.png")

    return precisions, recalls, execution_time