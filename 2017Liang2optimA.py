import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

script_start = time.time()

print("Loading data...")
start_time = time.time()
functional_matrices = np.load(
    '/home/tstei/PycharmProjects/pythonProject/data/connectomes_hcp/sub-all_task-EMOTION_atlas_Schaefer1000_desc-lrrl_FC.npy')
structural_matrices = np.load(
    '/home/tstei/PycharmProjects/pythonProject/data/connectomes_hcp/sub-all_ses-01_atlas_Schaefer1000_SC.npy')
print(f"Data loaded in {time.time() - start_time:.2f} seconds.")
print(f"Functional shape: {functional_matrices.shape}, Structural shape: {structural_matrices.shape}")

n_training = 50
n_testing = 12
structural_matrices_training = structural_matrices[:n_training, :, :]
structural_matrices_testing = structural_matrices[n_training:n_training + n_testing, :, :]
functional_matrices_training = functional_matrices[:n_training, :, :]
functional_matrices_testing = functional_matrices[n_training:n_training + n_testing, :, :]


def laplacian_normalized(matrix):
    D = np.diag(np.sum(matrix, axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
        D_inv_sqrt = np.nan_to_num(D_inv_sqrt)
    return np.eye(matrix.shape[0]) - D_inv_sqrt @ matrix @ D_inv_sqrt


print("Computing Laplacian matrices...")
start_time = time.time()
laplacian_matrices_training = np.zeros(structural_matrices_training.shape)
for i in range(n_training):
    laplacian_matrices_training[i] = laplacian_normalized(structural_matrices_training[i])

laplacian_matrices_testing = np.zeros(structural_matrices_testing.shape)
for i in range(n_testing):
    laplacian_matrices_testing[i] = laplacian_normalized(structural_matrices_testing[i])
print(f"Laplacian matrices computed in {time.time() - start_time:.2f} seconds.")


def precompute_S_powers(S, a):
    S_powers = []
    S_power = np.eye(S.shape[0], dtype=np.float64)
    for coeff in a:
        S_powers.append(coeff * S_power)
        S_power = S_power @ S
    return np.sum(S_powers, axis=0)


def liang_model(S_powers, C):
    return S_powers + C


def objective(F, P_S):
    return np.linalg.norm(F - P_S, 'fro') ** 2


optimizer_iteration = 0


def mean_objective_training_a(a_params, structural_matrices, functional_matrices, verbose=False, print_every=50):
    global optimizer_iteration
    M = len(a_params) - 1  # Number of coefficients in 'a'
    a = np.array(a_params)

    total_loss = 0
    for S, F_true in zip(structural_matrices, functional_matrices):
        S_powers = precompute_S_powers(S, a)
        F_pred = liang_model(S_powers, np.zeros_like(S))  # No C in this case
        total_loss += objective(F_true, F_pred)

    optimizer_iteration += 1
    if verbose and optimizer_iteration % print_every == 0:
        print(
            f"Optimizer iteration {optimizer_iteration}: Mean objective = {total_loss / len(structural_matrices):.4f}")

    return total_loss


# Optimize 'a' for training
print("Fitting 'a' for the entire training set...")
start_time = time.time()
M = 4  # Number of terms in the series
initial_a = np.random.uniform(low=-0.01, high=0.01, size=M + 1)  # Initialize 'a' coefficients randomly
result_a = minimize(
    mean_objective_training_a,
    initial_a,
    args=(structural_matrices_training, functional_matrices_training, True, 50),  # Print every 50 iterations
    method='CG',
    options={
        'maxiter': 100,  # Maximum iterations
        'ftol': 1e-3,  # Adjust function tolerance
        'gtol': 1e-5  # Adjust gradient tolerance
    }
)
print(f"'a' optimization completed in {time.time() - start_time:.2f} seconds.")
print("Optimization Result for 'a':")
print(f"Final Mean Objective: {result_a.fun}")
print(f"Optimized 'a': {result_a.x}")
print(f"Optimization Status: {result_a.message}")

optimized_a = result_a.x

# Testing phase with optimized 'a'
print("Starting testing phase with optimized 'a'...")
test_obj_values = []
test_correlations = []
start_time = time.time()
for i in range(n_testing):
    S_powers_test = precompute_S_powers(structural_matrices_testing[i], optimized_a)
    F_pred = liang_model(S_powers_test, np.zeros_like(S_powers_test))  # No C
    F_true = functional_matrices_testing[i]

    test_obj_values.append(objective(F_true, F_pred))
    corr, _ = pearsonr(F_true.flatten(), F_pred.flatten())
    test_correlations.append(corr)
print(f"Testing phase completed in {time.time() - start_time:.2f} seconds.")

mean_test_obj_value = np.mean(test_obj_values)
mean_test_correlation = np.mean(test_correlations)
print("Mean Testing Objective Function Value:", mean_test_obj_value)
print("Mean Testing Correlation:", mean_test_correlation)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('Optimized Coefficients (a_m)')
plt.plot(range(len(optimized_a)), optimized_a, marker='o')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.boxplot(test_correlations, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'))
plt.xlabel('Pearson Correlation')
plt.title('Distribution of Correlations for Testing Subjects')
plt.grid(True)

plt.tight_layout()
plt.show()
print(f"Total script execution time: {time.time() - script_start:.2f} seconds.")