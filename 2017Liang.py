import numpy as np
from scipy.linalg import expm
from scipy.stats import pearsonr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

script_start = time.time()
print("Loading data...")
functional_matrices = np.load('/home/tstei/PycharmProjects/pythonProject/data/connectomes_hcp/sub-all_task-EMOTION_atlas_Schaefer1000_desc-lrrl_FC.npy')
structural_matrices = np.load('/home/tstei/PycharmProjects/pythonProject/data/connectomes_hcp/sub-all_ses-01_atlas_Schaefer1000_SC.npy')

print(functional_matrices.shape, structural_matrices.shape)
print(f"Data loaded in {time.time() - script_start:.2f} seconds.")
print(f"Functional shape: {functional_matrices.shape}, Structural shape: {structural_matrices.shape}")

n_training = 10
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

laplacian_matrices_training = np.zeros(structural_matrices_training.shape)
for i in range(n_training):
    laplacian_matrices_training[i] = laplacian_normalized(structural_matrices_training[i])

laplacian_matrices_testing = np.zeros(structural_matrices_testing.shape)
for i in range(n_testing):
    laplacian_matrices_testing[i] = laplacian_normalized(structural_matrices_testing[i])

def objective(F, P_S):
    return np.linalg.norm(F - P_S, 'fro') ** 2

def liang_model(S_powers, C):
    return S_powers + C

def precompute_S_powers(S, a):
    S_powers = []
    S_power = np.eye(S.shape[0], dtype=np.float64)
    for coeff in a:
        S_powers.append(coeff * S_power)
        S_power = S_power @ S  # Compute S^(m+1)
    return np.sum(S_powers, axis=0)


def training_objective(C_params, S_powers, F_true):
    n = S_powers.shape[0]
    lower_tri_indices = np.tril_indices(n)
    C = np.zeros((n, n))
    C[lower_tri_indices] = C_params
    C = C + C.T - np.diag(np.diag(C))  # Make C symmetric
    F_pred = liang_model(S_powers, C)
    return objective(F_true, F_pred)


def fit_liang_model(S_powers, F_true):
    n = S_powers.shape[0]
    lower_tri_indices = np.tril_indices(n)
    num_params = len(lower_tri_indices[0])

    initial_params = np.zeros(num_params) + 0.01

    bounds = [(None, None)] * num_params

    result = minimize(training_objective, initial_params, args=(S_powers, F_true), method='L-BFGS-B', bounds=bounds)
    return result.x


M = 4
a = np.array([1 / (M + 1)] * (M + 1))

S_powers_training = [precompute_S_powers(S, a) for S in structural_matrices_training]

optimized_C_params = []
for i in range(n_training):
    print(f"Training subject {i+1}/{n_training}")
    C_params = fit_liang_model(S_powers_training[i], functional_matrices_training[i])
    optimized_C_params.append(C_params)


test_obj_values = []
test_correlations = []

for i in range(n_testing):
    # Use average C from training
    avg_C_params = np.mean(optimized_C_params, axis=0)
    lower_tri_indices = np.tril_indices(functional_matrices_testing[0].shape[0])
    C = np.zeros_like(functional_matrices_testing[0])
    C[lower_tri_indices] = avg_C_params
    C = C + C.T - np.diag(np.diag(C))  # C symmetric

    S_powers_test = precompute_S_powers(structural_matrices_testing[i], a)


    F_pred = liang_model(S_powers_test, C)
    F_true = functional_matrices_testing[i]  # Ground truth


    test_obj_values.append(objective(F_true, F_pred))
    corr, _ = pearsonr(F_true.flatten(), F_pred.flatten())
    test_correlations.append(corr)

mean_test_obj_value = np.mean(test_obj_values)
mean_test_correlation = np.mean(test_correlations)
print("Mean Testing Objective Function Value:", mean_test_obj_value)
print("Mean Testing Correlation:", mean_test_correlation)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('Fixed Coefficients (a_m)')
plt.plot(range(M + 1), a, marker='o')
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


