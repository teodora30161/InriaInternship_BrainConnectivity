import numpy as np
from scipy.linalg import expm
from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

functional_matrices = np.load(
    '/home/tstei/PycharmProjects/pythonProject/data/connectomes_hcp/sub-all_task-EMOTION_atlas_Schaefer1000_desc-lrrl_FC.npy')
structural_matrices = np.load(
    '/home/tstei/PycharmProjects/pythonProject/data/connectomes_hcp/sub-all_ses-01_atlas_Schaefer1000_SC.npy')

print(functional_matrices.shape, structural_matrices.shape)

n_training = 400
n_testing = 100
structural_matrices_training = structural_matrices[:n_training, :, :]
structural_matrices_testing = structural_matrices[n_training:n_testing + n_training, :, :]

functional_matrices_training = functional_matrices[:n_training, :, :]
functional_matrices_testing = functional_matrices[n_training:n_testing + n_training, :, :]


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


def structure_function_mapping2014(L, gamma):
    return expm(-gamma * L)


def objective(F, P_S):
    return np.linalg.norm(F - P_S, 'fro') ** 2

def mean_objective_function(gamma):
    obj_value = 0
    for i in range(n_training):
        P_S_train = structure_function_mapping2014(laplacian_matrices_training[i], gamma)
        obj_value += objective(functional_matrices_training[i], P_S_train)
    return obj_value / n_training


result = minimize_scalar(mean_objective_function, bounds=(0.1, 1.0), method='bounded')
optimal_gamma = result.x
print("Optimal gamma:", optimal_gamma)

test_obj_values = []
test_correlations = []

for i in range(n_testing):
    P_S_test = structure_function_mapping2014(laplacian_matrices_testing[i], optimal_gamma)
    ground_truth = functional_matrices_testing[i]
    test_obj_values.append(objective(ground_truth, P_S_test))
    corr, _ = pearsonr(ground_truth.flatten(), P_S_test.flatten())
    test_correlations.append(corr)

mean_test_obj_value = np.mean(test_obj_values)
mean_test_correlation = np.mean(test_correlations)

print("Mean Testing Objective Function Value with Individual Ground Truths:", mean_test_obj_value)
print("Mean Testing Correlation:", mean_test_correlation)


plt.figure(figsize=(12, 5))
gamma_values = np.linspace(0.1, 1.0, 10)
train_obj_values = [mean_objective_function(gamma) for gamma in gamma_values]
plt.subplot(1, 2, 1)
plt.plot(gamma_values, train_obj_values, label='Training Objective', marker='o')
plt.axvline(x=optimal_gamma, color='r', linestyle='--', label=f'Optimal Gamma = {optimal_gamma:.2f}')
plt.xlabel('Gamma')
plt.ylabel('Mean Objective Function Value')
plt.title('Objective Function vs Gamma (Training)')
plt.grid(True)
plt.legend()
plt.subplot(1, 2, 2)
plt.boxplot(test_correlations, vert=False, patch_artist=True, boxprops=dict(facecolor='orange', color='black'))
plt.xlabel('Pearson Correlation')
plt.title('Distribution of Correlations for Testing Subjects')
plt.grid(True)

plt.tight_layout()
plt.show()
