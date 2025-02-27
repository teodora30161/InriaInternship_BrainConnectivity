import numpy as np
from scipy.linalg import expm
from scipy.stats import pearsonr
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

# Load data
functional_matrices = np.load(
    '/home/tstei/PycharmProjects/pythonProject/data/connectomes_hcp/sub-all_task-EMOTION_atlas_Schaefer1000_desc-lrrl_FC.npy')
structural_matrices = np.load(
    '/user/tstei/home/PycharmProjects/pythonProject/data/connectomes_hcp/sub-all_ses-01_atlas_Schaefer1000_SC.npy')

print("Functional Matrices Shape:", functional_matrices.shape)
print("Structural Matrices Shape:", structural_matrices.shape)

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

I = np.eye(structural_matrices.shape[1])
def structure_function_mapping_2018(L, a, b, alpha):
    return a * expm(-alpha * L) + b*I


def objective(F, P_S):
    return np.linalg.norm(F - P_S, 'fro') ** 2


def mean_objective_function(params):
    a, b, alpha = params
    obj_value = 0
    for i in range(n_training):
        P_S_train = structure_function_mapping_2018(laplacian_matrices_training[i], a, b, alpha)
        obj_value += objective(functional_matrices_training[i], P_S_train)
    return (obj_value / n_training)


initial_params = [1.0, 0.0, 0.5]
bounds = [(0, 10), (-1, 1), (0, 5)]

result = minimize(mean_objective_function, initial_params, method='L-BFGS-B', bounds=bounds, options={'disp': True})

optimal_a, optimal_b, optimal_alpha = result.x
print("Optimized Parameters:")
print("a:", optimal_a, ", b:", optimal_b, ", alpha:", optimal_alpha)

mean_fc_training = np.mean(functional_matrices_training, axis=0)

baseline_correlations = []
for i in range(n_testing):
    test_fc = functional_matrices_testing[i]
    corr, _ = pearsonr(test_fc.flatten(), mean_fc_training.flatten())
    baseline_correlations.append(corr)

mean_baseline_correlation = np.mean(baseline_correlations)
print("Mean Baseline Correlation (Model 0):", mean_baseline_correlation)

test_obj_values = []
test_correlations = []

for i in range(n_testing):
    P_S_test = structure_function_mapping_2018(laplacian_matrices_testing[i], optimal_a, optimal_b, optimal_alpha)
    ground_truth = functional_matrices_testing[i]

    test_obj_values.append(objective(ground_truth, P_S_test))

    corr, _ = pearsonr(ground_truth.flatten(), P_S_test.flatten())
    test_correlations.append(corr)

mean_test_obj_value = np.mean(test_obj_values)
mean_test_correlation = np.mean(test_correlations)
print("Mean Test Objective Value:", mean_test_obj_value)
print("Mean Test Correlation:", mean_test_correlation)

print("Baseline Correlation (Model 0):", mean_baseline_correlation)
print("Optimized Model Correlation:", mean_test_correlation)

if mean_test_correlation > mean_baseline_correlation:
    print("The optimized model outperforms the baseline.")
else:
    print("The optimized model does not outperform the baseline.")

train_correlations = []
for i in range(n_training):
    P_S_train = structure_function_mapping_2018(laplacian_matrices_training[i], optimal_a, optimal_b, optimal_alpha)
    corr, _ = pearsonr(functional_matrices_training[i].flatten(), P_S_train.flatten())
    train_correlations.append(corr)
print("Mean Training Correlation:", np.mean(train_correlations))

param_values = np.linspace(0.1, 5.0, 10)
alpha_effect = []

for alpha in param_values:
    obj_value = 0
    for i in range(n_training):
        P_S_train = structure_function_mapping_2018(
            laplacian_matrices_training[i], optimal_a, optimal_b, alpha
        )
        obj_value += objective(functional_matrices_training[i], P_S_train)
    alpha_effect.append(obj_value / n_training)

plt.figure(figsize=(12, 5))
plt.plot(param_values, alpha_effect, label='Objective vs Alpha', marker='o')
plt.xlabel('Alpha')
plt.ylabel('Mean Objective Function Value')
plt.title('Effect of Alpha on Objective Function')
plt.legend()
plt.grid()
plt.show()
