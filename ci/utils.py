import numpy as np
from scipy.optimize import minimize
from typing import List
from dataclasses import dataclass
from sklearn.decomposition import PCA
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

@dataclass
class Result:
    weights: List[float]
    ci: float

def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """
    Perform Varimax rotation.
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = svd(np.dot(Phi.T, Lambda**3 - (gamma / p) * np.dot(Lambda, np.diag(np.sum(Lambda**2, axis=0)))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return np.dot(Phi, R)


def normalizar_dados(dados, orientacao="Min"):
    """
    Normaliza os dados usando o método Min-Max.

    Parâmetros:
        dados (list ou numpy.ndarray): Lista ou array de valores numéricos.
        orientacao (str): "Min" para normalização padrão (0 a 1, onde menor valor é 0 e maior é 1),
                          "Max" para inversão (0 a 1, onde menor valor é 1 e maior é 0).

    Retorno:
        list: Dados normalizados.
    """
    if not dados:
        raise ValueError("A lista de dados não pode estar vazia.")

    minimo = min(dados)
    maximo = max(dados)
    intervalo = maximo - minimo

    if intervalo == 0:
        return [0.5] * len(dados)  # Caso todos os valores sejam iguais

    if orientacao == "Min":
        return [(valor - minimo) / intervalo for valor in dados]
    elif orientacao == "Max":
        return [(maximo - valor) / intervalo for valor in dados]
    else:
        raise ValueError('A orientação deve ser "Min" ou "Max".')


def padronizar_dados(dados):
    """
    Padroniza os dados usando z-score.

    Parâmetros:
        dados (list ou numpy.ndarray): Lista ou array de valores numéricos.

    Retorno:
        list: Dados padronizados.
    """
    if not dados:
        raise ValueError("A lista de dados não pode estar vazia.")

    media = sum(dados) / len(dados)
    variancia = sum((valor - media) ** 2 for valor in dados) / len(dados)
    desvio_padrao = variancia ** 0.5

    if desvio_padrao == 0:
        return [0] * len(dados)  # Caso todos os valores sejam iguais

    return [(valor - media) / desvio_padrao for valor in dados]


class EqualWeights:
    """
    Classe para calcular indicadores compostos com pesos iguais.

    Atributos:
        dados (list of list ou numpy.ndarray): Matriz onde cada linha representa uma unidade de análise
                                               e cada coluna uma variável.
    """
    def __init__(self, data, aggregation_function=np.dot):
        self.data = data
        self.regs, self.n = self.data.shape
        self.aggregation_function = aggregation_function
    
    def compute_weights(self):
        return  [1 / self.n] * self.n

    def composite_indicator(self, data, weights, aggregation_function):
        return aggregation_function(data, weights) / np.sum(weights)
    
    def run(self):
        weights = self.compute_weights()
        results_ci = self.composite_indicator(self.data, weights, self.aggregation_function)
        results = []
        for idx  in results_ci:
            results.append(Result(weights=weights, ci=idx))
        
        return results


class BOD_Calculation:
    def __init__(self, data, aggregation_function=np.dot, bounds=None):
        self.data = np.array(data)
        self.regs, self.n = self.data.shape
        self.aggregation_function = aggregation_function
        
        if bounds is None:
            self.bounds = [(0, 1)] * self.n
        else:
            self.bounds = bounds
    
    # Objective function
    def objective(self, x, idx):
        return -self.aggregation_function(self.data[idx], x)
        
    # Constraints function
    def constraints(self, data):
        cons = []
        for row in data:
            cons.append({'type': 'ineq', 'fun': lambda x, row=row: 1 - self.aggregation_function(row, x)})

        cons.append({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)})  # Constraints sum = 1
        return cons

    # Optmize weights
    def optmizer(self, idx):
        x0 = np.full(self.n, 1 / self.n)

        cons = self.constraints(self.data)

        # Minimize objective function (objective function return negative for maximize)
        result = minimize(lambda x: self.objective(x, idx), x0, constraints=cons, bounds=self.bounds, method='SLSQP')
        
        if result.success:
                return result.x, -result.fun
        else:
            raise ValueError(f"Optimize failure: {result.message}")
    
    def composite_indicator(self, idx, weights):
        if idx >= self.regs or idx < 0:
            raise IndexError("Index outside data limits.")
        
        #Benchmark
        best_ci = 0
        for i in self.data:
            best_ci = max(self.aggregation_function(i, weights), best_ci)

        return self.aggregation_function(self.data[idx], weights) / best_ci
    
    def run(self):
        result = []
        for idx in range(self.regs):
            weights, _ = self.optmizer(idx)
            ci = self.composite_indicator(idx, weights)
            result.append(Result(weights=weights, ci=ci))
        
        return result


class Entropy_Calculation:
    def __init__(self, data, aggregation_function=np.dot):
        self.data = np.array(data)
        self.regs, self.n = self.data.shape
        self.aggregation_function = aggregation_function
        
    def compute_weights(self, data):
        
        """
        Compute Shannon entropy for the data.
        """
        probability = data / np.sum(data)
        entropy = -np.sum(probability * np.log(probability + np.finfo(float).eps), axis=0) / np.log(data.shape[0])

        """
        Compute weights based on entropy.
        """
        degrees_of_importance = 1 - entropy
        weights = degrees_of_importance / np.sum(degrees_of_importance)
        return weights

    def composite_indicator(self, data, weights, aggregation_function):
        return aggregation_function(data, weights) / np.sum(weights)
    
    def run(self):
        weights = self.compute_weights(self.data)
        results_ci = self.composite_indicator(self.data, weights, self.aggregation_function)
        results = []
        for idx  in results_ci:
            results.append(Result(weights=weights, ci=idx))
        
        return results


class PCA_Calculation:
    def __init__(self, data, aggregation_function=np.dot):
        self.data = np.array(data)
        self.regs, self.n = self.data.shape
        self.aggregation_function = aggregation_function
    
    def _standardize_data(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)
        
    def compute_weights(self, data):
        """
        Compute weights based on PCA.
        """
        # Fit PCA on data
        pca = PCA(n_components=self.n)
        pca.fit(self._standardize_data(data))
        
        # Extract explained variance ratio for each principal component
        variance = pca.explained_variance_ratio_
        eigenvalues = pca.explained_variance_
        cumulative_variance = np.cumsum(variance)

        # Filter PCs:
        # (i) eigenvalues > 1; 
        # (ii) contribute individually to the explanation of overall variance > 0.1; 
        valid_pcs = (np.round(eigenvalues) >= 1) & (variance >= 0.1)
        selected_components = pca.components_[valid_pcs, :]
        selected_eigenvalues = eigenvalues[valid_pcs]
        
        # (iii) contribute cumulatively to the explanation of the overall variance > 0.6. 
        last_true_index = np.where(valid_pcs)[0][-1]
        for i in range(last_true_index, len(cumulative_variance)):
            if cumulative_variance[i] >= 0.6:
                break
            else:
                valid_pcs[i] = False
        
        # Compute factor loadings
        loadings = selected_components.T * np.sqrt(selected_eigenvalues)
        
        # Apply Varimax rotation
        rotated_loadings = varimax(loadings)

        # Compute squared loadings
        squared_loadings = rotated_loadings**2
        
        # Scale squared loadings for unit sum
        scaled_squared_loadings = squared_loadings / np.sum(squared_loadings, axis=0, keepdims=True)

        variance_explained_by_factors = np.sum(squared_loadings, axis=0)    
        expl_tot = variance_explained_by_factors /  np.sum(variance_explained_by_factors)

        factor_weights = np.max(scaled_squared_loadings, axis=1)
        indices_max = np.argmax(scaled_squared_loadings, axis=1)

        weights = factor_weights * expl_tot[indices_max]
        weights = weights / np.sum(weights)

        return weights

    def composite_indicator(self, data, weights, aggregation_function):
        return aggregation_function(data, weights) / np.sum(weights)
    
    def run(self):
        weights = self.compute_weights(self.data)
        results_ci = self.composite_indicator(self.data, weights, self.aggregation_function)
        results = []
        for idx  in results_ci:
            results.append(Result(weights=weights, ci=idx))
        
        return results