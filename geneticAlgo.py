import random
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

class GAFeatureSelectorSVM:
    """
    Wrapper Feature Selection using a custom Genetic Algorithm (GA)
    and Support Vector Machine (SVM) classifier.
    """
    def __init__(
        self, 
        pop_size=20, 
        generations=15, 
        p_crossover=0.8, 
        p_mutation=0.05, 
        alpha=0.01, 
        tournament_k=3,
        random_state=42
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.alpha = alpha  # Parsimony weight (penalizes large feature subsets)
        self.tournament_k = tournament_k
        random.seed(random_state)
        np.random.seed(random_state)

    def _init_population(self, n_features):
        """Initializes a random population of binary chromosomes."""
        population = []
        for _ in range(self.pop_size):
            chrom = [random.choice([0, 1]) for _ in range(n_features)]
            # Ensure at least one feature is selected
            if sum(chrom) == 0:
                chrom[random.randint(0, n_features - 1)] = 1
            population.append(chrom)
        return population

    def _fitness(self, chrom, X, y):
        """
        Evaluates chromosome fitness using SVM 3-fold cross-validation accuracy
        minus a penalty for the proportion of features used.
        """
        selected_indices = [i for i, bit in enumerate(chrom) if bit == 1]
        if not selected_indices:
            return 0.0

        X_sub = X[:, selected_indices]
        clf = SVC(kernel='rbf', C=1.0)
        
        # Evaluate SVM accuracy via cross-validation
        scores = cross_val_score(clf, X_sub, y, cv=3, scoring='accuracy')
        accuracy = float(np.mean(scores))

        # Fitness = Accuracy - alpha * (selected_features / total_features)
        feature_ratio = len(selected_indices) / len(chrom)
        fitness_score = accuracy - (self.alpha * feature_ratio)
        return fitness_score

    def _tournament_selection(self, population, fitnesses):
        """Selects a parent using K-way tournament selection."""
        sampled_indices = random.sample(range(len(population)), self.tournament_k)
        best_idx = max(sampled_indices, key=lambda idx: fitnesses[idx])
        return population[best_idx].copy()

    def _crossover(self, parent1, parent2):
        """Performs single-point crossover between two parents."""
        if random.random() < self.p_crossover:
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, chrom):
        """Applies bit-flip mutation to a chromosome."""
        mutated = []
        for bit in chrom:
            if random.random() < self.p_mutation:
                mutated.append(1 - bit)
            else:
                mutated.append(bit)
        # Ensure chromosome does not mutate to zero features
        if sum(mutated) == 0:
            mutated[random.randint(0, len(mutated) - 1)] = 1
        return mutated

    def fit(self, X, y):
        """
        Runs the Genetic Algorithm feature selection process.
        """
        n_features = X.shape[1]
        population = self._init_population(n_features)

        best_chrom = None
        best_fitness = -float('inf')

        for gen in range(self.generations):
            fitnesses = [self._fitness(chrom, X, y) for chrom in population]

            # Track best individual across generations
            for chrom, fit_val in zip(population, fitnesses):
                if fit_val > best_fitness:
                    best_fitness = fit_val
                    best_chrom = chrom.copy()

            # Elitism: retain top performing chromosome directly
            new_pop = [best_chrom.copy()]

            # Generate next population
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            population = new_pop
            
            selected_count = sum(best_chrom)
            print(f"Gen {gen+1:02d}/{self.generations:02d} | "
                  f"Best Fitness: {best_fitness:.4f} | "
                  f"Features Selected: {selected_count}/{n_features}")

        selected_indices = [i for i, bit in enumerate(best_chrom) if bit == 1]
        return selected_indices, best_chrom, best_fitness


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=250, 
        n_features=20, 
        n_informative=5, 
        n_redundant=10, 
        random_state=42
    )

    ga_selector = GAFeatureSelectorSVM(
        pop_size=20, 
        generations=10, 
        p_crossover=0.8, 
        p_mutation=0.05, 
        alpha=0.01,
        random_state=42
    )

    selected_cols, mask, score = ga_selector.fit(X, y)

    print("\n--- Final Results ---")
    print(f"Selected Feature Indices ({len(selected_cols)}/20): {selected_cols}")
    print(f"Binary Mask: {mask}")
    print(f"Final Optimized Fitness: {score:.4f}")