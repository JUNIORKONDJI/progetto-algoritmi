import time
import random
import matplotlib.pyplot as plt

# 1️⃣ Generazione efficiente del grafo con seed fisso
def generate_graph(n, m):
    random.seed(42)
    edges = set()
    while len(edges) < m:
        u, v = random.randint(0, n-1), random.randint(0, n-1)
        if u != v:
            edges.add((u, v))
    print(f"📌 PRIMI 10 ARCHI DEL GRAFO ({n} nodi, {m} archi): {list(edges)[:10]}")
    return list(edges)

# 2️⃣ Disjoint Set con tre implementazioni
class DisjointSetLinkedList:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, u):
        while u != self.parent[u]:
            u = self.parent[u]
        return u
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            for i in range(len(self.parent)):
                if self.parent[i] == root_v:
                    self.parent[i] = root_u

class DisjointSetLinkedListWeighted:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

class DisjointSetForest:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

# 3️⃣ Trova componenti connesse
def find_connected_components(graph, disjoint_set_class):
    n = max(max(edge) for edge in graph) + 1
    ds = disjoint_set_class(n)
    for u, v in graph:
        ds.union(u, v)
    components = set(ds.find(i) for i in range(n))
    return len(components)

# 4️⃣ Misura i tempi di esecuzione
def test_disjoint_set_performance(n, m):
    print(f"\n🔹 Generazione del grafo con {n} nodi e {m} archi...")
    graph = generate_graph(n, m)

    execution_times = {}
    num_runs = 4

    print("\n>>> Test DisjointSetLinkedList <<<")
    times = []
    for i in range(num_runs):
        start = time.time()
        find_connected_components(graph, DisjointSetLinkedList)
        times.append(time.time() - start)
        print(f"⏱  Esecuzione {i+1}: {times[-1]:.6f} secondi")
    execution_times["LinkedList"] = sum(times) / num_runs
    print(f"📊 Tempo medio: {execution_times['LinkedList']:.6f} secondi")

    print("\n>>> Test DisjointSetLinkedListWeighted <<<")
    times = []
    for i in range(num_runs):
        start = time.time()
        find_connected_components(graph, DisjointSetLinkedListWeighted)
        times.append(time.time() - start)
        print(f"⏱  Esecuzione {i+1}: {times[-1]:.6f} secondi")
    execution_times["LinkedListWeighted"] = sum(times) / num_runs
    print(f"📊 Tempo medio: {execution_times['LinkedListWeighted']:.6f} secondi")

    print("\n>>> Test DisjointSetForest <<<")
    times = []
    for i in range(num_runs):
        start = time.time()
        find_connected_components(graph, DisjointSetForest)
        times.append(time.time() - start)
        print(f"⏱  Esecuzione {i+1}: {times[-1]:.6f} secondi")
    execution_times["Forest"] = sum(times) / num_runs
    print(f"📊 Tempo medio: {execution_times['Forest']:.6f} secondi")

    return execution_times
def test_performance(graph_sizes, edges_per_node, case_type):
    results = {"LinkedList": [], "LinkedListWeighted": [], "Forest": []}
    for n in graph_sizes:
        m = n * edges_per_node
        graph = generate_graph(n, m, case_type)
        for label, ds_class in [("LinkedList", DisjointSetLinkedList),
                                ("LinkedListWeighted", DisjointSetLinkedListWeighted),
                                ("Forest", DisjointSetForest)]:
            start = time.perf_counter()
            find_connected_components(graph, ds_class)
            end = time.perf_counter()
            results[label].append(end - start)
    return results


# 5️⃣ Esecuzione test per varie dimensioni del grafo
graph_sizes = [1000, 2000, 5000, 10000]
edges_per_node = 5
results = {}

for size in graph_sizes:
    print(f"\n--- Esecuzione test per grafo con {size} nodi ---")
    execution_times = test_disjoint_set_performance(size, size * edges_per_node)
    results[size] = execution_times

# 6️⃣ Grafico finale
plt.figure(figsize=(8, 6))
plt.plot(graph_sizes, [results[size]["LinkedList"] for size in graph_sizes], 'o--', label="Liste Concatenate")
plt.plot(graph_sizes, [results[size]["LinkedListWeighted"] for size in graph_sizes], 'x--', label="Liste Concatenate Ponderate")
plt.plot(graph_sizes, [results[size]["Forest"] for size in graph_sizes], 's--', label="Foreste con Compressione")
plt.xlabel("Dimensione del Grafo (nodi)")
plt.ylabel("Tempo di Esecuzione (s)")
plt.title("Confronto tra implementazioni di Disjoint Set")
plt.legend()
plt.grid()
plt.show()
import matplotlib.pyplot as plt

graph_sizes = [1000, 2000, 5000, 10000]
results = {
    1000: {"LinkedList": 0.26, "LinkedListWeighted": 0.08, "Forest": 0.15},
    2000: {"LinkedList": 0.89, "LinkedListWeighted": 0.19, "Forest": 0.32},
    5000: {"LinkedList": 3.52, "LinkedListWeighted": 0.76, "Forest": 1.14},
    10000: {"LinkedList": 8.30, "LinkedListWeighted": 1.65, "Forest": 2.71},
}

plt.figure(figsize=(10, 6))
plt.plot(graph_sizes, [results[n]["LinkedList"] for n in graph_sizes], 'o--', label="Liste Concatenate", color='orange')
plt.plot(graph_sizes, [results[n]["LinkedListWeighted"] for n in graph_sizes], 's--', label="Liste Concatenate Ponderate", color='blue')
plt.plot(graph_sizes, [results[n]["Forest"] for n in graph_sizes], 'x--', label="Foreste con Compressione", color='red')

plt.xlabel("Numero di Nodi")
plt.ylabel("Tempo medio di esecuzione (s)")
plt.title("Confronto tra implementazioni di Disjoint Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("confronto_disjoint_set.png")  # 🔴 salva il file da visualizzare

import matplotlib.pyplot as plt

# Dati (sostituisci con i tuoi se diversi)
graph_sizes = [1000, 2000, 5000, 10000]
forest_times = [0.15, 0.32, 1.14, 2.71]  # Esempio

plt.figure(figsize=(8, 6))
plt.plot(graph_sizes, forest_times, 'o-', color='red', label='Foreste')
plt.xlabel("Dimensione del Grafo")
plt.ylabel("Tempo di Esecuzione (s)")
plt.title("Tempi di Esecuzione: Foreste")
plt.legend()
plt.grid()
plt.savefig("graph_foreste.png")

# Dati (sostituisci con i tuoi se diversi)
linkedlist_times = [0.26, 0.89, 3.52, 8.30]  # Esempio

plt.figure(figsize=(8, 6))
plt.plot(graph_sizes, linkedlist_times, 'o-', color='orange', label='Liste Concatenate')
plt.xlabel("Dimensione del Grafo")
plt.ylabel("Tempo di Esecuzione (s)")
plt.title("Tempi di Esecuzione: Liste Concatenate")
plt.legend()
plt.grid()
plt.savefig("graph_liste_concatenate.png")

linkedlist_weighted_times = [0.08, 0.19, 0.76, 1.65]  # Esempio

plt.figure(figsize=(8, 6))
plt.plot(graph_sizes, linkedlist_weighted_times, 'o-', color='blue', label='Liste Concatenate Ponderate')
plt.xlabel("Dimensione del Grafo")
plt.ylabel("Tempo di Esecuzione (s)")
plt.title("Tempi di Esecuzione: Liste Concatenate Ponderate")
plt.legend()
plt.grid()
plt.savefig("graph_liste_ponderate.png")
import matplotlib.pyplot as plt

graph_sizes = [1000, 2000, 5000, 10000]

# Tempi ipotetici per il caso migliore (es. grafi già uniti, poche operazioni)
linked_list = [0.01, 0.02, 0.05, 0.10]
linked_list_weighted = [0.005, 0.01, 0.02, 0.04]
forest = [0.002, 0.005, 0.01, 0.02]

plt.figure(figsize=(8, 6))
plt.plot(graph_sizes, linked_list, 'o--', label='Liste Concatenate', color='orange')
plt.plot(graph_sizes, linked_list_weighted, 's--', label='Liste Concatenate Ponderate', color='blue')
plt.plot(graph_sizes, forest, 'x--', label='Foreste con Compressione', color='green')

plt.xlabel("Numero di Nodi")
plt.ylabel("Tempo di Esecuzione (s)")
plt.title("Caso Migliore - Disjoint Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_best_case.png")
plt.show()
import matplotlib.pyplot as plt

graph_sizes = [1000, 2000, 5000, 10000]

# Tempi medi calcolati (es. grafi random)
linked_list = [0.26, 0.89, 3.52, 8.30]
linked_list_weighted = [0.08, 0.19, 0.76, 1.65]
forest = [0.15, 0.32, 1.14, 2.71]

plt.figure(figsize=(8, 6))
plt.plot(graph_sizes, linked_list, 'o--', label='Liste Concatenate', color='orange')
plt.plot(graph_sizes, linked_list_weighted, 's--', label='Liste Concatenate Ponderate', color='blue')
plt.plot(graph_sizes, forest, 'x--', label='Foreste con Compressione', color='green')

plt.xlabel("Numero di Nodi")
plt.ylabel("Tempo di Esecuzione (s)")
plt.title("Caso Medio - Disjoint Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_average_case.png")
plt.show()
import matplotlib.pyplot as plt

graph_sizes = [1000, 2000, 5000, 10000]

# Tempi ipotetici per il caso peggiore (es. struttura molto sbilanciata, grafi poco connessi)
linked_list = [0.5, 2.0, 7.5, 15.0]
linked_list_weighted = [0.3, 1.2, 4.8, 10.0]
forest = [0.2, 0.9, 3.5, 7.0]

plt.figure(figsize=(8, 6))
plt.plot(graph_sizes, linked_list, 'o--', label='Liste Concatenate', color='orange')
plt.plot(graph_sizes, linked_list_weighted, 's--', label='Liste Concatenate Ponderate', color='blue')
plt.plot(graph_sizes, forest, 'x--', label='Foreste con Compressione', color='green')

plt.xlabel("Numero di Nodi")
plt.ylabel("Tempo di Esecuzione (s)")
plt.title("Caso Peggiore - Disjoint Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_worst_case.png")
plt.show()
