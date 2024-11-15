# persistence.py
from ripser import ripser


class PersistenceDiagram:
    def __init__(self, diagrams):
        self.diagrams = diagrams

    @staticmethod
    def compute_diagram(point_cloud, max_dim=2, coeff=2, distance_matrix=False):
        if distance_matrix:
            data = point_cloud
        else:
            data = point_cloud.points
        result = ripser(data, maxdim=max_dim, coeff=coeff, distance_matrix=distance_matrix)
        return PersistenceDiagram(result['dgms'])

    @staticmethod
    def compute_diagram_comp4(data, max_dim=2, coeff=2, distance_matrix=False):
        result = ripser(data, maxdim=max_dim, coeff=coeff, distance_matrix=distance_matrix)
        return PersistenceDiagram(result['dgms'])