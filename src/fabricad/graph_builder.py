#!/usr/bin/env python
"""
Build graph representation from STEP files for GNN-based feature detection.

Each node represents a face, each edge represents face adjacency.
Node features include geometric properties useful for feature recognition.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS_Face, TopoDS_Edge, TopoDS
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.GeomLProp import GeomLProp_SLProps
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus


class FaceGraph:
    """Graph representation of a STEP file for GNN processing."""

    def __init__(self):
        self.faces: List[TopoDS_Face] = []
        self.face_features: List[np.ndarray] = []  # Node features (N x D)
        self.adjacency: List[Tuple[int, int]] = []  # Edge list
        self.edge_features: List[np.ndarray] = []  # Edge features
        self.face_labels: Optional[List[int]] = None  # Ground truth labels

    def num_nodes(self) -> int:
        return len(self.faces)

    def num_edges(self) -> int:
        return len(self.adjacency)


def extract_face_geometric_features(face: TopoDS_Face) -> Dict[str, float]:
    """Extract geometric features from a single face.

    Features extracted:
    - Surface area
    - Surface type (plane, cylinder, cone, sphere, torus, other)
    - Curvature (mean, gaussian)
    - Normal vector (x, y, z components)
    - Bounding box dimensions

    Args:
        face: OpenCASCADE face object

    Returns:
        Dictionary of geometric features
    """
    features = {}

    # Surface area
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    features['area'] = props.Mass()

    # Surface type (one-hot encoded)
    surface = BRepAdaptor_Surface(face)
    surf_type = surface.GetType()
    features['is_plane'] = 1.0 if surf_type == GeomAbs_Plane else 0.0
    features['is_cylinder'] = 1.0 if surf_type == GeomAbs_Cylinder else 0.0
    features['is_cone'] = 1.0 if surf_type == GeomAbs_Cone else 0.0
    features['is_sphere'] = 1.0 if surf_type == GeomAbs_Sphere else 0.0
    features['is_torus'] = 1.0 if surf_type == GeomAbs_Torus else 0.0
    features['is_other_surface'] = 1.0 if surf_type not in [
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus
    ] else 0.0

    # Curvature and normal at face center
    try:
        u_min, u_max, v_min, v_max = BRep_Tool.Surface_s(face).Bounds()
        u_mid = (u_min + u_max) / 2
        v_mid = (v_min + v_max) / 2

        props_at_point = GeomLProp_SLProps(
            BRep_Tool.Surface_s(face), u_mid, v_mid, 2, 1e-6
        )

        if props_at_point.IsCurvatureDefined():
            features['mean_curvature'] = abs(props_at_point.MeanCurvature())
            features['gaussian_curvature'] = abs(props_at_point.GaussianCurvature())
        else:
            features['mean_curvature'] = 0.0
            features['gaussian_curvature'] = 0.0

        if props_at_point.IsNormalDefined():
            normal = props_at_point.Normal()
            features['normal_x'] = normal.X()
            features['normal_y'] = normal.Y()
            features['normal_z'] = normal.Z()
        else:
            features['normal_x'] = 0.0
            features['normal_y'] = 0.0
            features['normal_z'] = 1.0
    except:
        features['mean_curvature'] = 0.0
        features['gaussian_curvature'] = 0.0
        features['normal_x'] = 0.0
        features['normal_y'] = 0.0
        features['normal_z'] = 1.0

    # Bounding box
    try:
        bbox = cq.Face(face).BoundingBox()
        features['bbox_x'] = bbox.xlen
        features['bbox_y'] = bbox.ylen
        features['bbox_z'] = bbox.zlen
        features['bbox_center_x'] = bbox.center.x
        features['bbox_center_y'] = bbox.center.y
        features['bbox_center_z'] = bbox.center.z
    except:
        features['bbox_x'] = 0.0
        features['bbox_y'] = 0.0
        features['bbox_z'] = 0.0
        features['bbox_center_x'] = 0.0
        features['bbox_center_y'] = 0.0
        features['bbox_center_z'] = 0.0

    return features


def compute_edge_features(face1: TopoDS_Face, face2: TopoDS_Face) -> Dict[str, float]:
    """Compute features for edge between two adjacent faces.

    Args:
        face1, face2: Adjacent faces

    Returns:
        Dictionary of edge features
    """
    features = {}

    try:
        # Find shared edge length (sum of all shared edges)
        shared_length = 0.0
        edges1 = set()
        explorer1 = TopExp_Explorer(face1, TopAbs_EDGE)
        while explorer1.More():
            edge = explorer1.Current()
            edges1.add(edge)
            explorer1.Next()

        explorer2 = TopExp_Explorer(face2, TopAbs_EDGE)
        while explorer2.More():
            edge = explorer2.Current()
            if edge in edges1:
                # Shared edge
                props = GProp_GProps()
                BRepGProp.LinearProperties_s(edge, props)
                shared_length += props.Mass()
            explorer2.Next()

        features['shared_edge_length'] = shared_length

        # Angle between face normals (dihedral angle)
        # Get normals
        surface1 = BRepAdaptor_Surface(face1)
        surface2 = BRepAdaptor_Surface(face2)

        u1_min, u1_max, v1_min, v1_max = BRep_Tool.Surface_s(face1).Bounds()
        u1_mid, v1_mid = (u1_min + u1_max) / 2, (v1_min + v1_max) / 2
        props1 = GeomLProp_SLProps(BRep_Tool.Surface_s(face1), u1_mid, v1_mid, 1, 1e-6)

        u2_min, u2_max, v2_min, v2_max = BRep_Tool.Surface_s(face2).Bounds()
        u2_mid, v2_mid = (u2_min + u2_max) / 2, (v2_min + v2_max) / 2
        props2 = GeomLProp_SLProps(BRep_Tool.Surface_s(face2), u2_mid, v2_mid, 1, 1e-6)

        if props1.IsNormalDefined() and props2.IsNormalDefined():
            n1 = props1.Normal()
            n2 = props2.Normal()
            dot_product = n1.X() * n2.X() + n1.Y() * n2.Y() + n1.Z() * n2.Z()
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            features['dihedral_angle'] = angle
            features['is_convex'] = 1.0 if angle < np.pi / 2 else 0.0
            features['is_concave'] = 1.0 if angle > np.pi / 2 else 0.0
        else:
            features['dihedral_angle'] = 0.0
            features['is_convex'] = 0.0
            features['is_concave'] = 0.0
    except:
        features['shared_edge_length'] = 0.0
        features['dihedral_angle'] = 0.0
        features['is_convex'] = 0.0
        features['is_concave'] = 0.0

    return features


def build_face_graph(shape: cq.Shape, feature_labels: Optional[Dict[int, int]] = None) -> FaceGraph:
    """Build graph from CadQuery shape where nodes are faces.

    Args:
        shape: CadQuery shape (Solid or Compound)
        feature_labels: Optional dict mapping face index to feature ID

    Returns:
        FaceGraph object with nodes, edges, and features
    """
    graph = FaceGraph()

    # Extract all faces
    faces = []
    explorer = TopExp_Explorer(shape.wrapped, TopAbs_FACE)
    while explorer.More():
        # Downcast TopoDS_Shape to TopoDS_Face
        face = TopoDS.Face_s(explorer.Current())
        faces.append(face)
        explorer.Next()

    graph.faces = faces
    print(f"  Found {len(faces)} faces")

    # Extract node features for each face
    face_features = []
    for i, face in enumerate(faces):
        features = extract_face_geometric_features(face)
        feature_vector = [
            features['area'],
            features['is_plane'],
            features['is_cylinder'],
            features['is_cone'],
            features['is_sphere'],
            features['is_torus'],
            features['is_other_surface'],
            features['mean_curvature'],
            features['gaussian_curvature'],
            features['normal_x'],
            features['normal_y'],
            features['normal_z'],
            features['bbox_x'],
            features['bbox_y'],
            features['bbox_z'],
            features['bbox_center_x'],
            features['bbox_center_y'],
            features['bbox_center_z'],
        ]
        face_features.append(feature_vector)

    graph.face_features = np.array(face_features, dtype=np.float32)
    print(f"  Node features: {graph.face_features.shape}")

    # Build adjacency list by finding shared edges
    adjacency = []
    edge_features = []
    edge_to_faces = {}  # Map edge to list of face indices

    for i, face in enumerate(faces):
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            edge_hash = edge.__hash__()
            if edge_hash not in edge_to_faces:
                edge_to_faces[edge_hash] = []
            edge_to_faces[edge_hash].append(i)
            explorer.Next()

    # Create edges between faces sharing an edge
    seen_pairs = set()
    for face_indices in edge_to_faces.values():
        if len(face_indices) >= 2:
            # Create edges between all pairs of faces sharing this edge
            for i in range(len(face_indices)):
                for j in range(i + 1, len(face_indices)):
                    face_i = face_indices[i]
                    face_j = face_indices[j]
                    pair = tuple(sorted([face_i, face_j]))

                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        adjacency.append((face_i, face_j))
                        adjacency.append((face_j, face_i))  # Undirected graph

                        # Compute edge features
                        edge_feat = compute_edge_features(faces[face_i], faces[face_j])
                        edge_vector = [
                            edge_feat['shared_edge_length'],
                            edge_feat['dihedral_angle'],
                            edge_feat['is_convex'],
                            edge_feat['is_concave'],
                        ]
                        edge_features.append(edge_vector)
                        edge_features.append(edge_vector)  # Same features for both directions

    graph.adjacency = adjacency
    graph.edge_features = np.array(edge_features, dtype=np.float32)
    print(f"  Adjacency: {len(adjacency)} edges")

    # Set labels if provided
    if feature_labels is not None:
        labels = [feature_labels.get(i, 0) for i in range(len(faces))]
        graph.face_labels = np.array(labels, dtype=np.int64)
        print(f"  Labels: {len(set(labels))} unique features")

    return graph


def graph_to_pytorch_geometric(graph: FaceGraph):
    """Convert FaceGraph to PyTorch Geometric Data object.

    Args:
        graph: FaceGraph object

    Returns:
        torch_geometric.data.Data object
    """
    import torch
    from torch_geometric.data import Data

    x = torch.tensor(graph.face_features, dtype=torch.float)
    edge_index = torch.tensor(graph.adjacency, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(graph.edge_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if graph.face_labels is not None:
        data.y = torch.tensor(graph.face_labels, dtype=torch.long)

    return data
