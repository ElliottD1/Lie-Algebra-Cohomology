
from collections import deque
from typing import List, Dict, Tuple, Any
import numpy as np

def get_cartan(group: str, n: int) -> List[List[int]]:
    G = group.upper()
    if G == "A":
        C = [[0]*n for _ in range(n)]
        for i in range(n):
            C[i][i] = 2
        for i in range(n-1):
            C[i][i+1] = C[i+1][i] = -1
        return C

    if G == "B":
        if n < 2:
            raise ValueError("B_n requires n>=2")
        C = [[0]*n for _ in range(n)]
        for i in range(n):
            C[i][i] = 2
        for i in range(n-1):
            C[i][i+1] = C[i+1][i] = -1
        C[n-2][n-1] = -1
        C[n-1][n-2] = -2
        return C

    if G == "C":
        if n < 2:
            raise ValueError("C_n requires n>=2")
        C = [[0]*n for _ in range(n)]
        for i in range(n):
            C[i][i] = 2
        for i in range(n-1):
            C[i][i+1] = C[i+1][i] = -1
        C[n-2][n-1] = -2
        C[n-1][n-2] = -1
        return C

    if G == "D":
        if n < 4:
            raise ValueError("D_n requires n>=4")
        C = [[0]*n for _ in range(n)]
        for i in range(n):
            C[i][i] = 2
        for i in range(n-3):
            C[i][i+1] = C[i+1][i] = -1
        C[n-3][n-2] = C[n-2][n-3] = -1
        C[n-3][n-1] = C[n-1][n-3] = -1
        return C

    if G == "F":
        if n != 4:
            raise ValueError("F only supports n=4")
        C = [
            [2, -1,  0,  0],
            [-1, 2, -2,  0],
            [0, -1,  2, -1],
            [0,  0, -1,  2]
        ]
        C = _reverse_cartan(C)
        return C

    if G == "G":
        if n != 2:
            raise ValueError("G only supports n=2")
        C = [
            [2, -1],
            [-3, 2]
        ]
        C = _reverse_cartan(C)
        return C

    if G == "E":
        if n == 6:
            return [
                [2, -1,  0,  0,  0,  0],
                [-1, 2, -1,  0,  0,  0],
                [0, -1,  2, -1,  0, -1],
                [0,  0, -1,  2, -1,  0],
                [0,  0,  0, -1,  2,  0],
                [0,  0, -1,  0,  0,  2]
            ]
        if n == 7:
            return [
                [2, -1,  0,  0,  0,  0,  0],
                [-1, 2, -1,  0,  0,  0,  0],
                [0, -1,  2, -1,  0,  0,  0],
                [0,  0, -1,  2, -1,  0, -1],
                [0,  0,  0, -1,  2, -1,  0],
                [0,  0,  0,  0, -1,  2,  0],
                [0,  0,  0, -1,  0,  0,  2]
            ]
        if n == 8:
            return [
                [2, -1,  0,  0,  0,  0,  0,  0],
                [-1, 2, -1,  0,  0,  0,  0,  0],
                [0, -1,  2, -1,  0,  0,  0,  0],
                [0,  0, -1,  2, -1,  0,  0,  0],
                [0,  0,  0, -1,  2, -1,  0, -1],
                [0,  0,  0,  0, -1,  2, -1,  0],
                [0,  0,  0,  0,  0, -1,  2,  0],
                [0,  0,  0,  0, -1,  0,  0,  2]
            ]
        raise ValueError("E only supports n=6,7,8")

    raise ValueError(f"Unsupported group {group}")


def _reverse_cartan(C: List[List[int]]) -> List[List[int]]:
    """
    Reverse the Cartan matrix to reflect the flipped Dynkin diagram.
    This is done by reversing both rows and columns.
    """
    n = len(C)
    C_rev = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            C_rev[i][j] = C[n-1-i][n-1-j]
    return C_rev


def get_adjoint_weight(group: str, n: int) -> List[int]:
    """
    Return the highest weight of the adjoint representation in Dynkin (fundamental weight) coordinates.
    These are the ω-basis coefficients from the reference.
    
    Raises ValueError if the group/n combination is not supported.
    """
    G = group.upper()
    
    if G == "A":
        if n < 1:
            raise ValueError("A_n requires n>=1")
        result = [0] * n
        result[0] = 1
        result[n - 1] = 1
        return result
    
    if G == "B":
        if n < 2:
            raise ValueError("B_n requires n>=2")
        result = [0] * n
        result[1] = 1
        return result
    
    if G == "C":
        if n < 2:
            raise ValueError("C_n requires n>=2")
        result = [0] * n
        result[0] = 2
        return result
    
    if G == "D":
        if n < 4:
            raise ValueError("D_n requires n>=4")
        result = [0] * n
        result[1] = 1
        return result
    
    if G == "E":
        if n == 6:
            return [0, 0, 0, 0, 0, 1]
        elif n == 7:
            return [0, 0, 0, 0, 0, 0, 1]
        elif n == 8:
            return [0, 0, 0, 0, 0, 0, 0, 1]
        else:
            raise ValueError("E only supports n=6,7,8")
    
    if G == "F":
        if n != 4:
            raise ValueError("F only supports n=4")
        return [0, 0, 0, 1]
    
    if G == "G":
        if n != 2:
            raise ValueError("G only supports n=2")
        return [0, 1]
    
    raise ValueError(f"Unsupported group {group}")


def compute_cartan_inverse(cartan: List[List[int]]) -> List[List[float]]:
    """
    Compute the inverse of the Cartan matrix.
    Returns C^{-1} as a list of lists (may contain fractions).
    """
    n = len(cartan)
    C = np.array(cartan, dtype=float)
    C_inv = np.linalg.inv(C)
    return C_inv.tolist()


def convert_to_root_coordinates(weight: List[int], cartan_inv: List[List[float]]) -> List[float]:
    """
    Convert from fundamental weight coordinates to simple root coordinates.
    β = C^{-1} · λ
    """
    n = len(weight)
    beta = [0.0] * n
    for i in range(n):
        for j in range(n):
            beta[i] += cartan_inv[i][j] * weight[j]
    return beta


def reflect_vector(vec: List[int], cartan: List[List[int]], index: int, group: str = "A") -> List[int]:
    """
    Hasse branching reflection (requires vec[index] > 0):
      new_vec = vec - vec[index] * (column index of Cartan)
    
    For F₄ and G₂, the Cartan matrix is pre-reversed in get_cartan() to reflect
    the rightward-pointing arrows, so no special handling is needed here.
    """
    x = vec[index]
    if x <= 0:
        raise ValueError("Can only reflect on a positive coefficient")
    col = [row[index] for row in cartan]
    return [v - x * c for v, c in zip(vec, col)]


def build_tree(group: str, n: int, selected: List[int], l: int) -> Dict:
    """Build Hasse tree starting from selected nodes up to depth l."""
    if not selected:
        raise ValueError("selected nodes list cannot be empty")
    
    sel0 = []
    for s in selected:
        if not (1 <= s <= n):
            raise ValueError(f"selected index {s} out of range [1, {n}]")
        sel0.append(s - 1)
    
    cartan = get_cartan(group, n)

    nodes = []
    edges = []
    seen: Dict[Tuple[int, ...], int] = {}

    node_id = 0
    init_vec = [1 if i in sel0 else 0 for i in range(n)]
    nodes.append({
        "id": node_id, 
        "vector": init_vec, 
        "depth": 0, 
        "path": [], 
        "parent": None
    })
    seen[tuple(init_vec)] = node_id
    queue = deque()
    queue.append((node_id, init_vec, 0, []))
    node_id += 1

    while queue:
        cur_id, vec, depth, path = queue.popleft()
        if depth >= l:
            continue

        for i, val in enumerate(vec):
            if val > 0:
                try:
                    new_vec = reflect_vector(vec, cartan, i, group)
                    new_path = path + [i + 1] 
                    tup = tuple(new_vec)

                    if tup in seen:
                        target_id = seen[tup]
                    else:
                        target_id = node_id
                        nodes.append({
                            "id": target_id,
                            "vector": new_vec,
                            "depth": depth + 1,
                            "path": new_path.copy(),
                            "parent": cur_id
                        })
                        seen[tup] = target_id
                        queue.append((target_id, new_vec, depth + 1, new_path))
                        node_id += 1

                    edges.append({
                        "source": cur_id, 
                        "target": target_id, 
                        "move": i + 1
                    })
                except ValueError:
                    continue

    depth_l = []
    for node in nodes:
        if node["depth"] == l:
            depth_l.append({
                "vector": node["vector"], 
                "path": node["path"]
            })

    return {
        "nodes": nodes, 
        "edges": edges, 
        "depth_l": depth_l
    }


def reflect_weight_by_root(weight: List[int], cartan: List[List[int]], index: int, group: str = "A") -> List[int]:

    coeff = weight[index]
    col = [row[index] for row in cartan]
    return [w - coeff * c for w, c in zip(weight, col)]


def compute_affine_action(group: str, n: int, weight: List[int], path: List[int]) -> Dict[str, Any]:

    if isinstance(weight, str):
        s = weight.strip()
        if "," in s:
            w = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
        else:
            w = [int(ch) for ch in s]
    else:
        w = list(weight)

    if len(w) != n:
        raise ValueError(f"weight length ({len(w)}) must equal n ({n})")

    cartan = get_cartan(group, n)
    rho = [1] * n
    lambda_plus_rho = [wi + ri for wi, ri in zip(w, rho)]

    v = lambda_plus_rho[:]
    applied_path = []
    steps = []
    
    steps.append(f"Starting with λ + ρ = {v}")
    steps.append(f"Path {path} means w = s_{path[0]} ∘ ... ∘ s_{path[-1]}")
    steps.append(f"Applying reflections RIGHT TO LEFT (bottom to top of Hasse diagram):")
    steps.append("")
    
    for step_num, p in enumerate(reversed(path)):
        idx = int(p) - 1
        if idx < 0 or idx >= n:
            raise ValueError(f"reflection index {p} out of range for rank {n}")
        
        old_v = v[:]
        v = reflect_weight_by_root(v, cartan, idx, group)
        applied_path.append(p)
        
        coeff = old_v[idx]
        col = [row[idx] for row in cartan]
        steps.append(f"Step {step_num + 1}: Apply s_{p}")
        steps.append(f"  v[{idx}] = {coeff}, Cartan column {p}: {col}")
        steps.append(f"  s_{p}({old_v}) = {old_v} - ({coeff})*{col} = {v}")
        steps.append("")

    final = [vi - ri for vi, ri in zip(v, rho)]
    
    # Multiply by -1 to fix sign
    final = [-x for x in final]
    
    steps.append(f"Before negation: {[vi - ri for vi, ri in zip(v, rho)]}")
    steps.append(f"After negation: {final}")
    
    cartan_inv = compute_cartan_inverse(cartan)
    root_coords = convert_to_root_coordinates(final, cartan_inv)
    
    steps.append("")
    steps.append("=== ROOT COORDINATE CONVERSION ===")
    steps.append(f"Result in fundamental weights: {final}")
    steps.append(f"Cartan inverse C^{{-1}}:")
    for i, row in enumerate(cartan_inv):
        steps.append(f"  Row {i+1}: [{', '.join(f'{x:7.4f}' for x in row)}]")
    steps.append(f"Root coordinates β = C^{{-1}} · x: [{', '.join(f'{b:7.4f}' for b in root_coords)}]")
    
    return {
        "input": w,
        "rho": rho,
        "lambda_plus_rho": lambda_plus_rho,
        "path": list(path),
        "w_lambda_plus_rho": v,
        "output": final,
        "cartan_inverse": cartan_inv,
        "root_coordinates": root_coords,
        "debug_steps": steps
    }


def compute_affine_for_all_paths(group: str, n: int, selected: List[int], l: int, weight: List[int]) -> List[Dict[str, Any]]:
    """Compute affine action for all paths in the Hasse tree."""
    tree = build_tree(group, n, selected, l)
    results = []
    
    for node in tree["nodes"]:
        d = node["depth"]
        if d == 0:
            continue
            
        path = node["path"]
        vec_at_node = node["vector"]
        
        try:
            affine_info = compute_affine_action(group, n, weight, path)
        except Exception as e:
            affine_info = {"error": str(e)}
            
        results.append({
            "depth": d,
            "path": path,
            "vector_at_node": vec_at_node,
            "affine_result": affine_info
        })
    
    return results


def compute_gradation(group: str, n: int, selected: List[int], l: int, weight: List[int]) -> List[Dict[str, Any]]:
    tree = build_tree(group, n, selected, l)
    cartan = get_cartan(group, n)
    cartan_inv = compute_cartan_inverse(cartan)
    
    sel0 = [s - 1 for s in selected]
    
    gradations = []
    
    for node in tree["nodes"]:
        if node["depth"] == 0:
            continue
            
        path = node["path"]
        
        try:
            affine_info = compute_affine_action(group, n, weight, path)
            output_weight = affine_info["output"]
            
            root_coords = convert_to_root_coordinates(output_weight, cartan_inv)
            
            root_coords = [0.0 if x == 0 else x for x in root_coords]
            
            gradation_value = sum(root_coords[i] for i in sel0)
            
            if gradation_value == 0:
                gradation_value = 0.0
            
            gradations.append({
                "depth": node["depth"],
                "path": path,
                "vector": node["vector"],
                "output_weight": output_weight,
                "root_coordinates": root_coords,
                "gradation": gradation_value,
                "active_roots": [i + 1 for i in sel0]
            })
        except Exception as e:
            gradations.append({
                "depth": node["depth"],
                "path": path,
                "error": str(e)
            })
    

    return gradations
