import torch


def _compact_node_for_llm(node: dict) -> dict:
    return {
        "class": node.get("class", ""),
        "text": node.get("text", ""),
        "content-desc": node.get("content-desc", ""),
        "resource-id": node.get("resource-id", ""),
        "depth": node.get("depth", 0),
        "child_count": node.get("child_count", 0),
        "sibling_count": node.get("sibling_count", 0),
        "ancestors": [
            {"class": a.get("class", ""), "resource-id": a.get("resource-id", "")}
            for a in node.get("ancestors", [])
        ],
    }


def build_graph(
    nodes,
    containment_edges,
    sibling_edges,
    features,
    labels_heuristic,
    labels_llm=None,
    include_sibling_edges=True,
) -> dict:
    """
    Returns dict with:
      x [N,d], y_heuristic [N], y_llm [N], edge_index [2,E],
      containment_edge_index [2,Ec], sibling_edge_index [2,Es],
      num_nodes, nodes (compact, for LLM pass-2).
    Containment: both directions (parent→child AND child→parent).
    """
    N = len(nodes)

    # --- y_heuristic ---
    y_h = torch.tensor(
        [-1 if lbl is None else int(lbl) for lbl in labels_heuristic],
        dtype=torch.long,
    )

    # --- y_llm (all -1 until Pass 2) ---
    if labels_llm is not None:
        y_l = torch.tensor(
            [-1 if lbl is None else int(lbl) for lbl in labels_llm],
            dtype=torch.long,
        )
    else:
        y_l = torch.full((N,), -1, dtype=torch.long)

    # --- Containment edge index (bidirectional) ---
    if containment_edges:
        src_c = [e[0] for e in containment_edges]
        dst_c = [e[1] for e in containment_edges]
        # Both directions
        c_src = src_c + dst_c
        c_dst = dst_c + src_c
        containment_ei = torch.tensor([c_src, c_dst], dtype=torch.long)
    else:
        containment_ei = torch.zeros(2, 0, dtype=torch.long)

    # --- Sibling edge index ---
    if sibling_edges:
        s_src = [e[0] for e in sibling_edges]
        s_dst = [e[1] for e in sibling_edges]
        sibling_ei = torch.tensor([s_src, s_dst], dtype=torch.long)
    else:
        sibling_ei = torch.zeros(2, 0, dtype=torch.long)

    # --- Combined edge index ---
    if include_sibling_edges and sibling_ei.shape[1] > 0:
        edge_index = torch.cat([containment_ei, sibling_ei], dim=1)
    else:
        edge_index = containment_ei

    compact_nodes = [_compact_node_for_llm(n) for n in nodes]

    return {
        "x": features,
        "y_heuristic": y_h,
        "y_llm": y_l,
        "edge_index": edge_index,
        "containment_edge_index": containment_ei,
        "sibling_edge_index": sibling_ei,
        "num_nodes": N,
        "nodes": compact_nodes,
    }
