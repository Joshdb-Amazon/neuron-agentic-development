import torch
import torch.nn as nn
import json
import os
import hashlib
from typing import Dict, Any, List, Tuple

from transformers import AutoModelForCausalLM, set_seed

# # Get ranks from environment variables
# rank = int(os.environ["RANK"])
# local_rank = int(os.environ["LOCAL_RANK"])
# world_size = int(os.environ["WORLD_SIZE"])

# # ────────── 1. set one global seed *before* the model is created ───────
# MASTER_SEED = 1234            # pick any int
# set_seed(MASTER_SEED)         # sets python / numpy / torch
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def module_to_compressed_tree(
    module: nn.Module,
    name: str = "",
    seen_structures: Dict[str, Dict[str, Any]] = None
) -> Dict[str, Any]:
    if seen_structures is None:
        seen_structures = {}

    def structure_hash(mod: nn.Module) -> str:
        """Hash structure for deduplication based on module repr only."""
        return hashlib.md5(str(mod).encode()).hexdigest()

    h = structure_hash(module)

    # If seen, just add the name to the existing module and skip expansion
    if h in seen_structures:
        seen_structures[h]["names"].append(name)
        return {"__ref__": h}  # Placeholder to signal parent to not duplicate

    # Otherwise, create a new node
    node_dict = {
        "type": module.__class__.__name__,
        "names": [name],
        "children": [],
    }
    seen_structures[h] = node_dict  # Register this module before recursing

    for child_name, child_module in module.named_children():
        child = module_to_compressed_tree(child_module, name=child_name, seen_structures=seen_structures)
        if child is not None and "__ref__" not in child:
            node_dict["children"].append(child)
        # If child is already seen, its name was added above, so we skip appending again

    return node_dict

def module_to_compressed_dict(
    module: nn.Module,
    name: str = "",
    seen_structures: Dict[str, Dict[str, Any]] = None
) -> Dict[str, Any]:
    if seen_structures is None:
        seen_structures = {}

    def structure_hash(mod: nn.Module) -> str:
        """Hash structure for deduplication based on module repr only."""
        rep = str(mod)
        return hashlib.md5(rep.encode()).hexdigest()

    h = structure_hash(module)
    if h in seen_structures:
        seen_structures[h]["names"].append(name)
        return None  # skip re-inserting duplicate
    else:
        node_dict = {
            "type": module.__class__.__name__,
            # "repr": str(module).replace("\n", ""),
            # "parameters": {
            #     pname: list(p.shape) for pname, p in module.named_parameters(recurse=False)
            # },
            "children": [],
            "names": [name]
        }

        for child_name, child_module in module.named_children():
            child_dict = module_to_compressed_dict(child_module, name=child_name, seen_structures=seen_structures)
            if child_dict is not None:
                node_dict["children"].append(child_dict)

        seen_structures[h] = node_dict
        return node_dict
  

def compress_module_tree(tree: dict) -> dict:
    import hashlib

    def structure_hash(subtree: dict) -> str:
        """Hash full structure: type + param keys + child types + structure."""
        m = hashlib.md5()
        m.update(subtree["type"].encode())
        if "parameters" in subtree:
            for k in sorted(subtree["parameters"].keys()):
                m.update(k.encode())
        for child in subtree.get("children", []):
            m.update(child["type"].encode())
            m.update(child.get("name", "").encode())
            m.update(structure_hash(child).encode())
        return m.hexdigest()

    def compress_node(node: dict) -> dict:
        local_seen = {}  # reset for each parent
        compressed_children = []

        for child in node.get("children", []):
            # Compress the child recursively
            compressed_child = compress_node(child)
            h = structure_hash(compressed_child)

            if h in local_seen:
                # Merge name into names of already seen identical child
                local_seen[h]["names"].append(*compressed_child["names"])
            else:
                # First time seeing this structure in this parent
                compressed_child["names"] = compressed_child.pop("names")
                local_seen[h] = compressed_child
                compressed_children.append(compressed_child)

        # Build new node without modifying the original
        return {
            "type": node["type"],
            "names": [node.get("name", "")],
            # "parameters": node.get("parameters", {}),
            "children": compressed_children
        }
        
    compressed_tree = compress_node(tree)

    # for every node, sort the keys by "type", "names", "children"
    def sort_node(node: dict) -> dict:
        sorted_children = [sort_node(child) for child in node["children"]]
        
        return {
            "type": node["type"],
            "names": node["names"],
            "children": sorted_children
        }
        
    compressed_tree = sort_node(compressed_tree)

    return compressed_tree
        


def module_to_dict(module: nn.Module, name: str = "") -> dict:
    module_dict = {
        "name": name,
        "type": module.__class__.__name__,
        # "repr": str(module).replace("\n", ""),
        "parameters": {
            pname: list(p.shape) for pname, p in module.named_parameters(recurse=False)
        },
        "children": []
    }

    for child_name, child_module in module.named_children():
        child_dict = module_to_dict(child_module, name=child_name)
        module_dict["children"].append(child_dict)

    return module_dict


def expand_node(node: Dict[str, Any], prefixes: List[str]) -> List[str]:
    """
    Recursively expand a compressed module dict into a flat list of module paths.

    node: {
        "type": "...",
        "names": ["name1", "name2", ...],
        "children": [ ... ]
    }
    prefixes: list of current path prefixes to which this node's names will be appended
    """
    all_paths = []

    names = node.get("names", [""])
    children = node.get("children", [])

    # if this node has no names, keep prefixes as-is
    if not names:
        current_paths = prefixes[:]  # copy
    else:
        current_paths = []
        for prefix in prefixes:
            for nm in names:
                if nm == "" or nm is None:
                    current_paths.append(prefix)
                else:
                    if prefix:
                        current_paths.append(f"{prefix}.{nm}")
                    else:
                        current_paths.append(nm)

    # this node itself is a module at these current_paths
    all_paths.extend(current_paths)

    # recurse into children, using current_paths as the parent prefixes
    for child in children:
        all_paths.extend(expand_node(child, current_paths))

    return all_paths

def prune_module_tree(node: dict, suppressed_modules: List[str]) -> dict:
    """Keep node, but drop its children if it is shared_rotary."""
    # Determine if this node is a shared_rotary node
    should_suppress = False
    for full_name in node.get("names", []):
        if full_name.split(".")[-1] in suppressed_modules:
            should_suppress = True
            break
        
    if should_suppress:
        return None
    # Otherwise prune each child
    pruned_children = []
    for child in node.get("children", []):
        child = prune_module_tree(child, suppressed_modules)
        if child is None:
            continue
        pruned_children.append(prune_module_tree(child, suppressed_modules))
    node["children"] = pruned_children
    return node

def get_module_tree_and_flat_paths(model: nn.Module, suppressed_modules: List[str] = []) -> Tuple[dict, List[str]]:
    tree = module_to_dict(model, name="")
    compressed_tree = compress_module_tree(tree)
    compressed_tree = prune_module_tree(compressed_tree, suppressed_modules)
    flat_paths = expand_node(compressed_tree, [""])
    pruned_flat_paths = []
    for name in flat_paths:
        should_suppress = False
        for suppressed_module in suppressed_modules:
            if suppressed_module in name:
                should_suppress = True
                break
        if should_suppress:
            continue
        pruned_flat_paths.append(name)
    return compressed_tree, pruned_flat_paths

def summarize_names(names):
    """
    Rules:
    - If numeric & contiguous → "[start..end]".
    - If numeric & NOT contiguous → show ALL numbers explicitly, NO ellipsis.
    - Otherwise (text names): ellipsis if long.
    """
    if not names:
        return "", 0

    n = len(names)

    # Try numeric case
    all_int = False
    ints = []
    try:
        ints = [int(x) for x in names]
        all_int = True
    except ValueError:
        all_int = False

    if all_int:
        uniq_sorted = sorted(set(ints))

        # Contiguous → compress
        if len(uniq_sorted) > 1 and uniq_sorted == list(range(uniq_sorted[0], uniq_sorted[-1] + 1)):
            return f"[{uniq_sorted[0]}..{uniq_sorted[-1]}]", n

        # NON-contiguous numeric → list ALL numbers, NO ellipsis
        expanded = ", ".join(str(i) for i in uniq_sorted)
        return f"[{expanded}]", n

    # Non-numeric names → use ellipsis for long lists
    if n <= 8:
        preview = ", ".join(names)
    else:
        preview = ", ".join(names[:4] + ["..."] + names[-2:])
    return f"[{preview}]", n



def build_tree_string(node, prefix="", is_last=True, lines=None):
    """
    Recursively builds a list of lines representing the compressed tree.
    Does not print — returns a string at the end.
    """

    if lines is None:
        lines = []

    children = node.get("children", [])

    # Handle ModuleList + child with repeating names (layers)
    if node["type"] == "ModuleList" and len(children) == 1:
        child = children[0]
        child_names = child.get("names") or []
        if len(child_names) > 1:
            # ModuleList name(s) - handle multiple names (e.g., linear_q/k/v_adapter_list)
            mod_names = node.get("names") or [""]
            if len(mod_names) == 1:
                mod_name = mod_names[0]
                idx_summary, count = summarize_names(child_names)
                label = f"{mod_name}{idx_summary} (N={count}): {child['type']}"
            else:
                # Multiple ModuleList names - show all of them
                mod_summary, mod_count = summarize_names(mod_names)
                idx_summary, count = summarize_names(child_names)
                label = f"{mod_summary}{idx_summary} (N={count}): {child['type']}"

            lines.append(prefix + ("└── " if is_last else "├── ") + label)

            new_prefix = prefix + ("    " if is_last else "│   ")

            # Expand its children once
            for i, gc in enumerate(child.get("children", [])):
                build_tree_string(gc, new_prefix, i == len(child["children"]) - 1, lines)
            return lines

    # Otherwise — regular node
    names = node.get("names") or []
    node_type = node["type"]

    if len(names) == 1 and names[0]:
        # Single named node → "name: Type"
        label = f"{names[0]}: {node_type}"
    elif len(names) > 1:
        # Multi-name group → "[a,b,...]: Type (N=k)"
        summary, count = summarize_names(names)
        label = f"{summary} (N={count}): {node_type}"
    else:
        # No names → just type
        label = node_type

    lines.append(prefix + ("└── " if is_last else "├── ") + label)

    # Prefix for children
    new_prefix = prefix + ("    " if is_last else "│   ")

    for i, c in enumerate(children):
        build_tree_string(c, new_prefix, i == len(children) - 1, lines)

    return lines


def compressed_tree_to_pretty_string(root):
    """Top-level wrapper: returns the full ASCII tree as a single string."""
    lines = []

    # Root name + type
    root_type = root.get("type")
    root_names = root.get("names") or [""]

    if root_names[0]:  # e.g., "model"
        lines.append(f"{root_names[0]}: {root_type}")
    else:
        lines.append(root_type)

    # Process children
    children = root.get("children", [])
    for i, child in enumerate(children):
        build_tree_string(child, "", i == len(children) - 1, lines)

    return "\n".join(lines)

import re
from itertools import product
from typing import Dict, List, Optional, Any, Set


def delete_path(root_node: Dict[str, Any], path: str) -> bool:
    """
    Delete the node at the given dot-separated path in the *expanded* tree:

      - root_node: {"name": str, "type": str, "children": [...]}
      - path: e.g. "model.model.layers.3.self_attn.q_norm"

    Returns True if a node was deleted, False otherwise.
    """
    if not path:
        return False
    segments = path.split(".")

    def _delete(node: Dict[str, Any], segs: List[str]) -> bool:
        if not segs:
            return False
        head, *tail = segs

        for i, child in enumerate(node.get("children", [])):
            if child.get("name") == head:
                if not tail:
                    # Exact match → delete this child
                    del node["children"][i]
                    return True
                else:
                    return _delete(child, tail)
        return False

    return _delete(root_node, segments)


def remove_mapped_components(
    tree_json: Dict[str, Any],
    mapped_components: List[str],
    variables: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Remove mapped components from a *compressed* module tree.

    Behavior:
      - We only remove a node if:
          (1) its own path is mapped, AND
          (2) all nodes in its subtree are mapped.
      - If some descendants are unmapped, we keep the node and only drop
        children whose subtrees are fully mapped.

    Input:
        tree_json: compressed tree from get_model_tree, e.g.:
          { "type": "...", "names": ["", ...], "children": [...] }

        mapped_components: list of path templates (may contain {"i"} etc.)

        variables: dict mapping variable -> list of values
                   (or None if paths are concrete).

    Output:
        A NEW tree dict in the SAME compressed schema:
          { "type": ..., "names": [...], "children": [...] }
    """

    # ---------- helpers (local) ----------

    def _expand(node_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand compressed node (with 'names') into list of expanded nodes:

            {"name": str, "type": str, "children": [...]}

        This matches the shape expected by compress_module_tree().
        """
        names = node_json.get("names") or []
        node_type = node_json["type"]
        children_json = node_json.get("children") or []

        # expand children once as a template
        expanded_children_template: List[Dict[str, Any]] = []
        for c in children_json:
            expanded_children_template.extend(_expand(c))

        def _copy_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [
                {
                    "name": n["name"],
                    "type": n["type"],
                    "children": _copy_nodes(n.get("children", [])),
                }
                for n in nodes
            ]

        if names:
            # one expanded node per name
            return [
                {
                    "name": n,
                    "type": node_type,
                    "children": _copy_nodes(expanded_children_template),
                }
                for n in names
            ]
        else:
            # anonymous node (root or synthetic)
            return [
                {
                    "name": "",
                    "type": node_type,
                    "children": _copy_nodes(expanded_children_template),
                }
            ]

    PLACEHOLDER_RE = re.compile(r"\{(.*?)\}")

    def _instantiate(template: str) -> List[str]:
        """
        Instantiate a single template path using `variables`.
        If `variables` is None or no placeholders, returns [template].
        
        Handles multiple variables by generating the cartesian product of all
        variable values, ensuring every placeholder is substituted simultaneously.
        If the same variable appears multiple times, all occurrences are replaced
        with the same value from the cartesian product.
        """
        if variables is None:
            return [template]

        raw_vars = PLACEHOLDER_RE.findall(template)
        if not raw_vars:
            return [template]

        var_names = [v.strip('"').strip("'") for v in raw_vars]
        
        # Get unique variables in order of first appearance
        unique_vars = []
        seen = set()
        for v in var_names:
            if v not in seen:
                unique_vars.append(v)
                seen.add(v)
        
        # Validate all unique variables are provided
        for v in unique_vars:
            if v not in variables:
                raise ValueError(f"Variable '{v}' not provided in variables dict.")

        # Generate cartesian product of unique variable values
        value_lists = [variables[v] for v in unique_vars]
        results: List[str] = []
        for combo in product(*value_lists):
            s = template
            # Replace all occurrences of each variable with its value
            for var_name, val in zip(unique_vars, combo):
                s = s.replace("{" + var_name + "}", val)
            results.append(s)
        return results

    def _collect_mapped_paths() -> Set[str]:
        paths: Set[str] = set()
        for tmpl in mapped_components:
            for p in _instantiate(tmpl):
                paths.add(p)
        return paths

    def _prune(node: Dict[str, Any],
               parent_segments: List[str],
               mapped_paths: Set[str]) -> (Optional[Dict[str, Any]], bool):
        """
        Bottom-up prune over the *expanded* tree (name/type/children).

        Returns:
          (new_node_or_None, subtree_fully_mapped: bool)

        A subtree is fully mapped if:
          - this node's full path is in mapped_paths, AND
          - all children subtrees are fully mapped.

        If subtree is fully mapped → return (None, True) → node is deleted.
        Otherwise, keep node and drop only fully-mapped children.
        """
        segs = parent_segments.copy()
        name = node.get("name", "")
        if name:
            segs.append(name)
        full_path = ".".join(segs) if segs else ""

        # prune children first
        new_children: List[Dict[str, Any]] = []
        child_fully_flags: List[bool] = []

        for child in node.get("children", []):
            kept_child, child_fully = _prune(child, segs, mapped_paths)
            if kept_child is not None:
                new_children.append(kept_child)
                child_fully_flags.append(False)
            else:
                child_fully_flags.append(True)

        # if no children, all_children_fully = True by default
        all_children_fully = all(child_fully_flags) if child_fully_flags else True
        subtree_fully_mapped = (full_path in mapped_paths) and all_children_fully

        if subtree_fully_mapped:
            return None, True

        node["children"] = new_children
        return node, False

    # ---------- 1. Expand compressed tree to {name,type,children} ----------

    expanded_roots = _expand(tree_json)
    if len(expanded_roots) == 1:
        root = expanded_roots[0]
    else:
        # shouldn't happen for your saved trees, but keep it safe
        root = {
            "name": "",
            "type": "Root",
            "children": expanded_roots,
        }

    # ---------- 2. Compute all concrete mapped paths ----------

    mapped_paths = _collect_mapped_paths()

    # ---------- 3. Prune according to "node + all descendants mapped" ----------

    pruned_root, fully_mapped = _prune(root, [], mapped_paths)

    if pruned_root is None:
        # whole tree is fully mapped → return an empty root of same type as original
        pruned_root = {
            "name": "",
            "type": root["type"],
            "children": [],
        }

    # ---------- 4. Re-compress using your existing logic ----------

    # compress_module_tree expects an *expanded* dict with "name"/"type"/"children"
    compressed_tree = compress_module_tree(pruned_root)          # :contentReference[oaicite:2]{index=2}
    compressed_tree = prune_module_tree(compressed_tree, [])     # no extra suppressed modules

    # IMPORTANT: compressed_tree is already in {type, names, children} form.
    # DO NOT pass it through _to_json again (that’s what wiped names before).

    return compressed_tree


# with open("/home/ubuntu/diagnosis_agent_demo/bug-cases/vllm/vllm-26812/model_tree/model_tree_hf.json") as f:
#     tree = json.load(f)

# mapped_components = [
#     "model.model.layers.{j}.shared_transformer.self_attn.linear_q_adapter_list.{k}.0",
#     "model.model.layers.{j}.shared_transformer.self_attn.linear_q_adapter_list.{k}.1",
#     "model.model.layers.{j}.shared_transformer.self_attn.linear_q_adapter_list.{k}",
#     "model.model.layers.{j}.shared_transformer.self_attn.linear_q_adapter_list"
    
#     # "model.model.layers.{i}.self_attn.q_proj",
#     # "model.model.layers.{i}.self_attn.k_proj",
#     # "model.model.layers.{i}.self_attn.v_proj",
#     # "model.model"
#     # "model.model.layers.{i}.self_attn.o_proj"
#     # ...
# ]

# variables = {
#     # "i": [str(k) for k in range(32)],
#     'i': ['0', '1', '2', '3', '4', '6', '7', '8', '9', '10', '12', '13', '14', '15', '16', '18', '19', '20', '21', '22', '24', '25', '26', '27', '28', '30', '31', '32', '33', '34', '36', '37'], 
#     'j': ['5', '11', '17', '23', '29', '35'], 
#     'k': ['0', '1', '2', '3', '4', '5']
# }

# new_tree = remove_mapped_components(tree, mapped_components, variables)

# with open("model_tree_hf_pruned.json", "w") as f:
#     json.dump(new_tree, f, indent=2)

# print(compressed_tree_to_pretty_string(new_tree))

# Example usage
# enable tensor parallelism
# model = AutoModelForCausalLM.from_pretrained(
#     "/home/ubuntu/models/Llama-4-Scout-17B-16E-Instruct",
#     torch_dtype=torch.bfloat16,
#     tp_plan="auto"
# )



# tree = module_to_dict(model, name="model")

# with open("/home/ubuntu/diagnosis_agent_demo/bug-cases/vllm-16296/model_arch_vllm.json", "r") as f:
#     tree = json.load(f)

# # seen = {}
# compressed_tree = compress_module_tree(tree)

# # Save or print compressed output
# with open("compressed_tree_vllm.json", "w") as f:
#     json.dump(compressed_tree, f, indent=2)
