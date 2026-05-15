from __future__ import annotations

from pathlib import Path

from agent.tools.decorator import tool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(filename: str) -> Path | None:
    raw = (filename or "").strip().strip("\"'")
    if not raw:
        return None

    p = Path(raw)
    if p.is_absolute() and p.is_file():
        return p.resolve()
    if p.is_file():
        return p.resolve()

    for root in [_repo_root(), _repo_root() / "datasets", _repo_root() / "data"]:
        candidate = root / raw
        if candidate.is_file():
            return candidate.resolve()

    return None


@tool(
    "pdb_parse",
    "Parse a .pdb structure file and summarize chains/residues/atoms. Input: filename.",
)
def pdb_parse(filename: str) -> str:
    path = _resolve_path(filename)
    if path is None:
        return f"PDB file not found: {filename}"
    if path.suffix.lower() != ".pdb":
        return f"Expected a .pdb file, got: {path.name}"

    try:
        try:
            from Bio.PDB import PDBParser
        except ImportError:
            return "PDB parser unavailable. Install with: uv pip install biopython"

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(path.stem, str(path))

        models = list(structure.get_models())
        model = models[0] if models else None
        if model is None:
            return f"Failed to parse model from {path.name}"

        chains = list(model.get_chains())
        chain_lines: list[str] = []
        total_residues = 0
        total_atoms = 0
        for chain in chains:
            residues = list(chain.get_residues())
            atoms = list(chain.get_atoms())
            total_residues += len(residues)
            total_atoms += len(atoms)
            chain_lines.append(
                f"- Chain {chain.id}: {len(residues)} residues, {len(atoms)} atoms"
            )

        return "\n".join(
            [
                f"PDB Summary: {path.name}",
                f"Models   : {len(models)}",
                f"Chains   : {len(chains)}",
                f"Residues : {total_residues}",
                f"Atoms    : {total_atoms}",
                "",
                "Per-chain:",
                *chain_lines,
            ]
        )
    except Exception as e:
        return f"PDB parse failed: {e}"
