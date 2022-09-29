try:
    from .datasets import (
        InMemoryProteinGraphDataset,
        ProteinGraphDataset,
        ProteinGraphListDataset,
    )
except (ImportError, NameError):
    pass