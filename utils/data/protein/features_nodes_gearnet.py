"""
Appendix C.1 Protein Graph Construction
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterable, Optional

import numpy as np
import pandas as pd

from graphein.protein.resi_atoms import (
    BASE_AMINO_ACIDS,
    HYDROGEN_BOND_ACCEPTORS,
    HYDROGEN_BOND_DONORS,
    RESI_THREE_TO_1,
)


def amino_acid_one_hot_additional(
    n,
    d,
    return_array: bool = True,
    allowable_set: Optional[List[str]] = BASE_AMINO_ACIDS,
) -> Union[pd.Series, np.ndarray]:
    """Adds a one-hot encoding of amino acid types as a node attribute.
    :param n: node name, this is unused and only included for compatibility with the other functions
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :param return_array: If True, returns a numpy array of one-hot encoding, otherwise returns a pd.Series. Default is True.
    :type return_array: bool
    :param allowable_set: Specifies vocabulary of amino acids. Default is None (which uses `graphein.protein.resi_atoms.STANDARD_AMINO_ACIDS`).
    :return: One-hot encoding of amino acid types
    :rtype: Union[pd.Series, np.ndarray]
    """
    from utils.data.protein import onek_encoding_unk_additional
    features = onek_encoding_unk_additional(
        RESI_THREE_TO_1[d["residue_name"] ], allowable_set
    )
    
    if return_array:
        features = np.array(features).astype(int)
    else:
        features = pd.Series(features).astype(int)
        features.index = allowable_set

    d["node_feat"] = features
    return features

