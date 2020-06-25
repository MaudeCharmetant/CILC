# -*- coding: utf-8 -*-
# 
#  This file is part of the CILC code.
# 
#  the CILC code is free software; you can redistribute it and/or modify
#  it under the terms of the MIT License.
# 
#  CILC is distributed in the hope that it will be useful,but 
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
#  the provided copy of the MIT License for more details.

"""
  The code group all the function that are needed to perform an all sky ILC, CILC or NILC. 
"""

__version__ = "1.0"

__bibtex__ = """
"""

from .CILC import (smooth2reso, mergemaps, map2fields, covcorr_matrix, D_I_CMB, D_I_tSZ, mixing_vector_CMB, mixing_vector_tSZ, CILC_weights, ILC_weights, All_sky_ILC)
