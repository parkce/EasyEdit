from ..trainer import MEND
from ..trainer import SERAC, SERAC_MULTI
from ..trainer import CoRE_MULTI
from ..trainer import MALMEN


ALG_TRAIN_DICT = {
    'MEND': MEND,
    'SERAC': SERAC,
    'SERAC_MULTI': SERAC_MULTI,
    'CORE_MULTI': CoRE_MULTI,
    'MALMEN': MALMEN,
}