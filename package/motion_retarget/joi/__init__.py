from .joi_auto_gen import get_joi_body_name_of_auto_gen,get_T_joi_of_auto_gen
from .joi_baymax import get_joi_body_name_of_baymax,get_T_joi_of_baymax
from .joi_cmu import get_T_joi_of_chain_cmu
from .joi_cr import get_joi_body_name_of_cr, get_T_joi_of_cr
from .joi_g1 import get_joi_body_name_of_g1, get_T_joi_of_g1
from .joi_humanoid import get_joi_body_name_of_humanoid, get_T_joi_of_humanoid
from .joi_jf3 import get_joi_body_name_of_jf3, get_T_joi_of_jf3
from .joi_mahru import get_joi_body_name_of_mahru, get_T_joi_of_mahru
from .joi_smpl import get_T_joi_of_chain_smpl
from .joi_talos import get_joi_body_name_of_talos, get_T_joi_of_talos
from .joi_tiago_dual import get_joi_body_name_of_tiago_dual, get_T_joi_of_tiago_dual
from .joi_op3 import get_joi_body_name_of_op3, get_T_joi_of_op3
from .joi_h1 import get_joi_body_name_of_h1, get_T_joi_of_h1
from .joi_baxter import get_joi_body_name_of_baxter, get_T_joi_of_baxter
from .joi_digit import get_joi_body_name_of_digit, get_T_joi_of_digit
from .joi_ergoCub import get_joi_body_name_of_ergocub, get_T_joi_of_ergocub
from .joi_alex import get_joi_body_name_of_alex, get_T_joi_of_alex
from .joi_pr2 import get_joi_body_name_of_pr2, get_T_joi_of_pr2
from .joi_romeo import get_joi_body_name_of_romeo, get_T_joi_of_romeo
from .joi_gundam import get_joi_body_name_of_gundam, get_T_joi_of_gundam
from .joi_iCub import get_joi_body_name_of_iCub, get_T_joi_of_iCub
from .joi_jaxon import get_joi_body_name_of_jaxon, get_T_joi_of_jaxon
from .joi_jvrc import get_joi_body_name_of_jvrc, get_T_joi_of_jvrc
from .joi_poppy import get_joi_body_name_of_poppy, get_T_joi_of_poppy
from .joi_roid1 import get_joi_body_name_of_roid1, get_T_joi_of_roid1
from .joi_valkyrie import get_joi_body_name_of_valkyrie, get_T_joi_of_valkyrie
from .joi_nextage import get_joi_body_name_of_nextage, get_T_joi_of_nextage
from .joi_sciurus import get_joi_body_name_of_sciurus, get_T_joi_of_sciurus
from .joi_igus_op import get_joi_body_name_of_igus_op, get_T_joi_of_igus_op
from .joi_armar4 import get_joi_body_name_of_armar4, get_T_joi_of_armar4
from .joi_armar6 import get_joi_body_name_of_armar6, get_T_joi_of_armar6
from .joi_hrp4 import get_joi_body_name_of_hrp4, get_T_joi_of_hrp4
from .joi_bruce import get_joi_body_name_of_bruce, get_T_joi_of_bruce
from .joi_themis import get_joi_body_name_of_themis, get_T_joi_of_themis
from .joi_apollo import get_joi_body_name_of_apollo, get_T_joi_of_apollo
from .joi_booster import get_joi_body_name_of_booster, get_T_joi_of_booster
from .joi_x1 import get_joi_body_name_of_x1, get_T_joi_of_x1
from .joi_robonaut2 import get_joi_body_name_of_robonaut2, get_T_joi_of_robonaut2
from .joi_rby1 import get_joi_body_name_of_rby1, get_T_joi_of_rby1
from .joi_pepper import get_joi_body_name_of_pepper, get_T_joi_of_pepper
from .joi_irae import get_joi_body_name_of_irae, get_T_joi_of_irae
from .joi_lego_v2 import get_joi_body_name_of_lego_v2, get_T_joi_of_lego_v2
from .joi_lego_easy import get_joi_body_name_of_lego_easy, get_T_joi_of_lego_easy
from .joi_lego_middle import get_joi_body_name_of_lego_middle, get_T_joi_of_lego_middle
from .joi_lego_hard import get_joi_body_name_of_lego_hard, get_T_joi_of_lego_hard



T_JOI_FUNC = {
    'g1'        : get_T_joi_of_g1,
    'cr'        : get_T_joi_of_cr,
    'humanoid'  : get_T_joi_of_humanoid,
    'baymax'    : get_T_joi_of_baymax,
    'jf3'       : get_T_joi_of_jf3,
    'mahru'     : get_T_joi_of_mahru,
    'auto_gen'  : get_T_joi_of_auto_gen,
    'talos'     : get_T_joi_of_talos,
    'cmu'       : get_T_joi_of_chain_cmu,
    'smpl'      : get_T_joi_of_chain_smpl,
    'tiago_dual': get_T_joi_of_tiago_dual,
    'op3'       : get_T_joi_of_op3,
    'h1'        : get_T_joi_of_h1,
    'baxter'    : get_T_joi_of_baxter,
    'digit'     : get_T_joi_of_digit,
    'ergocub'   : get_T_joi_of_ergocub,
    'alex'      : get_T_joi_of_alex,
    'pr2'       : get_T_joi_of_pr2,
    'romeo'     : get_T_joi_of_romeo,
    'gundam'    : get_T_joi_of_gundam,
    'icub'      : get_T_joi_of_iCub,
    'jaxon'     : get_T_joi_of_jaxon,
    'jvrc'      : get_T_joi_of_jvrc,
    'poppy'     : get_T_joi_of_poppy,
    'roid1'     : get_T_joi_of_roid1,
    'valkyrie'  : get_T_joi_of_valkyrie,
    'nextage'   : get_T_joi_of_nextage,
    'sciurus'   : get_T_joi_of_sciurus,
    'igus_op'   : get_T_joi_of_igus_op,
    'armar4'    : get_T_joi_of_armar4,
    'armar6'    : get_T_joi_of_armar6,
    'pepper'    : get_T_joi_of_pepper,
    'hrp4'      : get_T_joi_of_hrp4,
    'bruce'     : get_T_joi_of_bruce,
    'themis'    : get_T_joi_of_themis,
    'apollo'    : get_T_joi_of_apollo,
    'booster'   : get_T_joi_of_booster,
    'x1'        : get_T_joi_of_x1,
    'robonaut2' : get_T_joi_of_robonaut2,
    'rby1'      : get_T_joi_of_rby1,
    'irae'      : get_T_joi_of_irae,
    'lego_v2'   : get_T_joi_of_lego_v2,
    'lego_easy' : get_T_joi_of_lego_easy, # use lego_v2 function for lego_easy as well
    'lego_middle': get_T_joi_of_lego_middle, # use lego_v2
    'lego_hard' : get_T_joi_of_lego_hard, # use lego_v2 function for lego_hard as well
}

JOI_BODY_NAME_FUNC = {
    'auto_gen'  : get_joi_body_name_of_auto_gen,
    'cr'        : get_joi_body_name_of_cr,
    'g1'        : get_joi_body_name_of_g1,
    'humanoid'  : get_joi_body_name_of_humanoid,
    'jf3'       : get_joi_body_name_of_jf3,
    'mahru'     : get_joi_body_name_of_mahru,
    'baymax'    : get_joi_body_name_of_baymax,
    'talos'     : get_joi_body_name_of_talos,
    'tiago_dual': get_joi_body_name_of_tiago_dual,
    'op3'       : get_joi_body_name_of_op3,
    'h1'        : get_joi_body_name_of_h1,
    'baxter'    : get_joi_body_name_of_baxter,
    'digit'     : get_joi_body_name_of_digit,
    'ergocub'   : get_joi_body_name_of_ergocub,
    'alex'      : get_joi_body_name_of_alex,
    'pr2'       : get_joi_body_name_of_pr2,
    'romeo'     : get_joi_body_name_of_romeo,
    'gundam'    : get_joi_body_name_of_gundam,
    'icub'      : get_joi_body_name_of_iCub,
    'jaxon'     : get_joi_body_name_of_jaxon,
    'jvrc'      : get_joi_body_name_of_jvrc,
    'poppy'     : get_joi_body_name_of_poppy,
    'roid1'     : get_joi_body_name_of_roid1,
    'valkyrie'  : get_joi_body_name_of_valkyrie,
    'nextage'   : get_joi_body_name_of_nextage,
    'sciurus'   : get_joi_body_name_of_sciurus,
    'igus_op'   : get_joi_body_name_of_igus_op,
    'armar4'    : get_joi_body_name_of_armar4,
    'armar6'    : get_joi_body_name_of_armar6,
    'pepper'    : get_joi_body_name_of_pepper,
    'hrp4'      : get_joi_body_name_of_hrp4,
    'bruce'     : get_joi_body_name_of_bruce,
    'themis'    : get_joi_body_name_of_themis,
    'apollo'    : get_joi_body_name_of_apollo,
    'booster'   : get_joi_body_name_of_booster,
    'x1'        : get_joi_body_name_of_x1,
    'robonaut2' : get_joi_body_name_of_robonaut2,
    'rby1'      : get_joi_body_name_of_rby1,
    'irae'      : get_joi_body_name_of_irae,
    'lego_v2'   : get_joi_body_name_of_lego_v2,
    'lego_easy' : get_joi_body_name_of_lego_easy, # use lego_v2 function for lego_easy as well
    'lego_middle': get_joi_body_name_of_lego_middle, # use lego_v2 function for lego_middle as well
    'lego_hard' : get_joi_body_name_of_lego_hard, # use lego_v2 function for lego_hard as well
}