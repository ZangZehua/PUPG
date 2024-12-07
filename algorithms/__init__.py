from .ocbt import PUPG as ocbt
from .ocby import Rainbow as ocby
from .pckl import Rainbow as pckl
from .pcwdnn import Rainbow as pcwdnn
from .pcwdnw import Rainbow as pcwdnw
from .pcwdns import Rainbow as pcwdns
from .pcwdws import Rainbow as pcwdws
from .pcwdww import Rainbow as pcwdww
from .pcwdss import Rainbow as pcwdss
from .pupgsc import Rainbow as pupgsc
from .pupgmc import Rainbow as pupgmc
from .pupgscby import Rainbow as pupgscby

from .pupgscbypro import Rainbow as pupgscbypro
from .pcwdwsl2 import Rainbow as pcwdwsl2

atari_agents = {
    "ocbt": ocbt,
    "ocby": ocby,
    "pckl": pckl,
    "pcwdnn": pcwdnn,
    "pcwdnw": pcwdnw,
    "pcwdns": pcwdns,
    "pcwdws": pcwdws,
    "pcwdss": pcwdss,
    "pcwdww": pcwdww,
    "pupgsc": pupgsc,
    "pupgmc": pupgmc,
    "pupgscby": pupgscby,

    "pupgscbypro": pupgscbypro,
    "pcwdwsl2": pcwdwsl2
}
