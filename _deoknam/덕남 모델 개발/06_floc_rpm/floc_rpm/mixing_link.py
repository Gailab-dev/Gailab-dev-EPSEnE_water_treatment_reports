# -*- coding: utf-8 -*-
"""05_mixing_gvalue 재사용 브리지 — sys.path 조작은 이 파일에만 격리.

05 폴더명이 숫자로 시작해 패키지 import가 불가하므로 디렉터리를 path에 넣고
내부 패키지 `mixing_g`를 import한다. K0 모델(표 2.3-13 오차 <2% 검증 완료)은
복제하지 않고 재사용한다. 다른 모듈은 `from . import mixing_link`로만 접근.
"""
import sys

from .config import MIXING_G_DIR

if str(MIXING_G_DIR) not in sys.path:
    sys.path.insert(0, str(MIXING_G_DIR))

from mixing_g import gvalue, water                       # noqa: E402
from mixing_g.gvalue import K0, g_from_rpm, rpm_for_g    # noqa: E402, F401
from mixing_g.water import G_ACC, props                  # noqa: E402, F401
