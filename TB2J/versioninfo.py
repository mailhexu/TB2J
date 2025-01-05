from datetime import datetime

import TB2J


def print_license():
    print(
        f"""
TB2J version {TB2J.__version__}
Copyright (C) 2018-{datetime.now().year}  TB2J group.
This software is distributed with the 2-Clause BSD License, without any warranty. For more details, see the LICENSE file delivered with this software.

"""
    )
