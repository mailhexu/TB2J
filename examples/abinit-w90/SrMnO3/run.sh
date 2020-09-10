#!/usr/bin/env bash
abinit < abinit.files | tee abinit.log
wannier90.x abinito_w90_up.win
wannier90.x abinito_w90_down.win
sh ./get_J.sh

