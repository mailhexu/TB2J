#!/usr/bin/env bash
siesta < siesta.fdf | tee siesta.log
sh getJ.sh
