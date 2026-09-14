#!/bin/bash

set -xe

# Default values
ratio=1.0

# Loop through command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--ratio)
            ratio="$2"
            shift 2
            ;;
        *) # Unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")" 

default_box_size=3.82

box_size=$(awk "BEGIN {print $default_box_size*$ratio}")

gmx editconf -f EC.gro -o conf_1.gro -box $box_size $box_size $box_size

gmx insert-molecules -f conf_1.gro -ci EC.gro -o conf_2.gro -nmol 169 -try 15000

gmx insert-molecules -f conf_2.gro -ci DMC.gro -o conf_3.gro -nmol 170 -try 15000

mv conf_3.gro solvent_salt.gro

rm -f conf_*.gro
