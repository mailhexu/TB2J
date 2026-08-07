source /home/hexu/projects/myenvs/mydev/bin/activate 
sprkkr2magnon.py -s RuO2.str -e RuO2_JXC_Jij.dat -S Ru -m 0.5674 -0.5674 -t isotropic -b -k GXSG -n 200 --qpoints "G:0,0,0,X:0.5,0,0,S:0.5,0.5,0" -o ruo2_isotropic_magnon_band.png
sprkkr2magnon.py -s RuO2.str -e RuO2_JXC_Jij.dat -S Ru -m 0.5674 -0.5674 -t transverse-block-jzz -b -k GXSG -n 200 --qpoints "G:0,0,0,X:0.5,0,0,S:0.5,0.5,0" -o ruo2_with_offdiagonal_magnon_band.png
