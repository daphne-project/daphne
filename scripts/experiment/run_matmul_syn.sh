# run: chmod +x scripts/experiment/run_matmul_syn.sh
# run: scripts/experiment/run_matmul_syn.sh

# all datasets: bcsstk02 (66,66), bp__1000 (822,822), bp_1200 (822,822), football (35,35), G10 (800,800), 
# G11 (800,800), G14(800,800), G22 (2000,2000), G27 (2000,2000), gre__115 (115,115), gre__185 (185,185), 
# gre__343 (343,343), gre__512 (512,512), str__200 (363,363), str__400 (363,363), lp_scagr7 (129, 185), 
# football__115 (115,115), dw256A (512,512), dwt__66 (66,66)
bin/daphne scripts/experiment/matmul_syn.daph M1_WITHOUT_MNC=\"./data/real/football.mtx\" M1_WITH_MNC=\"./data/real/withmnc/football.csv\"

