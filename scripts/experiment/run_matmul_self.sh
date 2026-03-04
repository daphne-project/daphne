# run: chmod +x scripts/experiment/run_matmul_self.sh
# run: scripts/experiment/run_matmul_self.sh

# all datasets: bcsstk02 (66,66), bp__1000 (822,822), bp_1200 (822,822), football (35,35), G10 (800,800), 
# G11 (800,800), G14(800,800), G22 (2000,2000), G27 (2000,2000), gre__115 (115,115), gre__185 (185,185), 
# gre__343 (343,343), gre__512 (512,512), str__200 (363,363), str__400 (363,363), lp_scagr7 (129, 185), 
# football__115 (115,115), dw256A (512,512), dwt__66 (66,66)
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/bcsstk02.mtx\" M1_WITH_MNC=\"./data/real/withmnc/bcsstk02.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/bp__1000.mtx\" M1_WITH_MNC=\"./data/real/withmnc/bp__1000.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/bp__1200.mtx\" M1_WITH_MNC=\"./data/real/withmnc/bp__1200.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/football.mtx\" M1_WITH_MNC=\"./data/real/withmnc/football.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/G10.mtx\" M1_WITH_MNC=\"./data/real/withmnc/G10.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/G11.mtx\" M1_WITH_MNC=\"./data/real/withmnc/G11.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/G14.mtx\" M1_WITH_MNC=\"./data/real/withmnc/G14.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/G22.mtx\" M1_WITH_MNC=\"./data/real/withmnc/G22.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/G27.mtx\" M1_WITH_MNC=\"./data/real/withmnc/G27.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/gre__115.mtx\" M1_WITH_MNC=\"./data/real/withmnc/gre__115.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/gre__185.mtx\" M1_WITH_MNC=\"./data/real/withmnc/gre__185.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/gre__343.mtx\" M1_WITH_MNC=\"./data/real/withmnc/gre__343.csv\" 
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/gre__512.mtx\" M1_WITH_MNC=\"./data/real/withmnc/gre__512.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/str__200.mtx\" M1_WITH_MNC=\"./data/real/withmnc/str__200.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/str__400.mtx\" M1_WITH_MNC=\"./data/real/withmnc/str__400.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/football__115.mtx\" M1_WITH_MNC=\"./data/real/withmnc/football__115.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/dw256A.mtx\" M1_WITH_MNC=\"./data/real/withmnc/dw256A.csv\"
bin/daphne scripts/experiment/matmul_self.daph M1_WITHOUT_MNC=\"./data/real/dwt__66.mtx\" M1_WITH_MNC=\"./data/real/withmnc/dwt__66.csv\"
