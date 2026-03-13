# run: chmod +x scripts/experiment/run_elementwise_add.sh
# run: scripts/experiment/run_elementwise_add.sh

# all datasets: bcsstk02 (66,66), bp__1000 (822,822), bp_1200 (822,822), football (35,35), G10 (800,800), 
# G11 (800,800), G14(800,800), G22 (2000,2000), G27 (2000,2000), gre__115 (115,115), gre__185 (185,185), 
# gre__343 (343,343), gre__512 (512,512), str__200 (363,363), str__400 (363,363), lp_scagr7 (129, 185), 
# football__115 (115,115), dw256A (512,512), dwt__66 (66,66)
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/bcsstk02.mtx\" M1_WITH_MNC=\"./data/real/withmnc/bcsstk02.csv\" N=66 M=66
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/bp__1000.mtx\" M1_WITH_MNC=\"./data/real/withmnc/bp__1000.csv\" N=822 M=822
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/bp__1200.mtx\" M1_WITH_MNC=\"./data/real/withmnc/bp__1200.csv\" N=822 M=822
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/football.mtx\" M1_WITH_MNC=\"./data/real/withmnc/football.csv\" N=35 M=35
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/G10.mtx\" M1_WITH_MNC=\"./data/real/withmnc/G10.csv\" N=800 M=800
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/G11.mtx\" M1_WITH_MNC=\"./data/real/withmnc/G11.csv\" N=800 M=800
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/G14.mtx\" M1_WITH_MNC=\"./data/real/withmnc/G14.csv\" N=800 M=800
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/G22.mtx\" M1_WITH_MNC=\"./data/real/withmnc/G22.csv\" N=2000 M=2000
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/G27.mtx\" M1_WITH_MNC=\"./data/real/withmnc/G27.csv\" N=2000 M=2000
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/gre__115.mtx\" M1_WITH_MNC=\"./data/real/withmnc/gre__115.csv\" N=115 M=115
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/gre__185.mtx\" M1_WITH_MNC=\"./data/real/withmnc/gre__185.csv\" N=185 M=185
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/gre__343.mtx\" M1_WITH_MNC=\"./data/real/withmnc/gre__343.csv\"  N=343 M=343
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/gre__512.mtx\" M1_WITH_MNC=\"./data/real/withmnc/gre__512.csv\" N=512 M=512
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/str__200.mtx\" M1_WITH_MNC=\"./data/real/withmnc/str__200.csv\" N=363 M=363
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/str__400.mtx\" M1_WITH_MNC=\"./data/real/withmnc/str__400.csv\" N=363 M=363
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/lp_scagr7.mtx\" M1_WITH_MNC=\"./data/real/withmnc/lp_scagr7.csv\" N=129 M=185
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/football__115.mtx\" M1_WITH_MNC=\"./data/real/withmnc/football__115.csv\"  N=115 M=115
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/dw256A.mtx\" M1_WITH_MNC=\"./data/real/withmnc/dw256A.csv\" N=512 M=512
bin/daphne scripts/experiment/elementwise_add.daph M1_WITHOUT_MNC=\"./data/real/dwt__66.mtx\" M1_WITH_MNC=\"./data/real/withmnc/dwt__66.csv\" N=66 M=66
