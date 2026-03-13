# write metadata for w matrices
# run: chmod +x scripts/experiment/run_write_mnc_md.sh
# run: scripts/experiment/run_write_mnc_md.sh
# all pairs = [(66,66), (822,822), (35,35), (800,800), (2000,2000), (115,115), (185,185), (343,343), (512,512), (363,363)]
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w/w_35_35.mtx\" M1_O=\"./data/w/withmnc/w_35_35.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w/w_66_66.mtx\" M1_O=\"./data/w/withmnc/w_66_66.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w/w_822_822.mtx\" M1_O=\"./data/w/withmnc/w_822_822.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w/w_800_800.mtx\" M1_O=\"./data/w/withmnc/w_800_800.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w/w_2000_2000.mtx\" M1_O=\"./data/w/withmnc/w_2000_2000.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w/w_115_115.mtx\" M1_O=\"./data/w/withmnc/w_115_115.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w/w_185_185.mtx\" M1_O=\"./data/w/withmnc/w_185_185.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w/w_343_343.mtx\" M1_O=\"./data/w/withmnc/w_343_343.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w/w_512_512.mtx\" M1_O=\"./data/w/withmnc/w_512_512.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w/w_363_363.mtx\" M1_O=\"./data/w/withmnc/w_363_363.csv\"

# w sparse
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w_sparse/w_35_35.mtx\" M1_O=\"./data/w_sparse/withmnc/w_35_35.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w_sparse/w_66_66.mtx\" M1_O=\"./data/w_sparse/withmnc/w_66_66.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w_sparse/w_822_822.mtx\" M1_O=\"./data/w_sparse/withmnc/w_822_822.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w_sparse/w_800_800.mtx\" M1_O=\"./data/w_sparse/withmnc/w_800_800.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w_sparse/w_2000_2000.mtx\" M1_O=\"./data/w_sparse/withmnc/w_2000_2000.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w_sparse/w_115_115.mtx\" M1_O=\"./data/w_sparse/withmnc/w_115_115.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w_sparse/w_185_185.mtx\" M1_O=\"./data/w_sparse/withmnc/w_185_185.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w_sparse/w_343_343.mtx\" M1_O=\"./data/w_sparse/withmnc/w_343_343.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w_sparse/w_512_512.mtx\" M1_O=\"./data/w_sparse/withmnc/w_512_512.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/w_sparse/w_363_363.mtx\" M1_O=\"./data/w_sparse/withmnc/w_363_363.csv\"

# write metadata for real matrices
# all datasets: bcsstk02 (66,66), bp__1000 (822,822), bp_1200 (822,822), football (35,35), G10 (800,800), 
# G11 (800,800), G14(800,800), G22 (2000,2000), G27 (2000,2000), gre__115 (115,115), gre__185 (185,185), 
# gre__343 (343,343), gre__512 (512,512), str__200 (363,363), str__400 (363,363), lp_scagr7 (129, 185), 
# football__115 (115,115), dw256A (512,512), dwt__66 (66,66)
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/bcsstk02.mtx\" M1_O=\"./data/real/withmnc/bcsstk02.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/bp__1000.mtx\" M1_O=\"./data/real/withmnc/bp__1000.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/bp__1200.mtx\" M1_O=\"./data/real/withmnc/bp__1200.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/football.mtx\" M1_O=\"./data/real/withmnc/football.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/G10.mtx\" M1_O=\"./data/real/withmnc/G10.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/G11.mtx\" M1_O=\"./data/real/withmnc/G11.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/G14.mtx\" M1_O=\"./data/real/withmnc/G14.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/G22.mtx\" M1_O=\"./data/real/withmnc/G22.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/G27.mtx\" M1_O=\"./data/real/withmnc/G27.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/gre__115.mtx\" M1_O=\"./data/real/withmnc/gre__115.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/gre__185.mtx\" M1_O=\"./data/real/withmnc/gre__185.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/gre__343.mtx\" M1_O=\"./data/real/withmnc/gre__343.csv\" 
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/gre__512.mtx\" M1_O=\"./data/real/withmnc/gre__512.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/str__200.mtx\" M1_O=\"./data/real/withmnc/str__200.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/str__400.mtx\" M1_O=\"./data/real/withmnc/str__400.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/lp_scagr7.mtx\" M1_O=\"./data/real/withmnc/lp_scagr7.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/football__115.mtx\" M1_O=\"./data/real/withmnc/football__115.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/dw256A.mtx\" M1_O=\"./data/real/withmnc/dw256A.csv\"
bin/daphne scripts/experiment/write_mnc_md.daph M1_WITHOUT_MNC=\"./data/real/dwt__66.mtx\" M1_O=\"./data/real/withmnc/dwt__66.csv\"
