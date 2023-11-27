import os
import pathlib

SNAP_OPT = '/usr/local/snap/bin/gpt'
JVM_MAX_MEMORY = os.environ.get('JVM_MAX_MEMORY') or '15G'
SNAPHU_BIN = 'snaphu'
GRAPH_DIR = '/app/graph'
from params import INPUT_PATH_1, INPUT_PATH_2, TMP_PATH, OUTPUT_PATH, FIRST_BURST, LAST_BURST, SWATH

def gen_DEM_pharse1(input1, input2, graph, wk_dir, kwargs):
    fisrt_burst = kwargs.get("fisrt_burst") or 1
    last_burst = kwargs.get("last_burst") or 2
    swath = kwargs.get("swath") or "IW2"
    output = "{}/S1A_IW_SLC__1SDV_20221215T145804_20221215T145831_046344_058CFC_75C4_Orb_Stack_Ifg_Deb_Flt.dim".format(wk_dir)
    snap_command = "{} {} -c {} -Pinput1={} -Pinput2={} -Pswath={} -PfirstBurst={} -PlastBurst={} -Poutput={}".format(
        SNAP_OPT, graph, JVM_MAX_MEMORY, input1, input2, swath, fisrt_burst, last_burst, output
    )

    os.system(snap_command)
    return output

def graphu_export(input, graph, wks_dir):
    snaphu_wks = '{}/snaphu'.format(wks_dir)
    if not os.path.exists(snaphu_wks):
        os.makedirs(snaphu_wks)
    snap_command = "{} {} -c {} -Pinput={} -Poutput={}".format(
        SNAP_OPT, graph, JVM_MAX_MEMORY, input, snaphu_wks
    )
    os.system(snap_command)
    name = os.listdir(snaphu_wks)[0]
    ouput_dir = f'{snaphu_wks}/{name}'
    return ouput_dir

def gen_DEM_pharse2(input, input_snaphu, graph, output_tif):
    for name in os.listdir(input_snaphu):
        if ("UnwPhase_ifg" in name) and ("snaphu.hdr" in name):
            input_unr = "{}/{}".format(input_snaphu, name)
    
    # output_tif = '{}/DEM.tif'.format(tmp_dir)
    snap_command = "{} {} -c {} -Pinput={} -Pinputunr={} -Poutput={}".format(
        SNAP_OPT, graph, JVM_MAX_MEMORY, input, input_unr, output_tif
    )
    os.system(snap_command)
    return output_tif

def gen_DEM_coherence(input, graph, tmp_dir):
    band_dir = input.replace("dim", "data")
    for name in os.listdir(band_dir):
        if ("coh" in name) and ("hdr" in name):
            band = name.split(".")[0]

    output_tif = f'{tmp_dir}/coherence.tif'
    snap_command = "{} {} -c {} -Pinput={} -Pband={} -Poutput={}".format(
        SNAP_OPT, graph, JVM_MAX_MEMORY, input, band, output_tif
    )

    os.system(snap_command)
    return output_tif

def snaphu_unrapping(wk_dir, linelength=25580):
    os.chdir(wk_dir)

    for name in os.listdir(wk_dir):
        if ("Phase_ifg" in name) and ("snaphu.img" in name):
            input = name

    snaphu_conf = 'snaphu.conf'

    snaphu_command = "{} -f {} {} {}".format(
        SNAPHU_BIN, snaphu_conf, input, linelength
    )
    os.system(snaphu_command)


def gen_DEM_sen1(input_raw1, input_raw2, tmp_dir, output_tif, kwargs):
    # creat DEM image from 2 sentinel1 raw image
    if kwargs.get("fisrt_burst") == kwargs.get("last_burst"):
        pharse1_graph = f'{GRAPH_DIR}/DEM_pharse1_1burst.xml'
    else:
        pharse1_graph = f'{GRAPH_DIR}/DEM_pharse1.xml'
    out_pharse1 = gen_DEM_pharse1(input_raw1, input_raw2, pharse1_graph, tmp_dir, kwargs)
    
    graph_export = f'{GRAPH_DIR}/DEM_graphu_export.xml'
    graphu_wks = graphu_export(out_pharse1, graph_export, tmp_dir)

    print('Execute pharse 1 done ...', GRAPH_DIR)
    snaphu_unrapping(graphu_wks)
    print('Wrapping done ...', GRAPH_DIR)
    graph_pharse2 = f'{GRAPH_DIR}/DEM_pharse2.xml'
    DEM_tif = gen_DEM_pharse2(out_pharse1, graphu_wks, graph_pharse2, output_tif)

    print('Finish......')
    return DEM_tif

if __name__ == '__main__':
    input_raw1 = INPUT_PATH_1
    input_raw2 = INPUT_PATH_2
    tmp_dir = TMP_PATH
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    kwargs = {
        "swath": SWATH,
        "fisrt_burst": FIRST_BURST,
        "last_burst": LAST_BURST
    }
    result = gen_DEM_sen1(input_raw1, input_raw2, tmp_dir, OUTPUT_PATH, kwargs)




