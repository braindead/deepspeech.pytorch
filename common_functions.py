import os
import math
import subprocess
import soundfile as sf

FNULL = open("/dev/null")

def ts_to_seconds(ts):
    h, m, s = ts.split(":")
    sec = int(h)*60*60 + int(m)*60 + float(s)
    return sec

def seconds_to_ts(sec):
    sec, ms = str(sec).split('.')
    sec = int(sec)
    h = int(math.floor(sec / 3600))
    m = int(math.floor((sec / 60 ) - (h * 60)))
    s = sec % 60 
    ts = ""
    ts += str(h) + ":"
    ts += str(m).zfill(2) + ":"
    ts += str(s).zfill(2)

    return ts + "." + ms

def sox_trim(input_path, output_path, start_time, duration):
    ret = subprocess.call(["sox", input_path, output_path, "trim", str(start_time), str(duration)], stdout=FNULL, stderr=FNULL)
    if ret != 0:
        raise Exception("sox trim failed for input {}, output {}, start {}, duration {}".format(input_path, output_path, start_time, duration))

def get_duration(wav):
    f = sf.SoundFile(wav)
    return len(f)/float(f.samplerate)

def convert_to_wav(input_file, output_file):
    ret = subprocess.call(["sox", input_file, "-r", "16k", "-e", "float", output_file, "remix", "-"], stdout=FNULL, stderr=FNULL)
    if ret != 0:
        raise Exception("sox wav conversion failed for {}, output {}".format(input_file, output_file))
