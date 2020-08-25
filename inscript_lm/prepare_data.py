"""
Prepare the data into a format for torchtext

{"gold_event": "5", "segment": "he then scanned my card ,"}
"""
import json

scenarios = ["train", "bicycle", "grocery", "tree", "bus", "flight", "haircut", "bath", "cake", "library"]


def write_json_torchtext(scname):
    dirpath = "/home/CE/skrjanec/data_seg_all/" + scname + "/"
    # open map file
    with open(dirpath+"map.json", "r") as jf:
        map = json.load(jf) # str_eventID : str_eventName

    new_train = open(dirpath+"join/train_line.json", "w")
    new_val = open(dirpath+"join/val_line.json", "w")

    for e in range(len(map)):
        # open train json e_train.json; "segments": [sentence1, sentence2 ...]
        with open(dirpath+str(e)+"_train.json", "r") as td:
            train_data = json.load(td)["segments"] # list of str

        for seg in train_data:
            # write e as a int
            dstring = json.dumps({'gold_event': e, 'segment': seg})
            new_train.write(dstring + "\n")

        # open val json e_val.json
        with open(dirpath+str(e)+"_val.json", "r") as td:
            val_data = json.load(td)["segments"] # list of str

        for seg in val_data:
            # write e as an int
            dstring = json.dumps({'gold_event': e, 'segment': seg})
            new_val.write(dstring + "\n")

    new_train.close()
    new_val.close()


for sc in scenarios:
    print("... current scenario", sc)
    input("ENTER for next scenario")
    write_json_torchtext(sc)
