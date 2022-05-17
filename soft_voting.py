import json
import os
import argparse

def main():
    nbests = []
    for cand in os.listdir(args.cand_dir):
        if cand != 'description.txt':
            with open(f"{args.cand_dir}/{cand}", "r") as f:
                nbest = json.load(f)
                nbests.append(nbest)
    
    # print(len(nbests), type(nbests))

    def preprocessing(nbests):
        ids = nbests[0].keys()
        total_nbest = {}

        for id_ in ids:
            total_nbest[id_] = {}
            for nbest in nbests:
                for info in nbest[id_]:
                    if info['text'] in total_nbest[id_].keys():
                        total_nbest[id_][info['text']] += info['probability']
                    else:
                        total_nbest[id_][info['text']] = info['probability']
        return total_nbest
    
    total_nbest = preprocessing(nbests)
    
    with open(f"{args.cand_dir}/total_nbest.json", 'w') as f:
        json.dump(total_nbest, f, indent=4, ensure_ascii=False)

    def get_preds(total_nbest):
        preds = {}
        for id_ in total_nbest.keys():
            best = sorted(total_nbest[id_].keys(), key=lambda x: total_nbest[id_][x])[-1]
            preds[id_] = best
        return preds
    
    preds = get_preds(total_nbest)

    with open(f"{args.cand_dir}/predictions.json", 'w') as f:
        json.dump(preds, f, indent=4, ensure_ascii=False)

    with open(f"{args.cand_dir}/description.txt", "w") as f:
        f.write(args.description)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--cand_dir",
        default="ensemble/soft_voting",
        type=str,
        help="dir which the candidates is in"
    )
    parser.add_argument(
        "--description",
        default="",
        type=str,
        help="description which you want put in sv dir"
    )
    args = parser.parse_args()

    if 'predictions.json' not in os.listdir(args.cand_dir):
        main()
    else:
        print("'prediction.json' is already in directory")
