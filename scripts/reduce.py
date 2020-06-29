import argparse
import csv
import json

import sklearn.manifold
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description="MiniConf Portal Command Line")
    parser.add_argument("papers", default=False, help="paper file")

    parser.add_argument("embeddings", default=False, help="embeddings file to shrink")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    emb = torch.load(args.embeddings)
    out = sklearn.manifold.TSNE(n_components=2).fit_transform(emb.numpy())
    d = []
    with open(args.papers, "r",encoding='utf-8') as f:
        abstracts = list(csv.DictReader(f))
        for i, row in enumerate(abstracts):
            d.append({"id": row["UID"], "author": row['author'], "department": row['Department/Institute Affiliation'], "talk": row['talk'], "pos": out[i].tolist(),"abstract": row['abstract']})
    print(json.dumps(d))
