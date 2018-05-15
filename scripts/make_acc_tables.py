import argparse
from collections import defaultdict


def load_metrics(filename):
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    with open(filename, 'r') as f:
        for line in f:
            model, component, split, metric, value = line.strip().split('\t')
            metrics[model][component][split][metric] = float(value)
    return metrics


def main(args):
    unconstrained_metrics = load_metrics(args.unconstrained_metrics)
    constrained_metrics = load_metrics(args.constrained_metrics)

    dist_acc = unconstrained_metrics['dist']['dist']['test']['acc']
    dist_edit = unconstrained_metrics['dist']['dist']['test']['edit']

    seq_acc = unconstrained_metrics['seq2seq']['seq2seq']['test']['acc']
    seq_edit = unconstrained_metrics['seq2seq']['seq2seq']['test']['edit']

    seqdist_acc = unconstrained_metrics['dist-seq2seq']['ensemble']['test']['acc']
    seqdist_edit = unconstrained_metrics['dist-seq2seq']['ensemble']['test']['edit']

    seqfreq_acc = unconstrained_metrics['seq2seq-reranker']['reranker']['test']['acc']
    seqfreq_edit = unconstrained_metrics['seq2seq-reranker']['reranker']['test']['edit']

    seqdistfreq_acc = unconstrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['acc']
    seqdistfreq_edit = unconstrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['edit']

    cons_seq_acc = constrained_metrics['seq2seq']['seq2seq']['test']['acc']
    cons_seq_edit = constrained_metrics['seq2seq']['seq2seq']['test']['edit']

    cons_seqdist_acc = constrained_metrics['dist-seq2seq']['ensemble']['test']['acc']
    cons_seqdist_edit = constrained_metrics['dist-seq2seq']['ensemble']['test']['edit']

    cons_seqfreq_acc = constrained_metrics['seq2seq-reranker']['reranker']['test']['acc']
    cons_seqfreq_edit = constrained_metrics['seq2seq-reranker']['reranker']['test']['edit']

    cons_seqdistfreq_acc = constrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['acc']
    cons_seqdistfreq_edit = constrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['edit']

    print(f'''
        \\begin{{tabular}}{{ccc}}
            \\toprule
            \\textbf{{Model}} & \\textbf{{Accuracy}} & \\textbf{{Edit}} \\\\
            \\midrule
            \\citet{{cotterell-EtAl:2017:EMNLP2017}} & 71.7 & 0.97 \\\\
            \\midrule
            \\textsc{{Dist}} & {dist_acc:.1f} & {dist_edit:.2f} \\\\
            \\textsc{{Seq}} & {seq_acc:.1f} & {seq_edit:.2f} \\\\
            \\textsc{{Aggr}} & {seqdist_acc:.1f} & {seqdist_edit:.2f} \\\\
            \\textsc{{Seq+Freq}} & {seqfreq_acc:.1f} & {seqfreq_edit:.2f} \\\\
            \\textsc{{Dual+Freq}} & {seqdistfreq_acc:.1f} & {seqdistfreq_edit:.2f} \\\\
            \\midrule
            \\textsc{{Seq+Dict}} & {cons_seq_acc:.1f} & {cons_seq_edit:.2f} \\\\
            \\textsc{{Aggr+Dict}} & {cons_seqdist_acc:.1f} & {cons_seqdist_edit:.2f} \\\\
            \\textsc{{Seq+Freq+Dict}} & {cons_seqfreq_acc:.1f} & {cons_seqfreq_edit:.2f} \\\\
            \\textsc{{Aggr+Freq+Dict}} & {cons_seqdistfreq_acc:.1f} & {cons_seqdistfreq_edit:.2f} \\\\
            \\bottomrule
        \\end{{tabular}}
    ''')


    seq10_nom_acc = unconstrained_metrics['seq2seq']['seq2seq']['test']['top-k-acc-((ADJ-NOM))']
    seq10_res_acc = unconstrained_metrics['seq2seq']['seq2seq']['test']['top-k-acc-((VERB-NOM))']
    seq10_age_acc = unconstrained_metrics['seq2seq']['seq2seq']['test']['top-k-acc-((SUBJECT))']
    seq10_adv_acc = unconstrained_metrics['seq2seq']['seq2seq']['test']['top-k-acc-((ADJ-ADV))']

    seqc10_nom_acc = constrained_metrics['seq2seq']['seq2seq']['test']['top-k-acc-((ADJ-NOM))']
    seqc10_res_acc = constrained_metrics['seq2seq']['seq2seq']['test']['top-k-acc-((VERB-NOM))']
    seqc10_age_acc = constrained_metrics['seq2seq']['seq2seq']['test']['top-k-acc-((SUBJECT))']
    seqc10_adv_acc = constrained_metrics['seq2seq']['seq2seq']['test']['top-k-acc-((ADJ-ADV))']

    seqdist_nom_acc = unconstrained_metrics['dist-seq2seq']['ensemble']['test']['acc-((ADJ-NOM))']
    seqdist_res_acc = unconstrained_metrics['dist-seq2seq']['ensemble']['test']['acc-((VERB-NOM))']
    seqdist_age_acc = unconstrained_metrics['dist-seq2seq']['ensemble']['test']['acc-((SUBJECT))']
    seqdist_adv_acc = unconstrained_metrics['dist-seq2seq']['ensemble']['test']['acc-((ADJ-ADV))']

    seqdist_nom_edit = unconstrained_metrics['dist-seq2seq']['ensemble']['test']['edit-((ADJ-NOM))']
    seqdist_res_edit = unconstrained_metrics['dist-seq2seq']['ensemble']['test']['edit-((VERB-NOM))']
    seqdist_age_edit = unconstrained_metrics['dist-seq2seq']['ensemble']['test']['edit-((SUBJECT))']
    seqdist_adv_edit = unconstrained_metrics['dist-seq2seq']['ensemble']['test']['edit-((ADJ-ADV))']

    seqdictdistfreq_nom_acc = constrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['acc-((ADJ-NOM))']
    seqdictdistfreq_res_acc = constrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['acc-((VERB-NOM))']
    seqdictdistfreq_age_acc = constrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['acc-((SUBJECT))']
    seqdictdistfreq_adv_acc = constrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['acc-((ADJ-ADV))']

    seqdictdistfreq_nom_edit = constrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['edit-((ADJ-NOM))']
    seqdictdistfreq_res_edit = constrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['edit-((VERB-NOM))']
    seqdictdistfreq_age_edit = constrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['edit-((SUBJECT))']
    seqdictdistfreq_adv_edit = constrained_metrics['dist-seq2seq-reranker']['ensemble']['test']['edit-((ADJ-ADV))']

    print(f'''
        \\begin{{tabular}}{{ccccccccccccccc}}
            \\toprule
            & \\multicolumn{{2}}{{c}}{{\\citet{{cotterell-EtAl:2017:EMNLP2017}}}} & \\phantom{{}} & \\multicolumn{{2}}{{c}}{{\\textsc{{Dual}}}} & \\phantom{{}} & \\multicolumn{{2}}{{c}}{{\\textsc{{Dual+Freq+Dict}}}} \\\\
            \\cmidrule{{2-3}} \\cmidrule{{5-6}} \\cmidrule{{8-9}}
            & acc & edit && acc & edit && acc & edit \\\\
            \\midrule
            \\textsc{{Nominal}} & 35.1 & 2.67 && {seqdist_nom_acc:.1f} & {seqdist_nom_edit:.2f} && {seqdictdistfreq_nom_acc:.1f} & {seqdictdistfreq_nom_edit:.2f} \\\\
            \\textsc{{Result}}  & 52.9 & 1.86 && {seqdist_res_acc:.1f} & {seqdist_res_edit:.2f} && {seqdictdistfreq_res_acc:.1f} & {seqdictdistfreq_res_edit:.2f} \\\\
            \\textsc{{Agent}}   & 65.6 & 0.78 && {seqdist_age_acc:.1f} & {seqdist_age_edit:.2f} && {seqdictdistfreq_age_acc:.1f} & {seqdictdistfreq_age_edit:.2f} \\\\
            \\textsc{{Adverb}}  & 93.3 & 0.18 && {seqdist_adv_acc:.1f} & {seqdist_adv_edit:.2f} && {seqdictdistfreq_adv_acc:.1f} & {seqdictdistfreq_adv_edit:.2f} \\\\
            \\bottomrule
        \\end{{tabular}}
    ''')

    print(f'''
        \\begin{{tabular}}{{ccccccc}}
            \\toprule
            & \\citet{{cotterell-EtAl:2017:EMNLP2017}} & \\phantom{{}} & \\textsc{{Seq}} & \\phantom{{}} & \\textsc{{Seq+Dict}} \\\\
            \\cmidrule{{2-2}} \\cmidrule{{4-4}} \\cmidrule{{6-6}}
            & top-10-acc && top-10-acc && top-10-acc \\\\
            \\midrule
            \\textsc{{Nominal}} & 70.2 && {seq10_nom_acc:.1f} && {seqc10_nom_acc:.1f} \\\\
            \\textsc{{Result}}  & 72.6 && {seq10_res_acc:.1f} && {seqc10_res_acc:.1f} \\\\
            \\textsc{{Agent}}   & 82.2 && {seq10_age_acc:.1f} && {seqc10_age_acc:.1f} \\\\
            \\textsc{{Adverb}}  & 96.5 && {seq10_adv_acc:.1f} && {seqc10_adv_acc:.1f} \\\\
            \\bottomrule
        \\end{{tabular}}
    ''')


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('unconstrained_metrics')
    argp.add_argument('constrained_metrics')
    args = argp.parse_args()
    main(args)
