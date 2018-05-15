import argparse
from collections import defaultdict


def main(args):
    data = defaultdict(lambda: defaultdict(dict))
    with open(args.summary, 'r') as f:
        for line in f:
            search, split, metric, value = line.split('\t')
            data[search][split][metric] = float(value)


    greedy_acc = data['greedy']['Dev']['Accuracy']
    beam_acc = data['beam']['Dev']['Accuracy']
    shortest_acc = data['shortest']['Dev']['Accuracy']

    greedy_states = data['greedy']['Dev']['Average States']
    beam_states = data['beam']['Dev']['Average States']
    shortest_states = data['shortest']['Dev']['Average States']

    c_greedy_acc = data['constraint-greedy']['Dev']['Accuracy']
    c_beam_acc = data['constraint-beam']['Dev']['Accuracy']
    c_shortest_acc = data['constraint-shortest']['Dev']['Accuracy']

    c_greedy_states = data['constraint-greedy']['Dev']['Average States']
    c_beam_states = data['constraint-beam']['Dev']['Average States']
    c_shortest_states = data['constraint-shortest']['Dev']['Average States']

    print(f'''
        \\begin{{tabular}}{{ccc}}
            \\toprule
            \\textbf{{Method}} & \\textbf{{Accuracy}} & \\textbf{{Avg. \\#States}} \\\\
            \\midrule
            \\textsc{{Greedy}} & {greedy_acc:.1f} & {greedy_states:.1f} \\\\
            \\textsc{{Beam}} & {beam_acc:.1f} & {beam_states:.1f} \\\\
            \\textsc{{Shortest}} & {shortest_acc:.1f} & {shortest_states:.1f} \\\\
            \\midrule
            \\textsc{{Dict+Greedy}} & {c_greedy_acc:.1f} & {c_greedy_states:.1f} \\\\
            \\textsc{{Dict+Beam}} & {c_beam_acc:.1f} & {c_beam_states:.1f} \\\\
            \\textsc{{Dict+Shortest}} & {c_shortest_acc:.1f} & {c_shortest_states:.1f} \\\\
            \\bottomrule
        \\end{{tabular}}
    ''')


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('summary')
    args = argp.parse_args()
    main(args)
