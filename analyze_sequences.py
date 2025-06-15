import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np

# ---------------------
# 1. 读取FASTA序列
# ---------------------
def read_sequences(filepath):
    sequences = []
    with open(filepath, "r") as f:
        seq = ""
        for line in f:
            if line.startswith(">"):
                if seq:
                    sequences.append(seq.replace("-", ""))
                seq = ""
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq.replace("-", ""))
    return sequences

# ---------------------
# 2. 可视化函数
# ---------------------
def plot_length_distribution(sequences):
    lengths = [len(s) for s in sequences]
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=20, color='cornflowerblue', edgecolor='black')
    plt.title("序列长度分布")
    plt.xlabel("长度")
    plt.ylabel("数量")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("length_distribution.png")
    plt.close()

def plot_amino_acid_distribution(sequences):
    aa_counter = Counter("".join(sequences))
    aa_freq_sorted = dict(sorted(aa_counter.items()))
    plt.figure(figsize=(10, 4))
    plt.bar(aa_freq_sorted.keys(), aa_freq_sorted.values(), color='salmon')
    plt.title("氨基酸频率分布")
    plt.xlabel("氨基酸")
    plt.ylabel("出现次数")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("aa_frequency.png")
    plt.close()

# ---------------------
# 3. 关键残基统计
# ---------------------
def check_key_residues(sequences, key_positions, key_amino_acids):
    match_count = 0
    for seq in sequences:
        if all(pos < len(seq) and seq[pos] == aa for pos, aa in zip(key_positions, key_amino_acids)):
            match_count += 1
    print(f"关键残基完全匹配的序列数: {match_count}/{len(sequences)}")
    print(f"匹配率: {match_count / len(sequences) * 100:.2f}%")

# ---------------------
# 4. t-SNE聚类
# ---------------------
def tsne_embedding(sequences, max_len=100):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    aa2idx = {aa: i for i, aa in enumerate(AA)}

    def encode(seq):
        vec = np.zeros((max_len, len(AA)))
        for i, aa in enumerate(seq[:max_len]):
            if aa in aa2idx:
                vec[i, aa2idx[aa]] = 1
        return vec.flatten()

    encoded = np.array([encode(seq) for seq in sequences])
    embedded = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(encoded)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1])
    plt.title("t-SNE 蛋白序列聚类可视化")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsne_plot.png")
    plt.close()

# ---------------------
# 5. 比对示意输出
# ---------------------
def compare_examples(sequences, num=3):
    ref = sequences[0]
    for i in range(1, min(num + 1, len(sequences))):
        test = sequences[i]
        print(f"\n--- 比对: 序列 {i} vs 参考 ---")
        print("参考: ", ref[:80])
        print("生成: ", test[:80])
        match = "".join(["|" if ref[j] == test[j] else "." for j in range(min(len(ref), len(test), 80))])
        print("匹配: ", match)

# ---------------------
# 主函数入口
# ---------------------
def main():
    sequences = read_sequences("data/gen_0.fasta")
    print(f"共读取序列数: {len(sequences)}")

    plot_length_distribution(sequences)
    plot_amino_acid_distribution(sequences)

    # 关键位点分析（默认 T114, F123, A220, M248, A317）
    key_positions = [113, 122, 219, 247, 316]
    key_amino_acids = ['T', 'F', 'A', 'M', 'A']
    check_key_residues(sequences, key_positions, key_amino_acids)

    tsne_embedding(sequences)
    compare_examples(sequences)

if __name__ == "__main__":
    main()
