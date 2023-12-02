"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import matplotlib.pyplot as plt
def plot_distribution(distribution,dataset_name,train=True):
    plt.bar(range(len(distribution)), distribution, color = "dodgerblue")
    plt.title(f"{dataset_name}", fontsize = 14)
    plt.xlabel("Class index", fontsize = 14)
    plt.ylabel("Num of images", fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    # mean = sum(distribution) / len(distribution)
    # plt.plot([0, len(distribution)], [mean, mean], color='red', linestyle='--')
    # plt.legend(["mean"])
    # plt.ylim([0,1000])
    plt.savefig(f"./{dataset_name} {'trainset' if train else 'valset'} class_distribution.pdf")
    plt.show()
    plt.close()
    head_num = int(len(distribution) * 0.3)
    medium_num = int(len(distribution) * 0.4)
    tail_num = len(distribution) - head_num - medium_num
    head = distribution[:head_num]
    medium = distribution[head_num:head_num + medium_num]
    tail = distribution[-tail_num:]
    print(dataset_name)
    print(f"head:{head_num},medium:{medium_num},tail:{tail_num}")
    medium_mean = sum(medium) / len(medium)
    tail_mean = sum(tail) / len(tail)
    print(f"medium_mean:{medium_mean},tail_mean:{tail_mean}")
    print(f"tail_mean/medium_mean:{tail_mean/medium_mean}")