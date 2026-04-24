import re
import random
import heapq
from collections import defaultdict

class TextGraphLab:
    def __init__(self):
        # graph[from_word][to_word] = weight
        self.graph = defaultdict(lambda: defaultdict(int))
        self.words = []
        self.nodes = set()
        self.random_generator = random.Random()

    # =========================
    # 基础：读取文件、预处理、建图
    # =========================
    def load_file(self, file_path: str):
        """
        从文本文件读取内容，预处理后构建有向图
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.words = self.tokenize(text)
        self.build_graph(self.words)

    def tokenize(self, text: str):
        """
        文本预处理规则：
        1. 全部转小写
        2. 非字母字符全部替换为空格
        3. 按空白分词
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z]+", " ", text)
        words = text.split()
        return words

    def build_graph(self, words):
        """
        构建有向图：
        如果 a 后面跟着 b，则建立 a -> b，权重 +1
        """
        self.graph.clear()
        self.nodes.clear()

        for w in words:
            self.nodes.add(w)

        for i in range(len(words) - 1):
            a = words[i]
            b = words[i + 1]
            self.graph[a][b] += 1

        # 保证所有节点都在 nodes 中，即使没有出边
        for w in words:
            _ = self.graph[w]

    def has_word(self, word: str) -> bool:
        return word in self.nodes

    # =========================
    # 功能1：展示有向图
    # =========================
    def showDirectedGraph(self, G=None, export_dot_path=None):
        """
        以文本形式展示图结构
        如果提供 export_dot_path，则导出 dot 文件
        """
        if G is None:
            G = self.graph

        if not self.nodes:
            print("图为空，请先加载文本文件。")
            return

        print("\n========== Directed Graph ==========")
        for node in sorted(self.nodes):
            neighbors = G.get(node, {})
            if neighbors:
                edge_str = ", ".join(
                    f"{to}({weight})" for to, weight in sorted(neighbors.items())
                )
                print(f"{node} -> {edge_str}")
            else:
                print(f"{node} -> (no outgoing edges)")
        print("====================================\n")

        if export_dot_path:
            self.export_to_dot(export_dot_path)
            print(f"已导出 DOT 文件到: {export_dot_path}")
            print("可使用 Graphviz 生成图片，例如：")
            print(f'  dot -Tpng "{export_dot_path}" -o graph.png')

    def export_to_dot(self, dot_path: str):
        """
        导出 Graphviz DOT 文件
        """
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write("digraph TextGraph {\n")
            f.write('    rankdir=LR;\n')
            f.write('    node [shape=ellipse, fontname="Arial"];\n')
            for u in sorted(self.nodes):
                if not self.graph[u]:
                    f.write(f'    "{u}";\n')
                for v, w in sorted(self.graph[u].items()):
                    f.write(f'    "{u}" -> "{v}" [label="{w}"];\n')
            f.write("}\n")

    # =========================
    # 功能2：查询桥接词
    # =========================
    def get_bridge_words_list(self, word1: str, word2: str):
        """
        返回 word1 和 word2 之间的所有桥接词列表
        桥接词定义：word1 -> bridge 且 bridge -> word2
        """
        word1 = word1.lower()
        word2 = word2.lower()

        if not self.has_word(word1) or not self.has_word(word2):
            return None

        bridges = []
        for mid in self.graph[word1]:
            if word2 in self.graph[mid]:
                bridges.append(mid)
        return bridges

    def queryBridgeWords(self, word1: str, word2: str) -> str:
        word1 = word1.lower()
        word2 = word2.lower()

        has1 = self.has_word(word1)
        has2 = self.has_word(word2)

        if not has1 and not has2:
            return f'No "{word1}" and "{word2}" in the graph!'
        if not has1:
            return f'No "{word1}" in the graph!'
        if not has2:
            return f'No "{word2}" in the graph!'

        bridges = self.get_bridge_words_list(word1, word2)
        if not bridges:
            return f'No bridge words from "{word1}" to "{word2}"!'

        if len(bridges) == 1:
            return f'The bridge word from "{word1}" to "{word2}" is: {bridges[0]}.'
        else:
            bridge_str = ", ".join(bridges[:-1]) + " and " + bridges[-1]
            return f'The bridge words from "{word1}" to "{word2}" are: {bridge_str}.'

    # =========================
    # 功能3：生成新文本
    # =========================
    def generateNewText(self, inputText: str) -> str:
        """
        对输入文本的每对相邻单词，若存在桥接词，则随机插入一个桥接词
        """
        raw_words = self.tokenize(inputText)
        if len(raw_words) <= 1:
            return " ".join(raw_words)

        result = []
        for i in range(len(raw_words) - 1):
            w1 = raw_words[i]
            w2 = raw_words[i + 1]
            result.append(w1)

            bridges = self.get_bridge_words_list(w1, w2)
            if bridges:
                chosen = self.random_generator.choice(bridges)
                result.append(chosen)

        result.append(raw_words[-1])
        return " ".join(result)

    # =========================
    # 功能4：最短路径
    # =========================
    def _dijkstra(self, start: str):
        """
        从 start 出发，计算到所有节点的最短距离
        边权就是图中的权重（出现次数）
        """
        dist = {node: float("inf") for node in self.nodes}
        prev = {node: None for node in self.nodes}
        dist[start] = 0

        heap = [(0, start)]

        while heap:
            cur_dist, u = heapq.heappop(heap)
            if cur_dist > dist[u]:
                continue

            for v, weight in self.graph[u].items():
                new_dist = dist[u] + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

        return dist, prev

    def _reconstruct_path(self, prev, start, end):
        if start == end:
            return [start]
        if prev[end] is None:
            return None

        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        if path[0] != start:
            return None
        return path

    def calcShortestPath(self, word1: str, word2: str = None) -> str:
        """
        1. 如果输入两个单词：计算二者最短路径
        2. 如果只输入一个单词：计算该词到其他所有可达单词的最短路径
        """
        word1 = word1.lower()

        if not self.has_word(word1):
            return f'No "{word1}" in the graph!'

        dist, prev = self._dijkstra(word1)

        # 只输入一个单词：到所有点的最短路径
        if word2 is None or word2.strip() == "":
            lines = [f'Shortest paths from "{word1}":']
            reachable = False
            for node in sorted(self.nodes):
                if node == word1:
                    continue
                if dist[node] == float("inf"):
                    lines.append(f'  to "{node}": unreachable')
                else:
                    reachable = True
                    path = self._reconstruct_path(prev, word1, node)
                    path_str = " -> ".join(path)
                    lines.append(f'  to "{node}": {path_str} (length={dist[node]})')
            if not reachable and len(self.nodes) > 1:
                lines.append("No reachable nodes.")
            return "\n".join(lines)

        word2 = word2.lower()
        if not self.has_word(word2):
            return f'No "{word2}" in the graph!'

        if dist[word2] == float("inf"):
            return f'No path from "{word1}" to "{word2}"!'

        path = self._reconstruct_path(prev, word1, word2)
        path_str = " -> ".join(path)
        return f'Shortest path from "{word1}" to "{word2}": {path_str} (length={dist[word2]})'

    # =========================
    # 功能5：PageRank
    # =========================
    def calPageRank(self, word: str, d: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        """
        计算指定单词的 PageRank
        处理悬挂节点（无出边）时，将其 PR 均匀分配给所有节点
        """
        word = word.lower()
        if not self.has_word(word):
            return None

        nodes = sorted(self.nodes)
        n = len(nodes)
        if n == 0:
            return None

        pr = {node: 1.0 / n for node in nodes}

        for _ in range(max_iter):
            new_pr = {node: (1 - d) / n for node in nodes}

            # 处理每个节点对其他节点的贡献
            for u in nodes:
                out_neighbors = self.graph[u]
                if len(out_neighbors) == 0:
                    # 悬挂节点：平均分给所有节点
                    share = d * pr[u] / n
                    for v in nodes:
                        new_pr[v] += share
                else:
                    out_degree = len(out_neighbors)
                    share = d * pr[u] / out_degree
                    for v in out_neighbors:
                        new_pr[v] += share

            diff = sum(abs(new_pr[node] - pr[node]) for node in nodes)
            pr = new_pr
            if diff < tol:
                break

        return pr[word]

    def get_all_pagerank(self, d: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        """
        返回所有节点的 PageRank
        """
        nodes = sorted(self.nodes)
        n = len(nodes)
        if n == 0:
            return {}

        pr = {node: 1.0 / n for node in nodes}

        for _ in range(max_iter):
            new_pr = {node: (1 - d) / n for node in nodes}

            for u in nodes:
                out_neighbors = self.graph[u]
                if len(out_neighbors) == 0:
                    share = d * pr[u] / n
                    for v in nodes:
                        new_pr[v] += share
                else:
                    out_degree = len(out_neighbors)
                    share = d * pr[u] / out_degree
                    for v in out_neighbors:
                        new_pr[v] += share

            diff = sum(abs(new_pr[node] - pr[node]) for node in nodes)
            pr = new_pr
            if diff < tol:
                break

        return pr

    # =========================
    # 功能6：随机游走
    # =========================
    def randomWalk(self, save_path="random_walk.txt") -> str:
        """
        从随机节点开始随机游走。
        停止条件：
        1. 当前节点没有出边
        2. 即将走一条已走过的边
        """
        if not self.nodes:
            return "图为空，请先加载文本文件。"

        current = self.random_generator.choice(list(self.nodes))
        path = [current]
        visited_edges = set()

        while True:
            neighbors = list(self.graph[current].keys())
            if not neighbors:
                break

            nxt = self.random_generator.choice(neighbors)
            edge = (current, nxt)

            if edge in visited_edges:
                path.append(nxt)
                break

            visited_edges.add(edge)
            path.append(nxt)
            current = nxt

        result = " -> ".join(path)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(result)

        return result

def print_menu():
    print("""
================= Lab1 Menu =================
1. Load text file and build graph
2. Show directed graph
3. Query bridge words
4. Generate new text
5. Calculate shortest path
6. Calculate PageRank
7. Random walk
8. Export graph to DOT file
9. Show all PageRank values
0. Exit
=============================================
""")

def main():
    app = TextGraphLab()

    while True:
        print_menu()
        choice = input("Please enter your choice: ").strip()

        if choice == "1":
            file_path = input("请输入文本文件路径: ").strip()
            try:
                app.load_file(file_path)
                print("文本加载成功，图已构建。")
                print(f"单词总数: {len(app.words)}")
                print(f"节点数: {len(app.nodes)}")
                edge_count = sum(len(v) for v in app.graph.values())
                print(f"边数: {edge_count}")
            except Exception as e:
                print(f"加载失败: {e}")

        elif choice == "2":
            app.showDirectedGraph()

        elif choice == "3":
            word1 = input("请输入第一个单词: ").strip()
            word2 = input("请输入第二个单词: ").strip()
            print(app.queryBridgeWords(word1, word2))

        elif choice == "4":
            text = input("请输入一段新文本: ").strip()
            new_text = app.generateNewText(text)
            print("生成的新文本：")
            print(new_text)

        elif choice == "5":
            word1 = input("请输入起点单词: ").strip()
            word2 = input("请输入终点单词（若留空则计算到所有点）: ").strip()
            if word2 == "":
                print(app.calcShortestPath(word1))
            else:
                print(app.calcShortestPath(word1, word2))

        elif choice == "6":
            word = input("请输入要查询 PageRank 的单词: ").strip()
            pr = app.calPageRank(word)
            if pr is None:
                print(f'No "{word}" in the graph!')
            else:
                print(f'PageRank("{word}") = {pr:.6f}')

        elif choice == "7":
            result = app.randomWalk()
            print("随机游走结果：")
            print(result)
            print("已保存到 random_walk.txt")

        elif choice == "8":
            dot_path = input("请输入导出的 DOT 文件名（如 graph.dot）: ").strip()
            if not dot_path:
                dot_path = "graph.dot"
            try:
                app.export_to_dot(dot_path)
                print(f"DOT 文件已导出到: {dot_path}")
                print(f'可执行: dot -Tpng "{dot_path}" -o graph.png')
            except Exception as e:
                print(f"导出失败: {e}")

        elif choice == "9":
            all_pr = app.get_all_pagerank()
            if not all_pr:
                print("图为空，请先加载文本文件。")
            else:
                print("All PageRank values:")
                for word, score in sorted(all_pr.items(), key=lambda x: (-x[1], x[0])):
                    print(f"{word}: {score:.6f}")

        elif choice == "0":
            print("程序结束。")
            break

        else:
            print("无效输入，请重新选择。")

if __name__ == "__main__":
    main()