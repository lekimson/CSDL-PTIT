def dfs(graph, start):

    stack = [start]

    visited = set()


    while stack:

        vertex = stack.pop()

        if vertex not in visited:

            print(vertex, end=' ')

            visited.add(vertex)

            # Thêm các nút kề vào ngăn xếp (đảo ngược để duyệt theo thứ tự)

            neighbors = reversed(graph[vertex])

            stack.extend(neighbors)

            print(f"Đã đẩy các nút kề của '{vertex}' vào ngăn xếp: {list(neighbors)}")

    print()


# Minh họa

graph = {

    'A': ['B', 'C'],

    'B': ['D', 'E'],

    'C': ['F'],

    'D': [],

    'E': ['F'],

    'F': []

}


print("DFS từ nút A:")

dfs(graph, 'A')  # Output: A C F B E D