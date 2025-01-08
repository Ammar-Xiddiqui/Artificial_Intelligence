from collections import deque

def bfs(graph,start,end):
    visit=set([start])
    que=deque([[start]])
    while que:
        path=que.popleft()
        val= path[-1]
        if val==end:
            return path
        for i in graph[val]:
            if i not in visit:
                npath=list(path)
                npath.append(i)
                que.append(npath)
                visit.add(i)

    return None


graph ={
    1 : [2,3],
    2 : [6],
    3 : [],
    4 : [5],
    5 : [1,2],
    6 : [3,4]

}


graph1 ={
    1 : [2,3],
    2 : [6,1],
    3 : [],
    4 : [5],
    5 : [1,2],
    6 : [3,4]

}

print(bfs(graph1,2,3))



                   
               






































# def dfs(start, goal, visited=None, depth=0, max_depth=20):  # Default max depth is 20
#     if visited is None:
#         visited = set()

#     if is_goal(start, goal):
#         return []

#     if depth == max_depth:  # Terminate if max depth is reached
#         return None

#     visited.add(tuple(map(tuple, start)))

#     for direction in MOVES:
#         next_state = move(start, direction)
#         if next_state and tuple(map(tuple, next_state)) not in visited:
#             result = dfs(next_state, goal, visited, depth + 1, max_depth)
#             if result is not None:
#                 return [direction] + result

#     visited.remove(tuple(map(tuple, start)))
#     return None


