
def dfs(graph, start, end, visit=None, path=None):
    if visit is None:
        visit = set([start])
    if path is None:
        path = [start]
    
    if start == end:
        return path
    
    for neighbor in graph[start]:
        if neighbor not in visit:
            visit.add(neighbor)
            npath = list(path)
            npath.append(neighbor)
            result = dfs(graph, neighbor, end, visit, npath)
            if result:
                return result
    
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

print(dfs(graph1,2,3))


