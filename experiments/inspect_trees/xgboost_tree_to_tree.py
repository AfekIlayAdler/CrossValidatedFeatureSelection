class Node:
    def __init__(self, text, depth):
        self.text = text
        self.left = None
        self.right = None
        self.depth = depth

    def fill_data(self):
        return


nodes = ['0:[f2<2.45000005] yes=1,no=2,missing=1',
         '\t1:leaf=-0.717703402',
         '\t2:[f3<1.75] yes=3,no=4,missing=3',
         '\t\t3:[f2<4.94999981] yes=5,no=6,missing=5',
         '\t\t\t5:leaf=1.38805974',
         '\t\t\t6:leaf=-3.25116254e-08',
         '\t\t4:[f2<4.85000038] yes=7,no=8,missing=7',
         '\t\t\t7:leaf=-2.5544848e-08',
         '\t\t\t8:leaf=-0.712707222',
         '']


def get_child_str(index, depth, l):
    temp_l = l[index:]
    for i, v in enumerate(temp_l):
        if v.count('\t') == depth:
            return temp_l[i]


root = Node(nodes[0], depth=0)
queue = [[root]]
depth = 1
while len(queue) > 0:
    level_nodes = queue.pop(0)
    next_level_nodes = []
    for node in level_nodes:
        node_index = nodes.index(node.text)
        left_child_text = nodes[node_index + 1]
        left_child_is_leaf = True if "leaf" in left_child_text else False
        right_child_text = get_child_str(node_index + 2, depth, nodes)
        right_child_is_leaf = True if "leaf" in right_child_text else False
        node.left = Node(left_child_text, depth)
        node.right = Node(right_child_text, depth)
        if not left_child_is_leaf:
            next_level_nodes.append(node.left)
        if not right_child_is_leaf:
            next_level_nodes.append(node.right)
        node.fill_data()
    if next_level_nodes:
        queue.append(next_level_nodes)
    depth += 1

print(root.right.left.text)
