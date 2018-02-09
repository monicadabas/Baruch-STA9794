import queue
from collections import deque


class Node:
    def __init__(self, val, left=None, right=None, depth = None):
        self.val = val
        self.left = left
        self.right = right
        self.depth = depth

    # breadth first traversal using generator
    def print_all_2(self):
        def gen(o):
            lAll = [o,]
            while lAll:
                oNext = lAll.pop(0)
                if oNext.left is not None:
                    lAll.append(oNext.left)
                if oNext.right is not None:
                    lAll.append(oNext.right)
                yield oNext
        for oNode in gen(self):
            print(oNode.val)


class BinaryTree:
    def __init__(self, root = None):
        self.root = root

    def add_node(self, node):
        if self.root is None:
            self.root = node

        else:
            self._add_node_helper(self.root, node)

    def _add_node_helper(self, node, node_x):
        if node.left is None:
            node.left = node_x
        elif node.right is None:
            node.right = node_x
        else:
            if node.left.right is None:
                self._add_node_helper(node.left, node_x)
            elif node.right.right is None:
                self._add_node_helper(node.right, node_x)
        return

    def breadth_first_traversal(self):
        if self.root is None:
            print("Empty tree")
            return
        q = queue.Queue()
        q.put_nowait(self.root)
        a = []
        while not q.empty():
            current = q.get_nowait()
            a.append(current.val)
            if current.left is not None:
                q.put_nowait(current.left)
            if current.right is not None:
                q.put_nowait(current.right)

        print(a)


class BSTree:
    def __init__(self, root = None):
        self.root = root

    def add_data(self, node):
        if self.root is None:
            node.depth = 0
            self.root = node

        else:
            self._add_data_helper(self.root, node)

    def _add_data_helper(self, node, node_x):
        if node_x.val > node.val:
            if node.right is None:
                node_x.depth = node.depth + 1
                node.right = node_x

            else:
                self._add_data_helper(node.right, node_x)
        else:
            if node.left is None:
                node_x.depth = node.depth + 1
                node.left = node_x

            else:
                self._add_data_helper(node.left, node_x)

    def print_tree_inorder(self):
        if self.root is None:
            print("Empty Tree")
            return
        self._print_tree_inorder_helper(self.root)

    def _print_tree_inorder_helper(self, node):
        if node is None:
            return

        if node.left is not None:
            self._print_tree_inorder_helper(node.left)

        print(node.val)

        if node.right is not None:
            self._print_tree_inorder_helper(node.right)

    def inorder_no_helper(self, node):
        if self.root is None:
            print("Empty Tree")
            return

        if node is None:
            return

        self.inorder_no_helper(node.left)
        print(node.val)
        self.inorder_no_helper(node.right)

    def preorder_no_helper(self, node):
        if self.root is None:
            print("Empty Tree")
            return

        if node is None:
            return

        print(node.val)
        self.preorder_no_helper(node.left)
        self.preorder_no_helper(node.right)

    def postorder_no_helper(self, node):
        if self.root is None:
            print("Empty Tree")
            return

        if node is None:
            return
        self.postorder_no_helper(node.left)
        self.postorder_no_helper(node.right)

        print(node.val)

    def breadth_first_traversal(self):
        if self.root is None:
            print("Empty tree")
            return
        q = queue.Queue()
        q.put_nowait(self.root)
        a = []
        while not q.empty():
            current = q.get_nowait()
            a.append(current.val)
            if current.left is not None:
                q.put_nowait(current.left)
            if current.right is not None:
                q.put_nowait(current.right)

        print(a)

    def find_data(self, node, x):
        if node is None:
            print("Element does not exist")
            return False

        if x == node.val:
            print("Element exists")
            return True

        elif x > node.val:
            flag = self.find_data(node.right, x)

        else:
            flag = self.find_data(node.left, x)
        return flag

    def find_min(self, node):
        if node.left is None:
            return node.val
        else:
            return self.find_min(node.left)

    def find_min_node(self, node):
        if node.left is None:
            return node
        else:
            return self.find_min_node(node.left)

    def _delete_node_helper(self,node):
        if node.left is None and node.right is None:
            return None

        elif node.left is None:
            return node.right

        elif node.right is None:
            return node.left

        else:
            temp_node = self.find_min_node(node.right)
            temp_node.right = self.delete_node(node.right, temp_node)
            temp_node.left = node.left
            return temp_node

    def delete_node(self,node,x):
        if x == node:
            return self._delete_node_helper(node)

        elif x.val > node.val and node.right is not None:
            node.right = self.delete_node(node.right, x)

        elif x.val < node.val and node.left is not None:
            node.left = self.delete_node(node.left, x)

        else:
            print("Node not found")
        return node

    def find_depth(self, node):
        if node is None:
            return -1
        else:
            return 1+ max(self.find_depth(node.left), self.find_depth(node.right))

    def find_node(self, node, node_x):
        if self.root is None:
            return False
        elif node == node_x:
            return True
        elif node.val > node_x.val: # elif node.left is not None (for non BST)
            return self.find_node(node.left, node_x)
        elif node.val < node_x.val: # elif node.right is not None (for non BST)
            return self.find_node(node.right, node_x)
        else:
            return False

    def find_common_ancestor(self, node, a, b):
        if self.root is None:
            print("Empty Tree")
            return
        if max(a.val,b.val) < node.val:
            return self.find_common_ancestor(node.left, a, b)
        elif min(a.val,b.val) > node.val:
            return self.find_common_ancestor(node.right, a, b)
        elif a.val <= node.val <= b.val or b.val <= node.val <= a.val:
            if self.find_node(node, a) and self.find_node(node, b):
                return node
            else:
                print("One of the nodes does not exist")
                return
        else:
            return

    def find_path(self, node, a, b):
        if self.root is None:
            print("Empty Tree")
            return
        low_node = a if a.val < b.val else b
        high_node = a if a.val > b.val else b

        if high_node.val < node.val and node.left is not None:
            return self.find_path(node.left, low_node, high_node)

        elif low_node.val > node.val and node.right is not None:
            return self.find_path(node.right, low_node, high_node)

        elif low_node == node:
            return self._find_path_helper(low_node, low_node, high_node)

        elif high_node == node:
            return self._find_path_helper(high_node, high_node, low_node)

        elif high_node.val > node.val > low_node.val:
            left = self._find_path_helper(node, node, low_node)
            right = self._find_path_helper(node, node.right, high_node)
            if left is None or right is None:
                return
            else:
                return left, right

        else:
            print("No path")
            return

    def _find_path_helper(self, ancestor, node, node_x):
        if node == node_x:
            return node.val

        elif node_x.val < ancestor.val:
            if node_x.val < node.val and node.left is not None:
                p = self._find_path_helper(ancestor, node.left, node_x)
                if p is not None:
                    return p, node.val
                else:
                    return
            elif node_x.val > node.val and node.right is not None:
                p = self._find_path_helper(ancestor, node.right, node_x)
                if p is not None:
                    return p, node.val
                else:
                    return
            else:
                print("Node {} does not exist".format(node_x.val))
                return

        elif node_x.val > ancestor.val:
            if node_x.val < node.val and node.left is not None:
                p = self._find_path_helper(ancestor, node.left, node_x)
                if p is not None:
                    return node.val, p
                else:
                    return
            elif node_x.val > node.val and node.right is not None:
                p = self._find_path_helper(ancestor, node.right, node_x)
                if p is not None:
                    return node.val, p
                else:
                    return
            else:
                print("Node {} does not exist".format(node_x.val))
                return

    def _path_helper(self,node, b, l=deque()):
        if node == b:
            return l.append(b)
        elif node is None or (node.left is None and node.right is None):
            return l
        else:
            if node.left is not None:
                return self._path_helper(node.left, b, l)
            if node.right is not None:
                return self._path_helper(node.right, b, l)

    def path(self, node, a, b, l=deque()):
        if node == a:
            l.append(a)
        elif node == b:
            l.append(b)
        elif node is None or (node.left is None and node.right is None):
            return l
        else:
            if node.left is not None:
                l = self.path(node.left, a, b, l)

            if not l:
                if node.right is not None:
                    l = self.path(node.right, a, b, l)
            else:
                l.append(node)
                if l[0] == a:
                    rest_l = self._path_helper(node.right, b)
                else:
                    rest_l = self._path_helper(node.right, a)

                if rest_l:
                    l.extend(rest_l)

        # if l and ((l[0] == a and l[-1] == b) or (l[0] == b and l[-1] == a)):
        #     print(i.val for i in l)
        # else:
        #     l.clear()
        return l


n1 = Node(11)
n2 = Node(6)
n3 = Node(5)
n4 = Node(9)
n5 = Node(8)
n6 = Node(7)
n7 = Node(15)
n8 = Node(12)
n9 = Node(19)
n10 = Node(14)
n11 = Node(17)
n12 = Node(22)
n13 = Node(20)
n14 = Node(10)
n15 = Node(4)

# t = BSTree()
# t.add_data(n1)
# t.add_data(n2)
# t.add_data(n3)
# t.add_data(n4)
# t.add_data(n5)
# t.add_data(n6)
# t.add_data(n7)
# t.add_data(n8)
# t.add_data(n9)
# t.add_data(n10)
# t.add_data(n11)
# t.add_data(n12)
# t.add_data(n13)
# t.add_data(n14)
# t.add_data(n15)

#t.breadth_first_traversal()

s = BinaryTree()
s.add_node(n15)
s.add_node(n3)
s.add_node(n2)
s.add_node(n6)
n3.right = n5
n5.left = n4
n5.right = n1
n2.left = n7
n2.right = n11
n11.left = n9
n11.right = n13
n13.left = n12

#n15.print_all_2()

#s.breadth_first_traversal()


""" Given a sorted list (increasing order) create a BST with minimal height """


def create_minimal_tree(array, start, end):
    if start > end:
        return
    if len(array) == 1:
        root = Node(array[0])
    else:
        mid = int((end-start)/2)
        root = Node(array[mid])
        root.left = create_minimal_tree(array[:mid], 0, mid-1)
        root.right = create_minimal_tree(array[mid+1:], 0, end-mid-1)
    return root


# a = [1,2,3,4,5,6,7,8,9,10,11]
# t = create_minimal_tree(a, 0, len(a)-1)
#
# tree = BSTree(t)
# tree.inorder_no_helper(t)

""" Given a binary tree return a list of nodes at each depth eg [[10], [6,15], ....]"""


def nodes_by_depth(node, d = -1,d_map=None):
    if node is None:
        return

    if d_map is None:
        d_map = {}

    d += 1
    if d in d_map:
        d_map[d].append(node.val)
    else:
        d_map[d] = [node.val]

    nodes_by_depth(node.left, d, d_map)
    nodes_by_depth(node.right, d, d_map)

    return d_map

# m = nodes_by_depth(t.root)
# print(m)

""" Given a binary tree check if it is balanced. i.e,
the difference of height of the left and right should not be more than 1"""


def is_balanced(node):
    if node is None:
        return -1
    else:
        left_height = 1 + is_balanced(node.left) if is_balanced(node.left)> -2 else -2
        right_height = 1 + is_balanced(node.right) if is_balanced(node.right)> -2 else -2

        if left_height != -2 and right_height != -2 and abs(left_height-right_height) < 2:
            return max(left_height, right_height)
        else:
            return -2

# a = is_balanced(t.root)
# print("Not Balanced") if a == -2 else print("Balanced")

"""Check if a binary tree is a binary search tree"""


import sys


def is_BST(node, max_val= sys.maxsize, min_val= -sys.maxsize-1):
    if node is None:
        return True

    if node.left is not None:
        left = node.left.val <= node.val and is_BST(node.left, node.val, min_val)
    else:
        left = True
    if node.right is not None:
        right = node.right.val > node.val and is_BST(node.right, max_val, node.val)
    else:
        right = True

    if left and right:
        return True
    else:
        return False


#print(is_BST(s.root))


""" Return the inorder successor of a given node. Assume a node has access to its parent node"""


def get_inorder_successor(node):
    if node.right is None:
        successor = node.parent
    else:
        successor = node.right
        while successor.left is not None:
            successor = successor.left
    return successor


"""Graph: Find order of execution of projects with dependencies on each other

For example: projects = [a,b,c,d,e,f]
Dependencies = [(a,d),(f,b),(b,d),(f,a),(d,c)] where 1st project in tuple should be done before the second
Output = [f,e,a,b,d,c] """


def get_order(projects, dependencies):
    depend_on_map = {}
    dependents_map = {}
    order = []

    for dependency in dependencies:
        if dependency[0] in dependents_map:
            dependents_map[dependency[0]].append(dependency[1])
        else:
            dependents_map[dependency[0]] = [dependency[1]]
        if dependency[1] in depend_on_map:
            depend_on_map[dependency[1]].append(dependency[0])
        else:
            depend_on_map[dependency[1]] = [dependency[0]]

    for project in projects:
        if project not in depend_on_map:
            order.append(project)

    for project in order:
        if project in dependents_map:
            for p in dependents_map[project]:
                depend_on_map[p].remove(project)
                if not depend_on_map[p]:
                    order.append(p)

    print(order)

#get_order(['a','b','c','d','e','f'], [('a','d'),('f','b'),('b','d'),('f','a'),('d','c')])


"""Given a binary tree provide all the lists of items that could have created it
For Example: Tree with 1<--2-->3 can be created by arrays [2,1,3] and [2,3,1]"""


from math import factorial

def get_lists(node):
    left, right = None, None

    if node.left is not None:
        left = get_lists(node.left)

    if node.right is not None:
        right = get_lists(node.right)

    l = []
    if right is None and left is None:
        l.append([node.val])
    elif right is None:
        for i in left:
            k = [node.val]
            for j in i:
                k.append(j)
            l.append(k)
    elif left is None:
        for i in right:
            k = [node.val]
            for j in i:
                k.append(j)
            l.append(k)
    else:
        for i in left:
            for m in right:
                k = [node.val]
                for j in i:
                    k.append(j)

                for n in m:
                    k.append(n)
                l.append(k)
        for i in right:
            for m in left:
                k = [node.val]
                for j in i:
                    k.append(j)

                for n in m:
                    k.append(n)
                l.append(k)
    return l


def get_list_by_level(node):
    level_map = nodes_by_depth(node)

    levels = 1 + max(level_map.keys())
    num_of_lists = 1
    for key in range(levels):
        num_of_lists *= factorial(len(level_map[key]))
    return num_of_lists


# a = get_lists(t.root)
# print(len(a), get_list_by_level(t.root))
# for i in a:
#     print(i)


"""Find common ancestor of two nodes in a binary tree
if tree has c-->a-->b then a is the common ancestor of a and b,
if one of the nodes is not in tree then return the other node
if both the nodes are not there or tree is empty return null"""


def find_ancestor(node, a, b):
    if node is None:
        return None
    if node == a or node == b:
        return node

    left = find_ancestor(node.left, a, b)
    right = find_ancestor(node.right, a, b)

    if left is not None and right is not None:
        return node

    if left is None and right is None:
        return None

    return left or right

# a, b = n5, n14
# n = find_ancestor(s.root, a, b)
# print(a.val, b.val)
# print(n.val if n is not None else None)


"""Given two binary trees check if one a subtree of another
If we can check that the root node of smaller tree is present in the bigger tree, it would suffice
but only if the nodes are same and not if the nodes are not same but has same data stored in them

a= Node(5), b = Node(5) then a and b are identical, hence we need to check the data in nodes instead of nodes"""

# if nodes used in the two trees are same that is to find if b is a subtree of a-->b


def check_for_subtree(node1, node2):
    if node2 is None:
        return True

    if node1 is None:
        return False

    if node1 == node2:
        return True

    left = check_for_subtree(node1.left, node2)
    right = check_for_subtree(node1.right, node2)

    if left or right:
        return True

    return False

# if nodes are different but data is same that is a = Node(4), b = Node(5), c = Node(4),
# then a-->b is identical to c-->b


def is_identical(node1, node2):
    if node1 is None and node2 is None:
        return True
    if node1 is not None and node2 is not None:
        return node1.val == node2.val and is_identical(node1.left, node2.left) and is_identical(node1.right, node2.right)

    return False


def find_similar_root_node(node1, node2):
    if node2 is None:
        return True
    if node1 is None:
        return False
    if node1.val == node2.val:
        return is_identical(node1, node2)

    return find_similar_root_node(node1.left, node2) or find_similar_root_node(node1.right, node2)


# m1 = Node(17)
# m2 = Node(19)
# m3 = Node(20)
# m4 = Node(22)
# t1 = BinaryTree()
# t1.add_node(m1)
# m1.left = m2
# m1.right = m3
# m3.left = m4
#
# print(find_similar_root_node(s.root, t1.root))


"""Implement a binary tree class with insert, find, delete, traverse methods and a method to return a random node.
 All the nodes should have same probability to be randomly returned.
 If the tree has a variable to keep a count of number of nodes in it, we can randomly pick a number and return
 the node at that index while traversing the tree. If we do not want a specific traversal then we may choose a random
 number from 1 to 4 specifying a traversal each (inorder, pre, post and breadth first)
"""
import random, queue


class BinaryTreeOwn:
    def __init__(self, node=None):
        self.root = node
        self.count = 0

    def add_node(self, n):
        self.count += 1
        if self.root is None:
            self.root = n

        else:  # get first node with less than 2 children in breadth first traversal and add node to it
            q = queue.Queue()
            q.put_nowait(self.root)

            while not q.empty():
                node = q.get_nowait()
                if node.left is not None and node.right is not None:
                    q.put_nowait(node.left)
                    q.put_nowait(node.right)
                elif node.left is None:
                    node.left = n
                    break
                elif node.right is None:
                    node.right = n
                    break

    def find_node(self, node, n): # initially node = root
        if node is None:
            return False
        if node == n:
            return True

        return self.find_node(node.left, n) or self.find_node(node.right, n)

    def breadth_first_traversal(self):
        if self.root is None:
            return
        a = []
        q = queue.Queue()
        q.put_nowait(self.root)
        while not q.empty():
            node = q.get_nowait()
            a.append(node.val)

            if node.left is not None:
                q.put_nowait(node.left)
            if node.right is not None:
                q.put_nowait(node.right)

        print(a)

    def random_node(self):
        if self.root is None:
            return

        c = random.randint(1,self.count)
        q = queue.Queue()
        q.put_nowait(self.root)
        node = None
        while not q.empty() and c > 0:
            c -= 1
            node = q.get_nowait()
            if node.left is not None:
                q.put_nowait(node.left)
            if node.right is not None:
                q.put_nowait(node.right)

        print(node.val)


b = BinaryTreeOwn()
m1 = Node(1)
m2 = Node(2)
m3 = Node(3)
m4 = Node(4)
m5 = Node(5)

b.add_node(m1)
b.add_node(m2)
b.add_node(m3)
b.add_node(m4)
b.breadth_first_traversal()
b.random_node()