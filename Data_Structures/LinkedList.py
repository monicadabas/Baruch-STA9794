

class Node:
    def __init__(self, val=None, next_node=None):
        self.val = val
        self.next = next_node


class LinkedList:
    def __init__(self, head=None):
        self.head = head

    def add_node_at_end(self, node):
        if self.head is None:
            self.head = node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = node

    def add_node_at_beg(self, node):
        if self.head is None:
            self.head = node
        else:
            node.next = self.head
            self.head = node

    def remove_item(self, item):
        if self.head is None:
            return

        if self.head.val == item:
            self.head = self.head.next
            return

        current = self.head
        while current.next is not None:
            if current.next.val == item:
                current.next = current.next.next

            else:
                current = current.next

        return

    def print_ll(self):
        if self.head is None:
            print("Empty LL")
            return

        current = self.head
        while current is not None:
            print(current.val)
            current = current.next
        return

    def find_length(self):
        current = self.head
        count = 0
        while current is not None:
            count += 1
            current = current.next
        return count

    def remove_dups(self):
        values = {}
        if self.head is None:
            return

        current = self.head
        values[current.val] = 1
        while current.next is not None:
            if current.next.val in values:
                current.next = current.next.next
            else:
                values[current.next.val] = 1
                current = current.next
        return

    def k_th_to_last_element(self, k):
        if self.head is None:
            print("Empty LL")
            return

        length = self.find_length()
        if type(k) is not int or 0 > k > length:
            print("Incorrect K value")
            return

        n = length - k + 1
        count = 0
        current = self.head

        while current is not None:
            count += 1
            if count == n:
                return current.val
            current = current.next

        return

    def delete_middle_node(self, node): # given access to only that node
        current = node
        while current.next is not None:
            current.val = current.next.val
            if current.next.next is not None:
                current = current.next
            else:
                current.next = None


# rearrange a linked list around a value x such that all nodes
# with value less than x comes to the left of the values equal or higher than x
# 3->5->8->5->10->2->1 (x = 5) to 3->2->1->any arrangement of the rest

def partition_ll(l, x):
    left_l = LinkedList()
    right_l = LinkedList()

    current = l.head
    while current is not None:
        if current.val < x:
            left_l.add_node_at_end(Node(current.val))
        else:
            right_l.add_node_at_end(Node(current.val))
        current = current.next

    left_l.add_node_at_end(right_l.head)

    return left_l


# sum of two numbers. Numbers are stored in two linked lists with 1's digit as head
# return the linked list with sum stored in a linked list in same format
# 1->2 + 3->4 = 4->6


def sum_forward(l1, l2):
    len_l1 = l1.find_length()
    len_l2 = l2.find_length()

    if len_l1 == 0 and len_l2 == 0:
        print("No linked list")
        return

    len_diff = abs(len_l1 - len_l2)

    if len_l1 > len_l2:
        for i in range(len_diff):
            l2.add_node_at_end(Node(0))
    else:
        for i in range(len_diff):
            l1.add_node_at_end(Node(0))

    l = LinkedList()

    curr_1 = l1.head
    curr_2 = l2.head
    carry = 0

    while curr_1 is not None:
        sum = curr_1.val + curr_2.val + carry
        l.add_node_at_end(Node(sum%10))
        carry = int(sum/10)
        curr_1 = curr_1.next
        curr_2 = curr_2.next

    if carry > 0:
        l.add_node_at_end(Node(carry))

    return l


# What if the numbers are stored backwards? 1's digit at the end


def sum_backward_helper(curr_1, curr_2, carry, l):
    if curr_1.next is not None:
        l, carry = sum_backward_helper(curr_1.next, curr_2.next, carry, l)

    sum = (curr_1.val + curr_2.val + carry)%10
    carry = int((curr_1.val + curr_2.val + carry)/10)

    l.add_node_at_beg(Node(sum))

    return l, carry


def sum_backward(l1, l2):
    len_l1 = l1.find_length()
    len_l2 = l2.find_length()

    if len_l1 == 0 and len_l2 == 0:
        print("No linked list")
        return

    len_diff = abs(len_l1 - len_l2)

    if len_l1 > len_l2:
        for i in range(len_diff):
            l2.add_node_at_beg(Node(0))
    else:
        for i in range(len_diff):
            l1.add_node_at_beg(Node(0))

    l = LinkedList()

    curr_1 = l1.head
    curr_2 = l2.head
    carry = 0

    l, carry = sum_backward_helper(curr_1, curr_2, carry, l)

    if carry > 0:
        l.add_node_at_beg(Node(carry))

    return l


# Check if a linked list is a palindrome

def is_palindrome_helper(curr_1, curr_2):
    if curr_2.next is not None:
        boool , curr_1 = is_palindrome_helper(curr_1, curr_2.next)

    if curr_1.val == curr_2.val:
        curr_1 = curr_1.next
        return True, curr_1

    return False, curr_1


def is_palindrome(l):
    if l.head is None:
        return "Empty list"

    curr_1 = l.head
    curr_2 = l.head

    bool_, curr = is_palindrome_helper(curr_1, curr_2)
    return bool_


# Check if two linked lists intersect. i.e., if they share common nodes


def does_intersect(l1, l2):
    len_l1 = l1.find_length()
    len_l2 = l2.find_length()

    curr_1 = l1.head
    curr_2 = l2.head

    if len_l1 == 0 and len_l2 == 0:
        print("No linked list")
        return

    len_diff = abs(len_l1 - len_l2)

    if len_l1 > len_l2:
        for i in range(len_diff):
            curr_2 = curr_2.next
    else:
        for i in range(len_diff):
            curr_1 = curr_1.next



    while curr_1 is not None:
        if curr_1 == curr_2:
            return True, curr_1, curr_1.val, curr_1.next

        curr_1 = curr_1.next
        curr_2 = curr_2.next

    return False


# return the node at the beginning of a loop in a circular linked list


def loop_detection(l):
    if l.head is None:
        return "Empty list"

    curr_1 = l.head
    curr_2 = l.head
    count = 0

    while count == 0:
        if curr_2 is None or curr_2.next is None or curr_2.next.next is None:
            return "Not a loop"

        curr_1 = curr_1.next
        curr_2 = curr_2.next.next

        if curr_1 == curr_2:
            count += 1

    curr_1 = l.head
    while curr_1 != curr_2:
        curr_1 = curr_1.next
        curr_2 = curr_2.next

    return curr_1, curr_1.val


l = LinkedList()

n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
n4 = Node(4)
n5 = Node(5)

l.add_node_at_end(n1)
l.add_node_at_end(n2)
l.add_node_at_end(n3)
l.add_node_at_end(n4)
l.add_node_at_end(n5)


def reverse_ll(head):
    if head is None:
        return
    current = head

    if current.next is None:
        return LinkedList(current), current

    ll, tail = reverse_ll(current.next)

    current.next = None
    tail.next = current

    return ll, current

# reverse_ll(l.head)[0].print_ll()




