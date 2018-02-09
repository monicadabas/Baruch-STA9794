import sys

# Stack implementation using Linked List


class Node:
    def __init__(self, val, node = None):
        self.val = val
        self.next = node
        self.min = None # implement a min function to return minimum value in stack in O(1) time


class Stack:
    def __init__(self):
        self.head = None

    def push(self, x): # add Node at beg of linked list
        if self.head is None:
            self.head = Node(x)
            self.head.min = x
            return

        curr = Node(x)
        # if x < self.head.min:
        #     curr.min = x
        # else:
        #     curr.min = self.head.min
        curr.next = self.head
        self.head = curr

        return

    def pop(self): # remove head of linked list
        if self.head is None:
            print("Empty stack")
            return
        x = self.head
        self.head = self.head.next
        return x.val

    def peek(self): # return the value of head node
        if self.head is None:
            #print("Empty stack")
            return

        return self.head.val

    def print_stack(self):
        if self.head is None:
            print("Empty stack")
            return

        curr = self.head
        while curr is not None:
            print(curr.val)
            curr = curr.next

    def min(self):
        if self.head is None:
            return "Empty Stack"
        return self.head.min


# Stack implementation using Array


class StackArray:
    def __init__(self):
        self.last_index = -1
        self.array = []
        self.min = sys.maxsize

    def push(self, x):
        self.array.append(x)
        self.last_index += 1
        if x < self.min:
            self.min = x

    def pop(self):
        if self.last_index == -1:
            print("Empty Stack")
            return

        del self.array[self.last_index]
        self.last_index -= 1
        return

    def peek(self):
        return self.array[self.last_index]

    def print_stack(self):
        for i in range(self.last_index, -1, -1):
            print(self.array[i])


# Implement 3 stacks using a single array

class Stack3Array1:
    def __init__(self, n): # n is the size of array
        self.n = n
        self.s1_start = -1
        self.s1_end = -1 # int(n/3) -1
        self.s2_start = int(n/3) - 1 # int(n/3)
        self.s2_end = int(n/3) - 1  # int(2*n/3) -1
        self.s3_start = int(2*n/3) - 1 # int(2*n/3)
        self.s3_end = int(2*n/3) - 1 # n-1
        self.array = []

    def push_helper(self, x, start, end, last):
        if end < last:
            self.array[last+1] = x
            last += 1

        else:
            print("Stack full. Cannot push element")

        return last

    def push(self, s, x):
        if s == 1:
            self.s1_end = self.push_helper(x, self.s1_start, self.s1_end, int(self.n/3)-1)
        elif s == 2:
            self.s2_end = self.push_helper(x, self.s2_start, self.s2_end, int(2*self.n/3) -1)
        elif s == 3:
            self.s3_end = self.push_helper(x, self.s3_start, self.s3_end, self.n-1)
        else:
            print("incorrect stack number")
        return

    def pop_helper(self, start, end):
        if end > start:
            del self.array[end]
            end -= 1
        else:
            print("Empty Stack")
        return end

    def pop(self, s):
        if s == 1:
            self.s1_end = self.pop_helper(self.s1_start, self.s1_end)
        elif s == 2:
            self.s2_end = self.pop_helper(self.s2_start, self.s2_end)
        elif s == 3:
            self.s3_end = self.pop_helper(self.s3_start, self.s3_end)
        else:
            print("Incorrect stack number")
        return


# Implement a stack of plates: each stack has a limited height, once full
# another stack starts but in all it works as a single big stack

class SetOfStacks:
    def __init__(self, max_height):
        self.no_of_stacks = 0
        self.height_of_last_stack = 0
        self.m = max_height
        self.array = [[None]*self.m]

    def push(self, x):
        if self.height_of_last_stack < self.m:
            if self.array[0][0] is None:
                self.no_of_stacks += 1
            self.array[self.no_of_stacks-1][self.height_of_last_stack] = x
            self.height_of_last_stack += 1

        else:
            self.array.append([None]*self.m)
            self.height_of_last_stack = 0
            self.no_of_stacks += 1
            self.array[self.no_of_stacks-1][self.height_of_last_stack] = x
            self.height_of_last_stack += 1

        return

    def print_stack(self):
        for s in range(self.height_of_last_stack-1, -1, -1):
            print(self.array[self.no_of_stacks-1][s])

        for s in range(self.no_of_stacks-2, -1, -1):
            for i in range(self.m-1, -1, -1):
                print(self.array[s][i])

    def pop_element(self):
        if self.array[0][0] is None:
            print("Empty Stack")
            return

        self.array[self.no_of_stacks-1][self.height_of_last_stack-1] = None
        self.height_of_last_stack -= 1
        if self.height_of_last_stack == 0:
            del self.array[self.no_of_stacks-1]

        return

    # since it will create substacks not filled till capacity, we will for now put its value as None

    def pop_at(self, index):
        if index in range((self.no_of_stacks-1)*self.m + self.height_of_last_stack-1):
            r = int((index+1)/4)
            c = index%4
            self.array[r][c] = None

        else:
            print("Index out of range")
        return


# Implement a queue using 2 stacks
# One stack will be used to push element and another to pop

class Queue_2_Stacks:
    def __init__(self):
        self.push = Stack()
        self.pop = Stack()

    def enqueue(self, x):
        self.push.push(x)

    def dequeue(self):
        if self.pop.head is None:
            if self.push.head is not None:
                current = self.push.head
                while current is not None:
                    self.pop.push(current.val)
                    self.push.head = current.next
                    current = self.push.head
            else:
                print("Empty queue")

        self.pop.pop()

    def push_stack_print_helper(self, node):
        current = node
        if current is not None:
            self.push_stack_print_helper(current.next)
        else:
            return
        print(current.val)

    def print_queue(self):
        if self.pop.head is not None:
            current = self.pop.head
            while current is not None:
                print(current.val)
                current = current.next

        if self.push.head is not None:
            self.push_stack_print_helper(self.push.head)

        if self.pop.head is None and self.push.head is None:
            print("Empty queue")

    def peek_queue(self):
        if self.pop.head is None:
            if self.push.head is not None:
                current = self.push.head
                while current is not None:
                    self.pop.push(current.val)
                    self.push.head = current.next
                    current = self.push.head
            else:
                return

        return self.pop.head.val




# Sort a stack such that smallest items are on the top without using any other data structure

def sort_stack(s=Stack()):
    if s.head is None:
        print("Empty Stack")
        return

    s_output = Stack()

    while s.peek() is not None:
        if s_output.head is None:
            s_output.push(s.pop())

        else:
            buff = s.pop()
            while s_output.peek() is not None and s_output.peek() < buff:
                s.push(s_output.pop())
            s_output.push(buff)

    s_output.print_stack()


# Implement an Animal shelter which has dogs and cats only,
# from which people can adopt only the oldest (first in queue) animal or the oldest dog or cat

class Animal:
    def __init__(self, pet_type, age_in_queue=None):
        self.pet_type = pet_type # dog = 1, cat = 2
        self.index = age_in_queue

class AnimalShelter:
    def __init__(self):
        self.count = 0  # index of the animal in shelter to tell which one came first
        self.dog_queue_head = None
        self.dog_queue_last = None
        self.cat_queue_head = None
        self.cat_queue_last = None

    def enqueue(self, Animal_instance):
        if Animal_instance.pet_type not in [1,2]:
            print("Incorrect Animal type")
            return

        self.count += 1
        Animal_instance.index = self.count
        n = Node(Animal_instance)

        if Animal_instance.pet_type == 1: # is dog
            if self.dog_queue_head is None:
                self.dog_queue_head = n
                self.dog_queue_last = n
            else:
                self.dog_queue_last.next = n
                self.dog_queue_last = n

        elif Animal_instance.pet_type == 2: # is cat
            if self.cat_queue_head is None:
                self.cat_queue_head = n
                self.cat_queue_last = n
            else:
                self.cat_queue_last.next = n
                self.cat_queue_last = n

    def dequeue_no_choice(self):
        if self.dog_queue_head is None and self.cat_queue_head is None:
            print("No Animal in shelter")
            return

        if self.dog_queue_head is None:
            pet_out = self.cat_queue_head
            self.cat_queue_head = self.cat_queue_head.next

        elif self.cat_queue_head is None:
            pet_out = self.dog_queue_head
            self.dog_queue_head = self.dog_queue_head.next

        else:
            if self.dog_queue_head.val.index < self.cat_queue_head.val.index:
                pet_out = self.dog_queue_head
                self.dog_queue_head = self.dog_queue_head.next
            else:
                pet_out = self.cat_queue_head
                self.cat_queue_head = self.cat_queue_head.next

        return pet_out.val

    def dequeue_choice(self, choice):
        if choice not in [1,2]:
            print("Incorrect Animal type")
            return

        if choice == 1:
            if self.dog_queue_head is not None:
                pet_out = self.dog_queue_head
                self.dog_queue_head = self.dog_queue_head.next
            else:
                print("No dog")
                return

        else:
            if self.cat_queue_head is not None:
                pet_out = self.cat_queue_head
                self.cat_queue_head = self.cat_queue_head.next
            else:
                print("No cat")
                return

        return pet_out.val.pet_type, pet_out.val.index

    def dequeue_dog(self):
        if self.dog_queue_head is not None:
            pet_out = self.dog_queue_head
            self.dog_queue_head = self.dog_queue_head.next
            return pet_out.val.pet_type, pet_out.val.index

        print("No dog")
        return

    def dequeue_cat(self):
        if self.cat_queue_head is not None:
            pet_out = self.cat_queue_head
            self.cat_queue_head = self.cat_queue_head.next
            return pet_out.val

        print("No cat")
        return

    def print_queue(self):
        if self.dog_queue_head is None and self.cat_queue_head is None:
            print("No Animal in shelter")
            return

        if self.cat_queue_head is None:
            current = self.dog_queue_head

            while current is not None:
                print(current.val.pet_type, current.val.index)
                current = current.next

        elif self.dog_queue_head is None:
            current = self.cat_queue_head
            while current is not None:
                print(current.val.pet_type, current.val.index)
                current = current.next

        else:
            current_dog = self.dog_queue_head
            current_cat = self.cat_queue_head
            while current_cat is not None or current_dog is not None:
                if current_dog is not None and current_cat is not None:
                    if current_cat.val.index < current_dog.val.index:
                        print(current_cat.val.pet_type, current_cat.val.index)
                        current_cat = current_cat.next
                    else:
                        print(current_dog.val.pet_type, current_dog.val.index)
                        current_dog = current_dog.next
                elif current_cat is None:
                    print(current_dog.val.pet_type, current_dog.val.index)
                    current_dog = current_dog.next

                elif current_dog is None:
                    print(current_cat.val.pet_type, current_cat.val.index)
                    current_cat = current_cat.next


dog1 = Animal(1)
dog2 = Animal(1)
dog3 = Animal(1)
dog4 = Animal(1)
dog5 = Animal(1)
cat1 = Animal(2)
cat2 = Animal(2)
cat3 = Animal(2)
cat4 = Animal(2)

# AS = AnimalShelter()
# AS.enqueue(dog1)
# AS.enqueue(cat1)
# AS.enqueue(cat2)
# AS.enqueue(dog2)
# AS.enqueue(dog3)
# AS.enqueue(cat3)
# AS.enqueue(cat4)
# AS.enqueue(dog4)
#
# AS.print_queue()


def TowerOfHanoi(n, A, C=Stack(), B=Stack()):
    if n < 1: return
    if n == 1:
        C.push(A.pop())

    TowerOfHanoi(n-1, A, B, C)
    TowerOfHanoi(1, A, C, B)
    TowerOfHanoi(n-1, B, C, A)


s = Stack()
s.push(4)
s.push(3)
s.push(2)
s.push(1)

s.print_stack()
TowerOfHanoi(4, s)









