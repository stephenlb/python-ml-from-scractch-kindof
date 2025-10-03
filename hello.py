Mono = "vim? is it worth it.YES!!!"
print(f"VSCode vs {Mono}")

## Future Stream Idea
##Kaushik Dwivedi
### List, Tuples, Sets

### Default lanaguages that jobs expect
# JavaScript Python SQL
## C++ Java C# Ruby


### most Common Stack
#PERN
#PostgresQL Express React node

## Naitik Yadav = Metasploit

#python dunder methods

#dunder = double underscore

## Getter and Setter
## OOP - update and retrieve values


### METASPLOIT
### Metaploit is a penetration testing framework that makes hacking simple.

### Manish Gaurav -
### Linked List in Python
#SHIFT+V
class Node(object):
    ## METHOD 
    def __init__(self, value):
        self.value = value
        self.next = None

    def __str__(self):
        return \
            f"Node: {self.value}"

## Linked List
class NodeList():
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, node):
        if not self.head:
            self.head = node
            self.tail = self.head
        else:
            self.tail.next = node
            self.tail = node

    def map(self, fn):
        current = self.head
        while (current):
            fn(current)
            current = current.next

nodeA = Node("A")
nodeB = Node("B")

nodeList = NodeList()
nodeList.append(nodeA)
nodeList.append(nodeB)

nodeList.map(print)
