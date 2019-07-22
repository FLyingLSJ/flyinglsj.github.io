http://funhacks.net/explore-python/Class/super.html

super 函数会继承父类的方法

```
class Animal(object):
    def __init__(self, name):
        self.name = name

    def greet(self):
        print('Hello, I am %s.' % self.name)


class Dog(Animal):
    def greet(self):
        super(Dog, self).greet()  # Python3 可使用 super().greet()
        print('WangWang...')


Dog('huang').greet()
# Hello, I am huang.
# WangWang...
```

